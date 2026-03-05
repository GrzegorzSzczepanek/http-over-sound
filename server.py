#!/usr/bin/env python3
"""
SoundHTTP Server — serwer HTTP over Sound

Nasłuchuje na mikrofonie, dekoduje żądania FSK,
przetwarza je i odpowiada dźwiękiem przez głośnik.

Użycie:
  python server.py --live                         # nasłuchuj na mikrofonie
  python server.py --live --mic 1 --speaker 0     # wybierz urządzenia
  python server.py --simulate request.wav         # tryb bez mikrofonu
  python server.py --list-devices                 # lista urządzeń audio

Scenariusz:
  Komputer A (serwer):  python server.py --live
  Komputer B (klient):  python client.py GET /ping
  Klient odtwarza request przez głośnik → serwer dekoduje z mikrofonu →
  serwer odtwarza response przez głośnik → klient dekoduje z mikrofonu.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from datetime import datetime

from protocol import (
    FRAME_REQUEST, FRAME_RESPONSE,
    METHOD_GET, METHOD_POST, METHOD_PUT, METHOD_DELETE,
    METHOD_NAMES, METHOD_FROM_NAME,
    http_to_compact, compact_to_http,
)
from modem import (
    SAMPLE_RATE, PREAMBLE,
    calc_steps, find_preamble, decode_request, encode_response,
    write_wav, read_wav, play_audio, list_audio_devices,
    detect_nibble, estimate_response_duration,
)


# ══════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════

class SoundRouter:
    """Prosty router HTTP — rejestruj handlery jak w Flask."""

    def __init__(self):
        self.routes: dict[tuple[int, str], callable] = {}
        self.storage: dict[str, any] = {}

    def route(self, path: str, methods: list[str] | None = None):
        if methods is None:
            methods = ["GET"]

        def decorator(func):
            for m in methods:
                code = METHOD_FROM_NAME.get(m.upper())
                if code is not None:
                    self.routes[(code, path)] = func
            return func
        return decorator

    def handle(self, method: int, path: str, body: bytes) -> tuple[int, bytes]:
        handler = self.routes.get((method, path))

        if handler is None:
            any_method = any(p == path for (_, p) in self.routes)
            if any_method:
                return http_to_compact(405), json.dumps(
                    {"error": "Method Not Allowed"}).encode()
            return http_to_compact(404), json.dumps(
                {"error": "Not Found", "path": path}).encode()

        try:
            status, response = handler(path=path, body=body, storage=self.storage)
            if isinstance(response, dict):
                response = json.dumps(response, ensure_ascii=False).encode()
            elif isinstance(response, str):
                response = response.encode()
            return http_to_compact(status), response
        except Exception as e:
            return http_to_compact(500), json.dumps({"error": str(e)}).encode()


# ══════════════════════════════════════════════════════════════════════
#  DOMYŚLNE ENDPOINTY
# ══════════════════════════════════════════════════════════════════════

def setup_default_routes(router: SoundRouter):

    @router.route("/", methods=["GET"])
    def index(path, body, storage):
        return 200, {
            "server": "SoundHTTP/1.0",
            "msg": "HTTP over Sound!",
            "routes": ["/", "/ping", "/time", "/echo", "/store", "/msg"],
        }

    @router.route("/ping", methods=["GET"])
    def ping(path, body, storage):
        return 200, {"pong": True, "ts": int(time.time())}

    @router.route("/time", methods=["GET"])
    def get_time(path, body, storage):
        now = datetime.now()
        return 200, {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
        }

    @router.route("/echo", methods=["POST"])
    def echo(path, body, storage):
        try:
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {"raw": body.hex()}
        return 200, {"echo": data}

    @router.route("/msg", methods=["POST"])
    def message(path, body, storage):
        """Endpoint do wysyłania wiadomości tekstowych."""
        try:
            data = json.loads(body) if body else {}
            text = data.get("text", "")
            sender = data.get("from", "anonymous")
            if "messages" not in storage:
                storage["messages"] = []
            msg = {
                "from": sender,
                "text": text,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            storage["messages"].append(msg)
            print(f"\n  >>> Wiadomosc od {sender}: {text}\n")
            return 200, {"ok": True, "received": msg}
        except Exception as e:
            return 400, {"error": str(e)}

    @router.route("/msg", methods=["GET"])
    def get_messages(path, body, storage):
        msgs = storage.get("messages", [])
        return 200, {"messages": msgs, "count": len(msgs)}

    @router.route("/store", methods=["GET"])
    def store_list(path, body, storage):
        keys = [k for k in storage if k != "messages"]
        return 200, {"keys": keys, "count": len(keys)}

    @router.route("/store", methods=["POST"])
    def store_set(path, body, storage):
        try:
            data = json.loads(body)
            key = data.get("key", "")
            value = data.get("value", "")
            if not key:
                return 400, {"error": "missing 'key'"}
            storage[key] = value
            return 201, {"stored": key, "value": value}
        except Exception as e:
            return 400, {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════
#  SERWER
# ══════════════════════════════════════════════════════════════════════

class SoundHTTPServer:

    def __init__(self, router: SoundRouter | None = None):
        self.router = router or SoundRouter()
        self.running = False
        self.request_count = 0

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts}] {msg}")

    def process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Dekoduj żądanie z audio, przetwórz przez router, zwróć audio odpowiedzi."""
        self.log("Dekodowanie zadania...")

        request = decode_request(audio, sr)
        if request is None:
            self.log("[x] Nie udalo sie zdekodowac zadania")
            return None

        self.request_count += 1
        method_name = request["method_name"]
        path = request["path"]
        crc = "CRC OK" if request["crc_ok"] else "CRC FAIL"

        body_info = ""
        if request["body"]:
            try:
                body_info = f'  body="{request["body"].decode("utf-8")[:80]}"'
            except UnicodeDecodeError:
                body_info = f"  body=<{len(request['body'])}B>"

        self.log(f"<-- #{self.request_count} {method_name} {path}{body_info}  [{crc}]")

        status_compact, response_body = self.router.handle(
            request["method"], request["path"], request["body"]
        )

        http_code, status_text = compact_to_http(status_compact)
        self.log(f"--> {http_code} {status_text} ({len(response_body)}B)")

        try:
            preview = response_body.decode("utf-8")[:120]
            self.log(f"    {preview}")
        except UnicodeDecodeError:
            pass

        return encode_response(status_compact, response_body, sr)

    # ── Tryb simulate ──

    def serve_simulate(self, wav_path: str, response_wav: str | None = None):
        """Przetwórz żądanie z pliku WAV i zapisz odpowiedź."""
        print(f"\n  SoundHTTP Server  [simulate]")
        print(f"  {'=' * 50}")

        audio, sr = read_wav(wav_path)
        self.log(f"Wczytano: {wav_path} ({len(audio)/sr:.2f}s)")

        response_audio = self.process_audio(audio, sr)
        if response_audio is not None:
            out = response_wav or wav_path.replace(".wav", "_response.wav")
            write_wav(out, response_audio, sr)
            self.log(f"Odpowiedz zapisana: {out} ({len(response_audio)/sr:.2f}s)")
            return out
        return None

    # ── Tryb live ──

    def serve_live(self, mic_index: int | None = None,
                   speaker_index: int | None = None,
                   sr: int = SAMPLE_RATE):
        """
        Nasłuchuj na mikrofonie, dekoduj żądania, odpowiadaj przez głośnik.
        """
        try:
            import pyaudio
        except ImportError:
            print("  [x] brak pyaudio: pip install pyaudio")
            sys.exit(1)

        self.running = True
        pa = pyaudio.PyAudio()

        mic_name = "default"
        spk_name = "default"
        if mic_index is not None:
            mic_name = pa.get_device_info_by_index(mic_index)["name"]
        if speaker_index is not None:
            spk_name = pa.get_device_info_by_index(speaker_index)["name"]

        print(f"\n  +{'=' * 50}+")
        print(f"  |  SoundHTTP Server  [live mode]                 |")
        print(f"  +{'=' * 50}+")
        print(f"  |  Mikrofon : {mic_name:<37}|")
        print(f"  |  Glosnik  : {spk_name:<37}|")
        print(f"  |  SR       : {sr:<37}|")
        print(f"  +{'=' * 50}+")
        print(f"  Nasluchuje... (Ctrl+C aby zatrzymac)\n")

        CHUNK = int(sr * 0.05)  # 50ms chunki
        BUFFER_SEC = 30
        buffer_size = int(sr * BUFFER_SEC)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=CHUNK,
        )

        audio_buffer = np.zeros(buffer_size, dtype=np.float64)
        write_pos = 0
        last_request_time = 0
        COOLDOWN = 3.0
        energy_history = []
        ENERGY_WINDOW = 20

        try:
            while self.running:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0

                # Ring buffer
                end = write_pos + len(chunk)
                if end <= buffer_size:
                    audio_buffer[write_pos:end] = chunk
                else:
                    first = buffer_size - write_pos
                    audio_buffer[write_pos:] = chunk[:first]
                    audio_buffer[:len(chunk) - first] = chunk[first:]
                write_pos = end % buffer_size

                # Śledzenie energii
                rms = np.sqrt(np.mean(chunk ** 2))
                energy_history.append(rms)
                if len(energy_history) > ENERGY_WINDOW:
                    energy_history.pop(0)

                if rms < 0.008:
                    continue

                now = time.time()
                if now - last_request_time < COOLDOWN:
                    continue

                avg_energy = np.mean(energy_history)
                if avg_energy < 0.01:
                    continue

                # Zbierz ostatnie sekundy z ring buffera
                lookback = int(sr * 8)
                if write_pos >= lookback:
                    segment = audio_buffer[write_pos - lookback:write_pos].copy()
                else:
                    segment = np.concatenate([
                        audio_buffer[buffer_size - (lookback - write_pos):],
                        audio_buffer[:write_pos]
                    ]).copy()

                # Szukaj preambuły
                preamble_pos = find_preamble(segment, sr)
                if preamble_pos is None:
                    continue

                last_request_time = now
                self.log("Preambula wykryta! Czekam na pelne zadanie...")

                # Poczekaj na resztę transmisji
                time.sleep(2.5)

                # Dozbieraj dane
                extra_chunks = []
                for _ in range(int(sr * 1.5 / CHUNK)):
                    try:
                        extra_raw = stream.read(CHUNK, exception_on_overflow=False)
                        extra_chunks.append(
                            np.frombuffer(extra_raw, dtype=np.int16).astype(np.float64) / 32767.0
                        )
                    except Exception:
                        break

                if extra_chunks:
                    full_audio = np.concatenate([segment] + extra_chunks)
                else:
                    full_audio = segment

                response_audio = self.process_audio(full_audio, sr)

                if response_audio is not None:
                    self.log("Odtwarzam odpowiedz przez glosnik...")
                    time.sleep(0.5)

                    out_stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sr,
                        output=True,
                        output_device_index=speaker_index,
                    )
                    out_data = np.clip(
                        response_audio * 32767, -32768, 32767
                    ).astype(np.int16)
                    out_stream.write(out_data.tobytes())
                    out_stream.close()

                    dur = len(response_audio) / sr
                    self.log(f"Odpowiedz wyslana ({dur:.1f}s)")
                    last_request_time = time.time()
                else:
                    self.log("Nie udalo sie zdekodowac — czekam dalej...")

        except KeyboardInterrupt:
            print(f"\n\n  Server zatrzymany. Obsluzono {self.request_count} zadan.\n")
        finally:
            stream.close()
            pa.terminate()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SoundHTTP Server — HTTP over Sound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przyklad:
  python server.py --live
  python server.py --live --mic 1 --speaker 0
  python server.py --simulate request.wav
  python server.py --list-devices
        """,
    )

    parser.add_argument("--simulate", metavar="WAV",
                        help="Przetworz zadanie z pliku WAV")
    parser.add_argument("--response-wav", metavar="PATH",
                        help="Gdzie zapisac odpowiedz WAV (simulate)")
    parser.add_argument("--live", action="store_true",
                        help="Nasluchuj na mikrofonie (tryb live)")
    parser.add_argument("--mic", type=int, default=None,
                        help="Indeks mikrofonu")
    parser.add_argument("--speaker", type=int, default=None,
                        help="Indeks glosnika")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE,
                        help="Sample rate (domyslnie: 44100)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Pokaz urzadzenia audio")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    router = SoundRouter()
    setup_default_routes(router)
    server = SoundHTTPServer(router)

    if args.simulate:
        server.serve_simulate(args.simulate, args.response_wav)
    elif args.live:
        server.serve_live(args.mic, args.speaker, args.sr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
