#!/usr/bin/env python3
"""
SoundHTTP Client — klient HTTP over Sound

Wysyła żądania HTTP jako dźwięk FSK i odbiera odpowiedzi.

Użycie:
  python client.py GET /ping
  python client.py POST /echo --body '{"hello":"world"}'
  python client.py POST /msg --body '{"from":"Alice","text":"Czesc!"}'
  python client.py --interactive                  # tryb interaktywny (REPL)
  python client.py GET /ping --simulate           # zapisz WAV bez odtwarzania

Tryby:
  --live         Odtwórz request przez głośnik, nasłuchuj odpowiedzi na mikrofonie
  --simulate     Zapisz request jako WAV (bez audio — do testów)
  --interactive  REPL — wpisuj komendy po kolei

Scenariusz:
  Komputer A (serwer):  python server.py --live
  Komputer B (klient):  python client.py GET /ping
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
    METHOD_GET, METHOD_POST,
    METHOD_NAMES, METHOD_FROM_NAME,
    compact_to_http,
)
from modem import (
    SAMPLE_RATE, PREAMBLE,
    calc_steps, find_preamble,
    encode_request, decode_response,
    write_wav, read_wav,
    play_audio, record_audio, list_audio_devices,
    estimate_request_duration, estimate_response_duration,
    BYTES_PER_SEC,
    build_request_frame, parse_response_frame,
)
from packets import LiveSession


# ══════════════════════════════════════════════════════════════════════
#  KLIENT
# ══════════════════════════════════════════════════════════════════════

class SoundHTTPClient:
    """Klient HTTP over Sound."""

    def __init__(self, speaker_index: int | None = None,
                 mic_index: int | None = None,
                 sr: int = SAMPLE_RATE):
        self.speaker_index = speaker_index
        self.mic_index = mic_index
        self.sr = sr

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts}] {msg}")

    # ── Wysyłanie żądania (packet mode) ──

    def send_request(self, method_name: str, path: str,
                     body: bytes = b"") -> dict | None:
        """
        Wyślij żądanie przez głośnik jako pakiety z ACK/NAK
        i nasłuchuj odpowiedzi pakietowej na mikrofonie.
        """
        from protocol import METHOD_FROM_NAME
        import pyaudio

        method = METHOD_FROM_NAME.get(method_name.upper())
        if method is None:
            print(f"  [x] Nieznana metoda: {method_name}")
            return None

        # Zbuduj ramkę request
        request_frame = build_request_frame(method, path, body)
        body_info = f"  body={len(body)}B" if body else ""
        self.log(f"--> {method_name} {path}{body_info} ({len(request_frame)}B frame)")

        # Otwórz sesję pakietową
        pa = pyaudio.PyAudio()
        session = LiveSession(pa, self.sr, self.mic_index,
                              self.speaker_index, self.log)

        # Wyślij request jako pakiety
        self.log("Wysylam request pakietami...")
        success = session.send_data(request_frame)

        if not success:
            self.log("[x] Nie udalo sie wyslac requestu")
            pa.terminate()
            return None

        self.log("Request wyslany. Czekam na odpowiedz...")
        time.sleep(0.5)

        # Odbierz odpowiedź jako pakiety
        response_data = session.receive_data()
        pa.terminate()

        if response_data is None:
            self.log("[x] Nie odebrano odpowiedzi")
            return None

        # Parsuj ramkę response
        response = parse_response_frame(response_data)
        if response is None:
            self.log("[x] Nie udalo sie sparsowac odpowiedzi")
            return None

        crc = "CRC OK" if response["crc_ok"] else "CRC FAIL"
        self.log(f"<-- {response['http_code']} {response['status_text']}  [{crc}]")

        if response["body"]:
            try:
                body_text = response["body"].decode("utf-8")
                try:
                    body_json = json.loads(body_text)
                    print(f"\n  {json.dumps(body_json, indent=2, ensure_ascii=False)}\n")
                except json.JSONDecodeError:
                    print(f"\n  {body_text}\n")
            except UnicodeDecodeError:
                print(f"\n  <{len(response['body'])}B binary>\n")

        return response

    def _listen_for_response(self, timeout: float = 20.0) -> dict | None:
        """Nasłuchuj na mikrofonie i zdekoduj odpowiedź FSK (legacy, non-packet)."""
        try:
            import pyaudio
        except ImportError:
            print("  [x] brak pyaudio: pip install pyaudio")
            return None

        pa = pyaudio.PyAudio()
        sr = self.sr
        CHUNK = int(sr * 0.05)  # 50ms

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=CHUNK,
        )

        collected = []
        start_time = time.time()
        signal_detected = False
        silence_after_signal = 0
        SILENCE_THRESHOLD = 0.005
        SILENCE_CHUNKS_TO_STOP = 30  # 30 * 50ms = 1.5s ciszy po sygnale

        try:
            while time.time() - start_time < timeout:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
                collected.append(chunk)

                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > SILENCE_THRESHOLD:
                    if not signal_detected:
                        self.log("Sygnal wykryty!")
                    signal_detected = True
                    silence_after_signal = 0
                elif signal_detected:
                    silence_after_signal += 1
                    if silence_after_signal >= SILENCE_CHUNKS_TO_STOP:
                        self.log("Koniec sygnalu (cisza). Dekoduje...")
                        break

                # Co sekundę pokaz status
                elapsed = time.time() - start_time
                if len(collected) % 20 == 0 and not signal_detected:
                    remaining = timeout - elapsed
                    print(f"\r  Czekam na odpowiedz... ({remaining:.0f}s)   ", end="", flush=True)

        except KeyboardInterrupt:
            print("\n  Przerwano.")
            return None
        finally:
            stream.close()
            pa.terminate()

        if not signal_detected:
            print("\r  [x] Timeout — brak odpowiedzi od serwera.        ")
            return None

        # Dekoduj
        audio = np.concatenate(collected)
        self.log(f"Zebrano {len(audio)/sr:.1f}s audio. Dekoduje...")

        response = decode_response(audio, sr)
        if response is None:
            self.log("[x] Nie udalo sie zdekodowac odpowiedzi")
            return None

        crc = "CRC OK" if response["crc_ok"] else "CRC FAIL"
        self.log(f"<-- {response['http_code']} {response['status_text']}  [{crc}]")

        if response["body"]:
            try:
                body_text = response["body"].decode("utf-8")
                try:
                    body_json = json.loads(body_text)
                    print(f"\n  {json.dumps(body_json, indent=2, ensure_ascii=False)}\n")
                except json.JSONDecodeError:
                    print(f"\n  {body_text}\n")
            except UnicodeDecodeError:
                print(f"\n  <{len(response['body'])}B binary>\n")

        return response

    # ── Tryb simulate ──

    def simulate_request(self, method_name: str, path: str,
                         body: bytes = b"",
                         output_wav: str | None = None) -> str:
        """Zakoduj żądanie i zapisz jako WAV (bez odtwarzania)."""
        method = METHOD_FROM_NAME.get(method_name.upper())
        if method is None:
            print(f"  [x] Nieznana metoda: {method_name}")
            return ""

        request_audio = encode_request(method, path, body, self.sr)

        if output_wav is None:
            safe_path = path.replace("/", "_").strip("_") or "root"
            output_wav = f"request_{method_name.lower()}_{safe_path}.wav"

        write_wav(output_wav, request_audio, self.sr)
        dur = len(request_audio) / self.sr
        self.log(f"Zapisano: {output_wav} ({dur:.1f}s)")
        return output_wav

    # ── Tryb simulate: dekoduj odpowiedź z WAV ──

    def decode_response_wav(self, wav_path: str) -> dict | None:
        """Wczytaj WAV z odpowiedzią i zdekoduj."""
        audio, sr = read_wav(wav_path)
        self.log(f"Wczytano: {wav_path} ({len(audio)/sr:.2f}s)")

        response = decode_response(audio, sr)
        if response is None:
            self.log("[x] Nie udalo sie zdekodowac odpowiedzi")
            return None

        crc = "CRC OK" if response["crc_ok"] else "CRC FAIL"
        self.log(f"<-- {response['http_code']} {response['status_text']}  [{crc}]")

        if response["body"]:
            try:
                body_text = response["body"].decode("utf-8")
                try:
                    body_json = json.loads(body_text)
                    print(f"\n  {json.dumps(body_json, indent=2, ensure_ascii=False)}\n")
                except json.JSONDecodeError:
                    print(f"\n  {body_text}\n")
            except UnicodeDecodeError:
                print(f"\n  <{len(response['body'])}B binary>\n")

        return response


# ══════════════════════════════════════════════════════════════════════
#  INTERAKTYWNY REPL
# ══════════════════════════════════════════════════════════════════════

def interactive_mode(client: SoundHTTPClient):
    """Tryb interaktywny — wpisuj komendy HTTP."""
    print(f"\n  +{'=' * 50}+")
    print(f"  |  SoundHTTP Client  [interactive]                |")
    print(f"  +{'=' * 50}+")
    print(f"  |  Wpisz: METHOD /path [json_body]               |")
    print(f"  |  Przyklad:                                      |")
    print(f"  |    GET /ping                                    |")
    print(f"  |    POST /echo {{\"hello\":\"world\"}}                 |")
    print(f"  |    POST /msg {{\"from\":\"Alice\",\"text\":\"Hi!\"}}     |")
    print(f"  |    quit                                         |")
    print(f"  +{'=' * 50}+\n")

    while True:
        try:
            line = input("  sound> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!\n")
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("  Bye!\n")
            break

        parts = line.split(None, 2)
        if len(parts) < 2:
            print("  Uzycie: METHOD /path [json_body]")
            continue

        method_name = parts[0].upper()
        path = parts[1]
        body = b""

        if len(parts) == 3:
            body = parts[2].encode("utf-8")

        client.send_request(method_name, path, body)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SoundHTTP Client — HTTP over Sound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przyklad:
  python client.py GET /ping
  python client.py POST /echo --body '{"key":"value"}'
  python client.py POST /msg --body '{"from":"Alice","text":"Czesc!"}'
  python client.py --interactive
  python client.py GET /ping --simulate
  python client.py --decode-response response.wav
  python client.py --list-devices
        """,
    )

    parser.add_argument("method", nargs="?", default=None,
                        help="Metoda HTTP: GET, POST")
    parser.add_argument("path", nargs="?", default=None,
                        help="Sciezka endpointu: /ping, /echo, ...")
    parser.add_argument("--body", default="",
                        help="Body zadania (JSON string)")
    parser.add_argument("--simulate", action="store_true",
                        help="Zapisz request jako WAV (bez audio)")
    parser.add_argument("--output-wav", metavar="PATH",
                        help="Sciezka WAV (simulate)")
    parser.add_argument("--decode-response", metavar="WAV",
                        help="Dekoduj odpowiedz z pliku WAV")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Tryb interaktywny (REPL)")
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

    client = SoundHTTPClient(
        speaker_index=args.speaker,
        mic_index=args.mic,
        sr=args.sr,
    )

    # Decode response mode
    if args.decode_response:
        client.decode_response_wav(args.decode_response)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(client)
        return

    # Single request
    if args.method is None or args.path is None:
        parser.print_help()
        return

    method_name = args.method.upper()
    path = args.path
    body = args.body.encode("utf-8") if args.body else b""

    if args.simulate:
        # Zapisz WAV bez odtwarzania
        wav_path = client.simulate_request(method_name, path, body, args.output_wav)
        print(f"  Zapisano request: {wav_path}")
        print(f"  Mozesz przetworzyc go serwerem:")
        print(f"    python server.py --simulate {wav_path}")
    else:
        # Live mode: odtwórz przez głośnik → nasłuchuj odpowiedzi
        print(f"\n  SoundHTTP Client  [live]")
        print(f"  {'=' * 50}")
        client.send_request(method_name, path, body)


if __name__ == "__main__":
    main()
