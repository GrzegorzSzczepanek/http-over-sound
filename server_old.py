#!/usr/bin/env python3
"""
SoundHTTP Server — serwer HTTP over Sound

Nasłuchuje na mikrofonie, dekoduje żądania FSK,
przetwarza je i odpowiada dźwiękiem FSK.

Wymaga: pyaudio (pip install pyaudio)

Użycie:
  python server.py [--port MIC_INDEX] [--speaker SPEAKER_INDEX]
  python server.py --simulate request.wav    # tryb symulacji bez mikrofonu
  python server.py --list-devices            # lista urządzeń audio
"""

import numpy as np
import struct
import sys
import os
import json
import time
import argparse
import threading
from datetime import datetime

# ── Import modułów modemu ──
from protocol import (
    FRAME_REQUEST, FRAME_RESPONSE,
    METHOD_GET, METHOD_POST, METHOD_PUT, METHOD_DELETE,
    METHOD_NAMES, METHOD_FROM_NAME,
    STATUS_MAP, http_to_compact, compact_to_http,
)

# ── Parametry FSK (identyczne z encoder/decoder) ──────────────────────
SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.08
SILENCE_GAP = 0.005
FREQ_BASE = 1000
FREQ_STEP = 200
FREQS = np.array([FREQ_BASE + i * FREQ_STEP for i in range(16)])
AMPLITUDE = 0.8
PREAMBLE = [0xA, 0xA, 0x5, 0x5, 0xA, 0xA, 0x5, 0x5]


def calc_steps(sr: int) -> tuple[int, int, int]:
    sym = int(sr * SYMBOL_DURATION)
    gap = int(sr * SILENCE_GAP)
    return sym, gap, sym + gap


SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
GAP_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP)
STEP_SAMPLES = SYMBOL_SAMPLES + GAP_SAMPLES


# ══════════════════════════════════════════════════════════════════════
#  FSK ENCODING (server → client response)
# ══════════════════════════════════════════════════════════════════════

def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def nibbles_from_bytes(data: bytes) -> list[int]:
    nibbles = []
    for b in data:
        nibbles.append((b >> 4) & 0xF)
        nibbles.append(b & 0xF)
    return nibbles


def generate_tone(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = AMPLITUDE * np.sin(2 * np.pi * freq * t)
    fade_samples = int(sr * 0.002)
    if fade_samples > 0 and len(tone) > 2 * fade_samples:
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return tone


def encode_response(status_compact: int, body: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zakoduj odpowiedź HTTP jako audio FSK."""
    # Ramka: [PREAMBLE] [TYPE=0x2] [STATUS:1byte] [BODY_LEN:2bytes] [BODY] [CRC16]
    frame = bytearray()
    frame.append((FRAME_RESPONSE << 4) | status_compact)  # type+status w 1 bajcie
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)

    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))

    nibbles = PREAMBLE + nibbles_from_bytes(bytes(frame))

    parts = [np.zeros(int(sr * 0.1))]  # 100ms ciszy
    for nib in nibbles:
        parts.append(generate_tone(FREQS[nib], SYMBOL_DURATION, sr))
        parts.append(np.zeros(int(sr * SILENCE_GAP)))
    parts.append(np.zeros(int(sr * 0.1)))

    return np.concatenate(parts)


def write_wav(path: str, data: np.ndarray, sr: int):
    audio_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    with open(path, "wb") as f:
        data_size = len(audio_int16) * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())


# ══════════════════════════════════════════════════════════════════════
#  FSK DECODING (client → server request)
# ══════════════════════════════════════════════════════════════════════

def detect_nibble(segment: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, float]:
    windowed = segment * np.hanning(len(segment))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)

    energies = np.zeros(16)
    bandwidth = FREQ_STEP * 0.4
    for i, target_freq in enumerate(FREQS):
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        if mask.any():
            energies[i] = np.max(fft[mask])

    best = np.argmax(energies)
    total = energies.sum()
    confidence = energies[best] / total if total > 0 else 0
    return int(best), confidence


def find_preamble(audio: np.ndarray, sr: int = SAMPLE_RATE) -> int | None:
    sym_samples, _, step_samples = calc_steps(sr)
    scan_step = step_samples // 4
    max_offset = len(audio) - step_samples * len(PREAMBLE)

    best_offset = None
    best_score = 0
    offset = 0

    while offset < max_offset:
        match_count = 0
        total_conf = 0
        for i, expected in enumerate(PREAMBLE):
            start = offset + i * step_samples
            end = start + sym_samples
            if end > len(audio):
                break
            nibble, conf = detect_nibble(audio[start:end], sr)
            if nibble == expected:
                match_count += 1
                total_conf += conf

        score = match_count + total_conf * 0.1
        if match_count == len(PREAMBLE) and score > best_score:
            best_score = score
            best_offset = offset
        offset += scan_step

    return best_offset


def decode_nibbles(audio: np.ndarray, start: int, count: int,
                   sr: int = SAMPLE_RATE) -> list[tuple[int, float]]:
    sym_samples, _, step_samples = calc_steps(sr)
    results = []
    for i in range(count):
        s = start + i * step_samples
        e = s + sym_samples
        if e > len(audio):
            results.append((0, 0.0))
            continue
        nib, conf = detect_nibble(audio[s:e], sr)
        results.append((nib, conf))
    return results


def nibbles_to_bytes(nibbles: list[int]) -> bytes:
    if len(nibbles) % 2 != 0:
        nibbles.append(0)
    result = bytearray()
    for i in range(0, len(nibbles), 2):
        result.append((nibbles[i] << 4) | nibbles[i + 1])
    return bytes(result)


def decode_request(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict | None:
    """
    Dekoduj żądanie HTTP z audio FSK.

    Ramka request:
      [PREAMBLE 8nib] [TYPE_METHOD 1byte/2nib] [PATH_LEN 1byte/2nib]
      [PATH... 2*path_len nib] [BODY_LEN 2bytes/4nib] [BODY... 2*body_len nib] [CRC16 2bytes/4nib]
    """
    _, _, step_samples = calc_steps(sr)

    preamble_offset = find_preamble(audio, sr)
    if preamble_offset is None:
        return None

    pos = preamble_offset + len(PREAMBLE) * step_samples

    # 1. TYPE + METHOD (1 bajt = 2 nibble'e)
    header_nibs = decode_nibbles(audio, pos, 2, sr)
    type_method_byte = (header_nibs[0][0] << 4) | header_nibs[1][0]
    frame_type = (type_method_byte >> 4) & 0xF
    method = type_method_byte & 0xF
    pos += 2 * step_samples

    if frame_type != FRAME_REQUEST:
        return None

    # 2. PATH_LEN (1 bajt = 2 nibble'e)
    plen_nibs = decode_nibbles(audio, pos, 2, sr)
    path_len = (plen_nibs[0][0] << 4) | plen_nibs[1][0]
    pos += 2 * step_samples

    # 3. PATH (path_len bajtów)
    path_nib_count = path_len * 2
    path_nibs = decode_nibbles(audio, pos, path_nib_count, sr)
    path_bytes = nibbles_to_bytes([n[0] for n in path_nibs])
    path = path_bytes.decode("utf-8", errors="replace")
    pos += path_nib_count * step_samples

    # 4. BODY_LEN (2 bajty = 4 nibble'e)
    blen_nibs = decode_nibbles(audio, pos, 4, sr)
    body_len_bytes = nibbles_to_bytes([n[0] for n in blen_nibs])
    body_len = struct.unpack(">H", body_len_bytes)[0]
    pos += 4 * step_samples

    # 5. BODY
    body = b""
    if body_len > 0:
        body_nib_count = body_len * 2
        body_nibs = decode_nibbles(audio, pos, body_nib_count, sr)
        body = nibbles_to_bytes([n[0] for n in body_nibs])
        pos += body_nib_count * step_samples

    # 6. CRC16 (2 bajty = 4 nibble'e)
    crc_nibs = decode_nibbles(audio, pos, 4, sr)
    received_crc = struct.unpack(">H", nibbles_to_bytes([n[0] for n in crc_nibs]))[0]

    # Odtwórz ramkę do weryfikacji CRC
    frame = bytearray()
    frame.append(type_method_byte)
    frame.append(path_len)
    frame.extend(path_bytes[:path_len])
    frame.extend(struct.pack(">H", body_len))
    frame.extend(body[:body_len])
    computed_crc = crc16(bytes(frame))

    return {
        "method": method,
        "method_name": METHOD_NAMES.get(method, "UNKNOWN"),
        "path": path,
        "body": body[:body_len],
        "crc_ok": received_crc == computed_crc,
        "crc_received": received_crc,
        "crc_computed": computed_crc,
    }


def encode_request(method: int, path: str, body: bytes = b"",
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zakoduj żądanie HTTP jako audio FSK (dla klienta)."""
    path_bytes = path.encode("utf-8")

    frame = bytearray()
    frame.append((FRAME_REQUEST << 4) | (method & 0xF))
    frame.append(len(path_bytes))
    frame.extend(path_bytes)
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)

    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))

    nibbles = PREAMBLE + nibbles_from_bytes(bytes(frame))

    parts = [np.zeros(int(sr * 0.15))]
    for nib in nibbles:
        parts.append(generate_tone(FREQS[nib], SYMBOL_DURATION, sr))
        parts.append(np.zeros(int(sr * SILENCE_GAP)))
    parts.append(np.zeros(int(sr * 0.15)))

    return np.concatenate(parts)


# ══════════════════════════════════════════════════════════════════════
#  ROUTER — definicja endpointów
# ══════════════════════════════════════════════════════════════════════

class SoundRouter:
    """Prosty router HTTP — rejestruj handlery jak w Flask."""

    def __init__(self):
        self.routes: dict[tuple[int, str], callable] = {}
        self.storage: dict[str, any] = {}  # prosty key-value store

    def route(self, path: str, methods: list[str] = None):
        """Dekorator do rejestracji endpointów."""
        if methods is None:
            methods = ["GET"]

        def decorator(func):
            for m in methods:
                method_code = METHOD_FROM_NAME.get(m.upper())
                if method_code is not None:
                    self.routes[(method_code, path)] = func
            return func
        return decorator

    def handle(self, method: int, path: str, body: bytes) -> tuple[int, bytes]:
        """
        Obsłuż żądanie. Zwraca (status_compact, response_body).
        """
        handler = self.routes.get((method, path))

        # Próbuj pattern matching dla parametrycznych ścieżek
        if handler is None:
            for (m, p), h in self.routes.items():
                if m == method and self._match_path(p, path):
                    handler = h
                    break

        if handler is None:
            # Sprawdź czy ścieżka istnieje z inną metodą
            any_method = any(p == path for (_, p) in self.routes)
            if any_method:
                return http_to_compact(405), json.dumps(
                    {"error": "Method Not Allowed"}
                ).encode()
            return http_to_compact(404), json.dumps(
                {"error": "Not Found", "path": path}
            ).encode()

        try:
            status, response = handler(path=path, body=body, storage=self.storage)
            if isinstance(response, dict):
                response = json.dumps(response, ensure_ascii=False).encode()
            elif isinstance(response, str):
                response = response.encode()
            return http_to_compact(status), response
        except Exception as e:
            return http_to_compact(500), json.dumps(
                {"error": str(e)}
            ).encode()

    @staticmethod
    def _match_path(pattern: str, actual: str) -> bool:
        """Proste porównanie ścieżek (na razie exact match)."""
        return pattern == actual


# ══════════════════════════════════════════════════════════════════════
#  DOMYŚLNE ENDPOINTY
# ══════════════════════════════════════════════════════════════════════

def setup_default_routes(router: SoundRouter):
    """Zarejestruj domyślne endpointy."""

    @router.route("/", methods=["GET"])
    def index(path, body, storage):
        return 200, {
            "server": "SoundHTTP/1.0",
            "message": "Welcome to HTTP over Sound!",
            "endpoints": ["/", "/ping", "/time", "/echo", "/store"],
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
            "unix": int(now.timestamp()),
        }

    @router.route("/echo", methods=["POST"])
    def echo(path, body, storage):
        try:
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {"raw": body.hex()}
        return 200, {"echo": data}

    @router.route("/store", methods=["GET"])
    def store_list(path, body, storage):
        return 200, {"keys": list(storage.keys()), "count": len(storage)}

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
#  WAV I/O (bez pyaudio — do trybu simulate)
# ══════════════════════════════════════════════════════════════════════

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        f.read(4); f.read(4); f.read(4)
        fmt_found = False; data_bytes = None
        sr = 44100; num_channels = 1; sample_width = 2
        while True:
            cid = f.read(4)
            if len(cid) < 4: break
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                fmt = f.read(csz)
                num_channels = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                sample_width = struct.unpack("<H", fmt[14:16])[0] // 8
                fmt_found = True
            elif cid == b"data":
                data_bytes = f.read(csz)
            else:
                f.read(csz)
        if sample_width == 2:
            samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32767.0
        else:
            samples = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float64) - 128) / 128.0
        if num_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples, sr


# ══════════════════════════════════════════════════════════════════════
#  SERWER
# ══════════════════════════════════════════════════════════════════════

class SoundHTTPServer:
    """Główna klasa serwera."""

    def __init__(self, router: SoundRouter = None):
        self.router = router or SoundRouter()
        self.running = False
        self.request_count = 0

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts}] {msg}")

    def handle_audio(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Przetwórz audio z żądaniem, zwróć audio z odpowiedzią."""
        self.log("📡 Dekodowanie żądania...")

        request = decode_request(audio, sr)
        if request is None:
            self.log("❌ Nie udało się zdekodować żądania")
            return None

        self.request_count += 1
        method_name = request["method_name"]
        path = request["path"]
        crc_status = "✅" if request["crc_ok"] else "⚠️ CRC FAIL"

        body_preview = ""
        if request["body"]:
            try:
                body_preview = f' body="{request["body"].decode("utf-8")[:80]}"'
            except UnicodeDecodeError:
                body_preview = f" body=<{len(request['body'])}B binary>"

        self.log(f"📨 #{self.request_count} {method_name} {path}{body_preview} [{crc_status}]")

        # Przetwarzaj nawet z CRC fail (best effort)
        status_compact, response_body = self.router.handle(
            request["method"], request["path"], request["body"]
        )

        http_code, status_text = compact_to_http(status_compact)
        self.log(f"📤 → {http_code} {status_text} ({len(response_body)}B)")

        try:
            resp_preview = response_body.decode("utf-8")[:100]
            self.log(f"   {resp_preview}")
        except UnicodeDecodeError:
            pass

        # Enkoduj odpowiedź
        response_audio = encode_response(status_compact, response_body, sr)
        return response_audio

    def serve_simulate(self, wav_path: str, response_wav: str = None):
        """Tryb symulacji — wczytaj WAV, przetwórz, zapisz odpowiedź."""
        print(f"\n🔊 SoundHTTP Server (simulate mode)")
        print(f"{'─' * 50}")

        audio, sr = read_wav(wav_path)
        self.log(f"Wczytano: {wav_path} ({len(audio)/sr:.2f}s, SR={sr})")

        response_audio = self.handle_audio(audio, sr)

        if response_audio is not None:
            out_path = response_wav or wav_path.replace(".wav", "_response.wav")
            write_wav(out_path, response_audio, sr)
            self.log(f"💾 Odpowiedź zapisana: {out_path} ({len(response_audio)/sr:.2f}s)")
            return out_path
        return None

    def serve_live(self, mic_index: int = None, speaker_index: int = None,
                   listen_sr: int = 44100):
        """
        Tryb live — nasłuchuj na mikrofonie, odpowiadaj przez głośnik.
        Wymaga pyaudio.
        """
        try:
            import pyaudio
        except ImportError:
            print("❌ pyaudio nie jest zainstalowany!")
            print("   pip install pyaudio")
            sys.exit(1)

        pa = pyaudio.PyAudio()
        self.running = True

        print(f"\n🔊 SoundHTTP Server (live mode)")
        print(f"{'─' * 50}")
        print(f"  Mikrofon SR: {listen_sr}")
        print(f"  Nasłuchuję... (Ctrl+C aby zatrzymać)\n")

        CHUNK = int(listen_sr * 0.05)  # 50ms chunks
        BUFFER_SECONDS = 30  # max bufor
        buffer_size = int(listen_sr * BUFFER_SECONDS)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=listen_sr,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=CHUNK,
        )

        audio_buffer = np.zeros(buffer_size, dtype=np.float64)
        write_pos = 0
        last_preamble_time = 0
        COOLDOWN = 2.0  # sekundy między żądaniami

        try:
            while self.running:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0

                # Dopisz do bufora kołowego
                end = write_pos + len(chunk)
                if end <= buffer_size:
                    audio_buffer[write_pos:end] = chunk
                else:
                    first = buffer_size - write_pos
                    audio_buffer[write_pos:] = chunk[:first]
                    audio_buffer[:len(chunk) - first] = chunk[first:]
                write_pos = end % buffer_size

                # Sprawdź RMS — czy jest sygnał
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.005:
                    continue

                now = time.time()
                if now - last_preamble_time < COOLDOWN:
                    continue

                # Szukaj preambuły w ostatnich N sekundach
                lookback = int(listen_sr * 5)  # 5s
                if write_pos >= lookback:
                    segment = audio_buffer[write_pos - lookback:write_pos]
                else:
                    segment = np.concatenate([
                        audio_buffer[buffer_size - (lookback - write_pos):],
                        audio_buffer[:write_pos]
                    ])

                preamble_pos = find_preamble(segment, listen_sr)
                if preamble_pos is not None:
                    last_preamble_time = now
                    self.log("🎯 Preambuła wykryta! Czekam na pełne żądanie...")

                    # Poczekaj chwilę na resztę danych
                    time.sleep(1.5)

                    # Zbierz świeże dane
                    extra_raw = stream.read(
                        int(listen_sr * 0.5),
                        exception_on_overflow=False
                    )
                    extra = np.frombuffer(extra_raw, dtype=np.int16).astype(np.float64) / 32767.0
                    full_audio = np.concatenate([segment, extra])

                    response_audio = self.handle_audio(full_audio, listen_sr)

                    if response_audio is not None:
                        self.log("🔊 Odtwarzam odpowiedź...")
                        out_stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=listen_sr,
                            output=True,
                            output_device_index=speaker_index,
                        )
                        out_data = np.clip(
                            response_audio * 32767, -32768, 32767
                        ).astype(np.int16)
                        out_stream.write(out_data.tobytes())
                        out_stream.close()
                        self.log("✅ Odpowiedź wysłana")

        except KeyboardInterrupt:
            print("\n\n👋 Server zatrzymany")
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
Tryby:
  --simulate request.wav    Przetwórz zapisany WAV z żądaniem
  --live                    Nasłuchuj na mikrofonie (wymaga pyaudio)
  --list-devices            Lista urządzeń audio

Przykłady:
  python server.py --simulate get_root.wav
  python server.py --simulate post_echo.wav --response-wav resp.wav
  python server.py --live
  python server.py --live --mic 1 --speaker 0
        """
    )

    parser.add_argument("--simulate", metavar="WAV",
                        help="Tryb symulacji — przetwórz plik WAV")
    parser.add_argument("--response-wav", metavar="PATH",
                        help="Ścieżka do zapisu odpowiedzi WAV")
    parser.add_argument("--live", action="store_true",
                        help="Tryb live — nasłuchuj na mikrofonie")
    parser.add_argument("--mic", type=int, default=None,
                        help="Indeks mikrofonu")
    parser.add_argument("--speaker", type=int, default=None,
                        help="Indeks głośnika")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate mikrofonu (domyślnie: 44100)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Lista urządzeń audio")

    args = parser.parse_args()

    # Setup router
    router = SoundRouter()
    setup_default_routes(router)
    server = SoundHTTPServer(router)

    if args.list_devices:
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            print("Urządzenia audio:")
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                direction = []
                if info["maxInputChannels"] > 0:
                    direction.append("IN")
                if info["maxOutputChannels"] > 0:
                    direction.append("OUT")
                print(f"  [{i}] {info['name']} ({'/'.join(direction)}) "
                      f"SR={int(info['defaultSampleRate'])}")
            pa.terminate()
        except ImportError:
            print("❌ pyaudio nie zainstalowany")
        return

    if args.simulate:
        server.serve_simulate(args.simulate, args.response_wav)
    elif args.live:
        server.serve_live(args.mic, args.speaker, args.sr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()