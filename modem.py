#!/usr/bin/env python3
"""
modem.py — Współdzielony moduł FSK modemu

Cały kod przetwarzania dźwięku FSK w jednym miejscu:
  - Enkodowanie / dekodowanie ramek (request & response)
  - Generowanie / analiza tonów FSK
  - CRC16, nibble conversion
  - WAV I/O
  - Odtwarzanie / nagrywanie przez PyAudio

Używany przez server.py i client.py.
"""

import numpy as np
import struct
import json
import time

from protocol import (
    FRAME_REQUEST, FRAME_RESPONSE,
    METHOD_GET, METHOD_POST, METHOD_PUT, METHOD_DELETE,
    METHOD_NAMES, METHOD_FROM_NAME,
    STATUS_MAP, STATUS_FROM_HTTP,
    http_to_compact, compact_to_http,
)

# ══════════════════════════════════════════════════════════════════════
#  STAŁE FSK
# ══════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.08       # 80ms na symbol
SILENCE_GAP = 0.005          # 5ms cisza między symbolami
FREQ_BASE = 1000
FREQ_STEP = 200
FREQS = np.array([FREQ_BASE + i * FREQ_STEP for i in range(16)])
AMPLITUDE = 0.8
PREAMBLE = [0xA, 0xA, 0x5, 0x5, 0xA, 0xA, 0x5, 0x5]

SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
GAP_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP)
STEP_SAMPLES = SYMBOL_SAMPLES + GAP_SAMPLES

NIBBLES_PER_SEC = 1.0 / (SYMBOL_DURATION + SILENCE_GAP)
BYTES_PER_SEC = NIBBLES_PER_SEC / 2


def calc_steps(sr: int) -> tuple[int, int, int]:
    """Oblicz (symbol_samples, gap_samples, step_samples) dla danego SR."""
    sym = int(sr * SYMBOL_DURATION)
    gap = int(sr * SILENCE_GAP)
    return sym, gap, sym + gap


# ══════════════════════════════════════════════════════════════════════
#  PRYMITYWY: CRC, NIBBLE, TONE
# ══════════════════════════════════════════════════════════════════════

def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE."""
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
    """Bajty → lista nibble'ów (po 2 na bajt)."""
    out = []
    for b in data:
        out.append((b >> 4) & 0xF)
        out.append(b & 0xF)
    return out


def nibbles_to_bytes(nibbles: list[int]) -> bytes:
    """Lista nibble'ów → bajty."""
    if len(nibbles) % 2 != 0:
        nibbles = list(nibbles) + [0]
    result = bytearray()
    for i in range(0, len(nibbles), 2):
        result.append((nibbles[i] << 4) | nibbles[i + 1])
    return bytes(result)


def generate_tone(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generuj sinusoidę z fade-in/out."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = AMPLITUDE * np.sin(2 * np.pi * freq * t)
    fade = int(sr * 0.002)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


# ══════════════════════════════════════════════════════════════════════
#  WAV I/O
# ══════════════════════════════════════════════════════════════════════

def write_wav(path: str, data: np.ndarray, sr: int = SAMPLE_RATE):
    """Zapisz audio jako WAV 16-bit mono."""
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


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Wczytaj WAV → (samples float64, sample_rate)."""
    with open(path, "rb") as f:
        f.read(12)  # RIFF header
        sr = 44100
        channels = 1
        sample_width = 2
        data_bytes = b""
        while True:
            cid = f.read(4)
            if len(cid) < 4:
                break
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                fmt = f.read(csz)
                channels = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                sample_width = struct.unpack("<H", fmt[14:16])[0] // 8
            elif cid == b"data":
                data_bytes = f.read(csz)
            else:
                f.read(csz)
        if sample_width == 2:
            samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32767.0
        else:
            samples = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float64) - 128) / 128.0
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples, sr


# ══════════════════════════════════════════════════════════════════════
#  NIBBLE → AUDIO (encode)
# ══════════════════════════════════════════════════════════════════════

def _nibbles_to_audio(nibbles: list[int], sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zamień sekwencję nibble'ów na audio FSK (z preambuła + ciszą)."""
    full = PREAMBLE + nibbles
    parts = [np.zeros(int(sr * 0.15))]  # 150ms ciszy przed
    for nib in full:
        parts.append(generate_tone(FREQS[nib], SYMBOL_DURATION, sr))
        parts.append(np.zeros(int(sr * SILENCE_GAP)))
    parts.append(np.zeros(int(sr * 0.15)))  # 150ms ciszy po
    return np.concatenate(parts)


# ══════════════════════════════════════════════════════════════════════
#  AUDIO → NIBBLE (decode)
# ══════════════════════════════════════════════════════════════════════

def detect_nibble(segment: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, float]:
    """Wykryj nibble z segmentu audio (zero-padded FFT, energy-based)."""
    windowed = segment * np.hanning(len(segment))
    nfft = max(len(windowed), 8192)  # zero-pad dla lepszej rozdzielczości
    fft_mag = np.abs(np.fft.rfft(windowed, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    energies = np.zeros(16)
    bw = FREQ_STEP * 0.35
    for i, tf in enumerate(FREQS):
        mask = (freqs >= tf - bw) & (freqs <= tf + bw)
        if mask.any():
            energies[i] = np.sum(fft_mag[mask] ** 2)  # energia zamiast max
    best = np.argmax(energies)
    total = energies.sum()
    conf = energies[best] / total if total > 0 else 0
    return int(best), conf


def find_preamble(audio: np.ndarray, sr: int = SAMPLE_RATE) -> int | None:
    """Znajdź offset preambuły w audio. Zwróci None jeśli nie znaleziono."""
    sym, _, step = calc_steps(sr)
    scan = step // 4
    max_off = len(audio) - step * len(PREAMBLE)
    if max_off <= 0:
        return None

    best_off = None
    best_score = 0.0
    off = 0

    while off < max_off:
        match = 0
        conf_sum = 0.0
        for i, expected in enumerate(PREAMBLE):
            s = off + i * step
            e = s + sym
            if e > len(audio):
                break
            nib, c = detect_nibble(audio[s:e], sr)
            if nib == expected:
                match += 1
                conf_sum += c

        score = match + conf_sum * 0.1
        if match >= len(PREAMBLE) - 1 and score > best_score:
            best_score = score
            best_off = off
        off += scan

    return best_off


def decode_nibbles(audio: np.ndarray, start: int, count: int,
                   sr: int = SAMPLE_RATE) -> list[tuple[int, float]]:
    """Dekoduj N nibble'ów zaczynając od offsetu `start`."""
    sym, _, step = calc_steps(sr)
    results = []
    for i in range(count):
        s = start + i * step
        e = s + sym
        if e > len(audio):
            results.append((0, 0.0))
            continue
        nib, conf = detect_nibble(audio[s:e], sr)
        results.append((nib, conf))
    return results


# ══════════════════════════════════════════════════════════════════════
#  ENCODE / DECODE REQUEST
# ══════════════════════════════════════════════════════════════════════

def encode_request(method: int, path: str, body: bytes = b"",
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Zakoduj żądanie HTTP → audio FSK.

    Ramka: [TYPE+METHOD:1B] [PATH_LEN:1B] [PATH...] [BODY_LEN:2B] [BODY...] [CRC16:2B]
    """
    path_bytes = path.encode("utf-8")

    frame = bytearray()
    frame.append((FRAME_REQUEST << 4) | (method & 0xF))
    frame.append(len(path_bytes))
    frame.extend(path_bytes)
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)

    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))

    nibbles = nibbles_from_bytes(bytes(frame))

    total_nib = len(PREAMBLE) + len(nibbles)
    total_dur = total_nib * (SYMBOL_DURATION + SILENCE_GAP)
    print(f"  [encode] {METHOD_NAMES.get(method, '?')} {path}  "
          f"{len(frame)}B → {total_nib} symbols → {total_dur:.1f}s  "
          f"CRC=0x{checksum:04X}")

    return _nibbles_to_audio(nibbles, sr)


def decode_request(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   strict: bool = True) -> dict | None:
    """
    Dekoduj żądanie HTTP z audio FSK.

    Zwraca dict z: method, method_name, path, body, crc_ok, ...
    strict=False: best-effort — nie odrzuca przy złym frame_type,
                  zwraca częściowe dane nawet przy błędach.
    """
    _, _, step = calc_steps(sr)

    preamble_off = find_preamble(audio, sr)
    if preamble_off is None:
        return None

    pos = preamble_off + len(PREAMBLE) * step

    # TYPE + METHOD (1 bajt = 2 nibble)
    hdr = decode_nibbles(audio, pos, 2, sr)
    tm_byte = (hdr[0][0] << 4) | hdr[1][0]
    frame_type = (tm_byte >> 4) & 0xF
    method = tm_byte & 0xF
    pos += 2 * step

    if frame_type != FRAME_REQUEST:
        if strict:
            return None
        print(f"  [lenient] frame_type=0x{frame_type:X} (oczekiwano REQUEST=0x{FRAME_REQUEST:X}), kontynuuje...")

    # PATH_LEN (1 bajt)
    pl = decode_nibbles(audio, pos, 2, sr)
    path_len = (pl[0][0] << 4) | pl[1][0]
    pos += 2 * step

    # PATH
    pn = path_len * 2
    pnibs = decode_nibbles(audio, pos, pn, sr)
    path_bytes = nibbles_to_bytes([n[0] for n in pnibs])
    path = path_bytes.decode("utf-8", errors="replace")
    pos += pn * step

    # BODY_LEN (2 bajty)
    bl = decode_nibbles(audio, pos, 4, sr)
    body_len = struct.unpack(">H", nibbles_to_bytes([n[0] for n in bl]))[0]
    pos += 4 * step

    # BODY
    body = b""
    if body_len > 0:
        bn = body_len * 2
        bnibs = decode_nibbles(audio, pos, bn, sr)
        body = nibbles_to_bytes([n[0] for n in bnibs])
        pos += bn * step

    # CRC16 (2 bajty)
    cn = decode_nibbles(audio, pos, 4, sr)
    received_crc = struct.unpack(">H", nibbles_to_bytes([n[0] for n in cn]))[0]

    # Weryfikacja
    frame = bytearray()
    frame.append(tm_byte)
    frame.append(path_len)
    frame.extend(path_bytes[:path_len])
    frame.extend(struct.pack(">H", body_len))
    frame.extend(body[:body_len])
    computed_crc = crc16(bytes(frame))

    crc_ok = received_crc == computed_crc
    if not crc_ok and not strict:
        print(f"  [lenient] CRC FAIL request (recv=0x{received_crc:04X} "
              f"calc=0x{computed_crc:04X}), zwracam dane best-effort")

    return {
        "method": method,
        "method_name": METHOD_NAMES.get(method, "UNKNOWN"),
        "path": path,
        "body": body[:body_len],
        "crc_ok": crc_ok,
        "crc_received": received_crc,
        "crc_computed": computed_crc,
    }


# ══════════════════════════════════════════════════════════════════════
#  ENCODE / DECODE RESPONSE
# ══════════════════════════════════════════════════════════════════════

def encode_response(status_compact: int, body: bytes,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Zakoduj odpowiedź HTTP → audio FSK.

    Ramka: [TYPE+STATUS:1B] [BODY_LEN:2B] [BODY...] [CRC16:2B]
    """
    frame = bytearray()
    frame.append((FRAME_RESPONSE << 4) | (status_compact & 0xF))
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)

    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))

    nibbles = nibbles_from_bytes(bytes(frame))

    http_code, status_text = compact_to_http(status_compact)
    total_nib = len(PREAMBLE) + len(nibbles)
    total_dur = total_nib * (SYMBOL_DURATION + SILENCE_GAP)
    print(f"  [encode] {http_code} {status_text}  "
          f"{len(frame)}B → {total_nib} symbols → {total_dur:.1f}s  "
          f"CRC=0x{checksum:04X}")

    return _nibbles_to_audio(nibbles, sr)


def decode_response(audio: np.ndarray, sr: int = SAMPLE_RATE,
                    strict: bool = True) -> dict | None:
    """
    Dekoduj odpowiedź HTTP z audio FSK.

    Zwraca dict z: status_compact, http_code, status_text, body, crc_ok, ...
    strict=False: best-effort — nie odrzuca przy złym frame_type.
    """
    _, _, step = calc_steps(sr)

    preamble_off = find_preamble(audio, sr)
    if preamble_off is None:
        return None

    pos = preamble_off + len(PREAMBLE) * step

    # TYPE + STATUS (1 bajt = 2 nibble)
    hdr = decode_nibbles(audio, pos, 2, sr)
    ts_byte = (hdr[0][0] << 4) | hdr[1][0]
    frame_type = (ts_byte >> 4) & 0xF
    status_compact = ts_byte & 0xF
    pos += 2 * step

    if frame_type != FRAME_RESPONSE:
        if strict:
            return None
        print(f"  [lenient] frame_type=0x{frame_type:X} (oczekiwano RESPONSE=0x{FRAME_RESPONSE:X}), kontynuuje...")

    # BODY_LEN (2 bajty = 4 nibble)
    bl = decode_nibbles(audio, pos, 4, sr)
    body_len = struct.unpack(">H", nibbles_to_bytes([n[0] for n in bl]))[0]
    pos += 4 * step

    # BODY
    body = b""
    if body_len > 0:
        bn = body_len * 2
        bnibs = decode_nibbles(audio, pos, bn, sr)
        body = nibbles_to_bytes([n[0] for n in bnibs])
        pos += bn * step

    # CRC16 (2 bajty = 4 nibble)
    cn = decode_nibbles(audio, pos, 4, sr)
    received_crc = struct.unpack(">H", nibbles_to_bytes([n[0] for n in cn]))[0]

    # Weryfikacja
    frame = bytearray()
    frame.append(ts_byte)
    frame.extend(struct.pack(">H", body_len))
    frame.extend(body[:body_len])
    computed_crc = crc16(bytes(frame))

    http_code, status_text = compact_to_http(status_compact)

    crc_ok = received_crc == computed_crc
    if not crc_ok and not strict:
        print(f"  [lenient] CRC FAIL response (recv=0x{received_crc:04X} "
              f"calc=0x{computed_crc:04X}), zwracam dane best-effort")

    return {
        "status_compact": status_compact,
        "http_code": http_code,
        "status_text": status_text,
        "body": body[:body_len],
        "crc_ok": crc_ok,
        "crc_received": received_crc,
        "crc_computed": computed_crc,
    }


# ══════════════════════════════════════════════════════════════════════
#  BUDOWANIE / PARSOWANIE RAMEK (bajty, bez audio)
# ══════════════════════════════════════════════════════════════════════

def build_request_frame(method: int, path: str, body: bytes = b"") -> bytes:
    """Zbuduj ramkę HTTP request jako bajty (bez kodowania FSK)."""
    path_bytes = path.encode("utf-8")
    frame = bytearray()
    frame.append((FRAME_REQUEST << 4) | (method & 0xF))
    frame.append(len(path_bytes))
    frame.extend(path_bytes)
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)
    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))
    return bytes(frame)


def build_response_frame(status_compact: int, body: bytes) -> bytes:
    """Zbuduj ramkę HTTP response jako bajty (bez kodowania FSK)."""
    frame = bytearray()
    frame.append((FRAME_RESPONSE << 4) | (status_compact & 0xF))
    frame.extend(struct.pack(">H", len(body)))
    frame.extend(body)
    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))
    return bytes(frame)


def parse_request_frame(data: bytes, strict: bool = True) -> dict | None:
    """Parsuj ramkę HTTP request z bajtów.
    strict=False: best-effort — próbuje wyciągnać co się da."""
    if len(data) < 6:  # min: TYPE+METHOD(1) + PATH_LEN(1) + BODY_LEN(2) + CRC(2)
        if strict:
            return None
        # w lenient spróbuj gołą dekodację
        if len(data) < 2:
            return None
    try:
        pos = 0
        tm_byte = data[pos]; pos += 1
        frame_type = (tm_byte >> 4) & 0xF
        method = tm_byte & 0xF
        if frame_type != FRAME_REQUEST:
            if strict:
                return None
            print(f"  [lenient] parse_request: frame_type=0x{frame_type:X}, kontynuuje...")

        path_len = data[pos]; pos += 1
        if pos + path_len + 4 > len(data):
            if strict:
                return None
            # lenient: obetnij path do dostępnych danych
            available = len(data) - pos
            path_len = min(path_len, max(0, available - 4))
            print(f"  [lenient] obcięto path_len do {path_len}")

        path_bytes = data[pos:pos + path_len]; pos += path_len
        path = path_bytes.decode("utf-8", errors="replace")

        # BODY_LEN
        if pos + 2 > len(data):
            if strict:
                return None
            return {
                "method": method,
                "method_name": METHOD_NAMES.get(method, "UNKNOWN"),
                "path": path,
                "body": b"",
                "crc_ok": False,
                "crc_received": 0, "crc_computed": 0,
            }

        body_len = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
        body = data[pos:pos + body_len]; pos += body_len

        # CRC
        if pos + 2 > len(data):
            if strict:
                return None
            print(f"  [lenient] brak CRC, zwracam best-effort")
            return {
                "method": method,
                "method_name": METHOD_NAMES.get(method, "UNKNOWN"),
                "path": path,
                "body": body[:body_len],
                "crc_ok": False,
                "crc_received": 0, "crc_computed": 0,
            }

        received_crc = struct.unpack(">H", data[pos:pos + 2])[0]

        frame_for_crc = bytearray()
        frame_for_crc.append(tm_byte)
        frame_for_crc.append(path_len)
        frame_for_crc.extend(path_bytes[:path_len])
        frame_for_crc.extend(struct.pack(">H", body_len))
        frame_for_crc.extend(body[:body_len])
        computed_crc = crc16(bytes(frame_for_crc))

        return {
            "method": method,
            "method_name": METHOD_NAMES.get(method, "UNKNOWN"),
            "path": path,
            "body": body[:body_len],
            "crc_ok": received_crc == computed_crc,
            "crc_received": received_crc,
            "crc_computed": computed_crc,
        }
    except Exception as e:
        if not strict:
            print(f"  [lenient] parse_request exception: {e}")
        return None


def parse_response_frame(data: bytes, strict: bool = True) -> dict | None:
    """Parsuj ramkę HTTP response z bajtów.
    strict=False: best-effort — próbuje wyciągnać co się da."""
    if len(data) < 5:  # min: TYPE+STATUS(1) + BODY_LEN(2) + CRC(2)
        if strict:
            return None
        if len(data) < 1:
            return None
    try:
        pos = 0
        ts_byte = data[pos]; pos += 1
        frame_type = (ts_byte >> 4) & 0xF
        status_compact = ts_byte & 0xF
        if frame_type != FRAME_RESPONSE:
            if strict:
                return None
            print(f"  [lenient] parse_response: frame_type=0x{frame_type:X}, kontynuuje...")

        # BODY_LEN
        if pos + 2 > len(data):
            if strict:
                return None
            http_code, status_text = compact_to_http(status_compact)
            return {
                "status_compact": status_compact,
                "http_code": http_code, "status_text": status_text,
                "body": b"", "crc_ok": False,
                "crc_received": 0, "crc_computed": 0,
            }

        body_len = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
        body = data[pos:pos + body_len]; pos += body_len

        # CRC
        if pos + 2 > len(data):
            if strict:
                return None
            http_code, status_text = compact_to_http(status_compact)
            print(f"  [lenient] brak CRC, zwracam best-effort")
            return {
                "status_compact": status_compact,
                "http_code": http_code, "status_text": status_text,
                "body": body[:body_len], "crc_ok": False,
                "crc_received": 0, "crc_computed": 0,
            }

        received_crc = struct.unpack(">H", data[pos:pos + 2])[0]

        frame_for_crc = bytearray()
        frame_for_crc.append(ts_byte)
        frame_for_crc.extend(struct.pack(">H", body_len))
        frame_for_crc.extend(body[:body_len])
        computed_crc = crc16(bytes(frame_for_crc))

        http_code, status_text = compact_to_http(status_compact)

        return {
            "status_compact": status_compact,
            "http_code": http_code,
            "status_text": status_text,
            "body": body[:body_len],
            "crc_ok": received_crc == computed_crc,
            "crc_received": received_crc,
            "crc_computed": computed_crc,
        }
    except Exception as e:
        if not strict:
            print(f"  [lenient] parse_response exception: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  PYAUDIO HELPERS — odtwarzanie i nagrywanie
# ══════════════════════════════════════════════════════════════════════

def play_audio(audio: np.ndarray, sr: int = SAMPLE_RATE,
               device_index: int | None = None):
    """Odtwórz audio przez głośnik (blokujące)."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            output=True,
            output_device_index=device_index,
        )
        out = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        stream.write(out.tobytes())
        stream.close()
    finally:
        pa.terminate()


def record_audio(duration: float, sr: int = SAMPLE_RATE,
                 device_index: int | None = None) -> np.ndarray:
    """Nagraj audio z mikrofonu (blokujące)."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        chunk = 1024
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk,
        )
        frames = []
        total = int(sr * duration / chunk)
        for _ in range(total):
            raw = stream.read(chunk, exception_on_overflow=False)
            frames.append(raw)
        stream.close()
        raw_all = b"".join(frames)
        return np.frombuffer(raw_all, dtype=np.int16).astype(np.float64) / 32767.0
    finally:
        pa.terminate()


def list_audio_devices():
    """Wyświetl dostępne urządzenia audio."""
    import pyaudio
    pa = pyaudio.PyAudio()
    print("\n  Urządzenia audio:")
    print(f"  {'─' * 60}")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        dirs = []
        if info["maxInputChannels"] > 0:
            dirs.append(f"IN:{info['maxInputChannels']}ch")
        if info["maxOutputChannels"] > 0:
            dirs.append(f"OUT:{info['maxOutputChannels']}ch")
        sr = int(info["defaultSampleRate"])
        print(f"  [{i:2d}] {info['name']:<40} {'/'.join(dirs)}  SR={sr}")
    print()
    pa.terminate()


# ══════════════════════════════════════════════════════════════════════
#  ESTYMACJA CZASU TRANSMISJI
# ══════════════════════════════════════════════════════════════════════

def estimate_duration(frame_bytes: int) -> float:
    """Oszacuj czas transmisji ramki w sekundach."""
    nibbles = len(PREAMBLE) + frame_bytes * 2
    return nibbles * (SYMBOL_DURATION + SILENCE_GAP) + 0.3  # + cisza


def estimate_request_duration(path: str, body: bytes = b"") -> float:
    """Oszacuj czas transmisji żądania."""
    # TYPE+METHOD(1) + PATH_LEN(1) + PATH + BODY_LEN(2) + BODY + CRC(2)
    frame_len = 1 + 1 + len(path.encode()) + 2 + len(body) + 2
    return estimate_duration(frame_len)


def estimate_response_duration(body: bytes = b"") -> float:
    """Oszacuj czas transmisji odpowiedzi."""
    # TYPE+STATUS(1) + BODY_LEN(2) + BODY + CRC(2)
    frame_len = 1 + 2 + len(body) + 2
    return estimate_duration(frame_len)
