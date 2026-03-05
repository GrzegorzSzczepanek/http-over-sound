#!/usr/bin/env python3
"""
SoundHTTP Client — klient HTTP over Sound

Wysyła żądania jako dźwięk FSK i dekoduje odpowiedzi.

Użycie:
  python client.py GET /ping
  python client.py POST /echo --body '{"hello":"world"}'
  python client.py GET / --simulate --response response.wav

Tryby:
  --simulate    Zapisz request jako WAV (bez odtwarzania)
  --live        Odtwórz przez głośnik, nasłuchuj odpowiedzi (wymaga pyaudio)
"""

import numpy as np
import struct
import sys
import os
import json
import time
import argparse

# ── Import modułów ──
from protocol import (
    FRAME_REQUEST, FRAME_RESPONSE,
    METHOD_GET, METHOD_POST,
    METHOD_NAMES, METHOD_FROM_NAME,
    STATUS_MAP, compact_to_http,
)

# ── Parametry FSK ──
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


# ── Kryptografia & kodowanie ──

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


def nibbles_to_bytes(nibbles: list[int]) -> bytes:
    if len(nibbles) % 2 != 0:
        nibbles.append(0)
    result = bytearray()
    for i in range(0, len(nibbles), 2):
        result.append((nibbles[i] << 4) | nibbles[i + 1])
    return bytes(result)


# ── FSK audio generation ──

def generate_tone(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = AMPLITUDE * np.sin(2 * np.pi * freq * t)
    fade = int(sr * 0.002)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


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


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        f.read(12)
        fmt_found = False; data_bytes = None
        sr = 44100; channels = 1; sw = 2
        while True:
            cid = f.read(4)
            if len(cid) < 4: break
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                fmt = f.read(csz)
                channels = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                sw = struct.unpack("<H", fmt[14:16])[0] // 8
                fmt_found = True
            elif cid == b"data":
                data_bytes = f.read(csz)
            else:
                f.read(csz)
        samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32767.0
        if channels == 2: samples = samples.reshape(-1, 2).mean(axis=1)
        return samples, sr


# ── Request encoding ──

def encode_request(method: int, path: str, body: bytes = b"",
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zakoduj żądanie HTTP jako audio FSK."""
    path_bytes = path.encode("utf-8")

    frame = bytearray()
    frame.append((FRAME_REQUEST << 4) | (method & 0xF))  # TYPE + METHOD
    frame.append(len(path_bytes))                          # PATH_LEN
    frame.extend(path_bytes)                               # PATH
    frame.extend(struct.pack(">H", len(body)))             # BODY_LEN
    frame.extend(body)                                     # BODY

    checksum = crc16(bytes(frame))
    frame.extend(struct.pack(">H", checksum))              # CRC16

    nibbles = PREAMBLE + nibbles_from_bytes(bytes(frame))

    # Info
    total_dur = len(nibbles) * (SYMBOL_DURATION + SILENCE_GAP)
    print(f"📡 Encoding request: {METHOD_NAMES.get(method, '?')} {path}")
    if body:
        print(f"   Body: {len(body)} bytes")
    print(f"   Frame: {len(bytes(frame))} bytes → {len(nibbles)} symbols → {total_dur:.1f}s")
    print(f"   CRC16: 0x{checksum:04X}")

    parts = [np.zeros(int(sr * 0.15))]  # silence before
    for nib in nibbles:
        parts.append(generate_tone(FREQS[nib], SYMBOL_DURATION, sr))
        parts.append(np.zeros(int(sr * SILENCE_GAP)))
    parts.append(np.zeros(int(sr * 0.15)))  # silence after

    return np.concatenate(parts)


# ── Response decoding ──

def detect_nibble(segment: np.ndarray, sr: int) -> tuple[int, float]:
    windowed = segment * np.hanning(len(segment))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    energies = np.zeros(16)
    bw = FREQ_STEP * 0.4
    for i, tf in enumerate(FREQS):
        mask = (freqs >= tf - bw) & (freqs <= tf + bw)
        if mask.any():
            energies[i] = np.max(fft[mask])
    best = np.argmax(energies)
    total = energies.sum()
    return int(best), (energies[best] / total if total > 0 else 0)


def find_preamble(audio: np.ndarray, sr: int) -> int | None:
    sym, _, step = calc_steps(sr)
    scan = step // 4
    max_off = len(audio) - step * len(PREAMBLE)
    best_off = None; best_score = 0
    off = 0
    while off < max_off:
        match = 0; conf_sum = 0
        for i, exp in enumerate(PREAMBLE):
            s = off + i * step; e = s + sym
            if e > len(audio): break
            nib, c = detect_nibble(audio[s:e], sr)
            if nib == exp: match += 1; conf_sum