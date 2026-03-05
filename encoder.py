#!/usr/bin/env python3
"""
Acoustic Modem — Decoder (NumPy FFT)
Dekoduje audio FSK z powrotem do pliku.

Na razie używa FFT peak detection (numpy).
Docelowo: podmienić detect_nibble() na sieć neuronową.

Obsługuje:
  - Automatyczne wykrywanie preambuły
  - Dekodowanie nibble'ów z FFT
  - Weryfikację CRC16
  - Zapis odtworzonego pliku
  - [TEST] Nakładanie szumu / zakłóceń na audio przed dekodowaniem
"""

import argparse
import os
import struct
import sys

import numpy as np

# ── Parametry (muszą zgadzać się z encoderem!) ────────────────────────
SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.08
SILENCE_GAP = 0.005

FREQ_BASE = 1000
FREQ_STEP = 200
FREQS = np.array([FREQ_BASE + i * FREQ_STEP for i in range(16)])

PREAMBLE = [0xA, 0xA, 0x5, 0x5, 0xA, 0xA, 0x5, 0x5]

SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
GAP_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP)
STEP_SAMPLES = SYMBOL_SAMPLES + GAP_SAMPLES


# ══════════════════════════════════════════════════════════════════════
#  NOISE / DISTORTION ENGINE — symulacja warunków rzeczywistych
# ══════════════════════════════════════════════════════════════════════

def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Biały szum gaussowski o zadanym SNR (dB)."""
    sig_power = np.mean(audio ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def add_pink_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Szum różowy (1/f) — bardziej realistyczny niż biały."""
    n = len(audio)
    white = np.random.normal(0, 1, n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    freqs[0] = 1
    fft_pink = fft_white / np.sqrt(freqs)
    pink = np.fft.irfft(fft_pink, n)
    sig_power = np.mean(audio ** 2)
    pink_power = np.mean(pink ** 2)
    if pink_power > 0:
        scale = np.sqrt(sig_power / (10 ** (snr_db / 10)) / pink_power)
        pink *= scale
    return audio + pink


def add_echo(audio: np.ndarray, delay_ms: float, decay: float) -> np.ndarray:
    """Symuluje echo/odbicie w pomieszczeniu."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    echoed = np.copy(audio)
    if delay_samples < len(audio):
        echoed[delay_samples:] += decay * audio[:len(audio) - delay_samples]
    return echoed


def add_multi_echo(audio: np.ndarray, room_size: str = "medium") -> np.ndarray:
    """Wielokrotne odbicia — symulacja pokoju."""
    configs = {
        "small":  [(15, 0.4), (30, 0.2), (50, 0.1)],
        "medium": [(30, 0.35), (65, 0.2), (100, 0.1), (150, 0.05)],
        "large":  [(50, 0.3), (120, 0.2), (200, 0.15), (350, 0.08), (500, 0.04)],
    }
    echoes = configs.get(room_size, configs["medium"])
    result = np.copy(audio)
    for delay_ms, decay in echoes:
        result = add_echo(result, delay_ms, decay)
    return result


def add_frequency_interference(audio: np.ndarray, freq: float,
                                amplitude: float = 0.3) -> np.ndarray:
    """Stała interferencja na jednej częstotliwości (np. buczenie 50Hz)."""
    t = np.arange(len(audio)) / SAMPLE_RATE
    interference = amplitude * np.sin(2 * np.pi * freq * t)
    return audio + interference


def add_band_noise(audio: np.ndarray, low_hz: float, high_hz: float,
                   snr_db: float) -> np.ndarray:
    """Szum ograniczony do pasa częstotliwości (np. w paśmie sygnału)."""
    n = len(audio)
    white = np.random.normal(0, 1, n)
    fft_w = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    fft_w[~mask] = 0
    band = np.fft.irfft(fft_w, n)
    sig_power = np.mean(audio ** 2)
    band_power = np.mean(band ** 2)
    if band_power > 0:
        scale = np.sqrt(sig_power / (10 ** (snr_db / 10)) / band_power)
        band *= scale
    return audio + band


def add_clipping(audio: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Symulacja przesterowania mikrofonu."""
    return np.clip(audio, -threshold, threshold)


def add_dropout(audio: np.ndarray, prob: float = 0.01,
                max_ms: float = 20) -> np.ndarray:
    """Losowe wypadanie próbek (glitche, mikro-przerwy)."""
    result = np.copy(audio)
    max_samples = int(SAMPLE_RATE * max_ms / 1000)
    i = 0
    while i < len(result):
        if np.random.random() < prob:
            dropout_len = np.random.randint(1, max_samples + 1)
            end = min(i + dropout_len, len(result))
            result[i:end] = 0
            i = end
        else:
            i += 1
    return result


def add_time_stretch(audio: np.ndarray, factor: float = 1.02) -> np.ndarray:
    """Lekkie rozciągnięcie/skompresowanie w czasie (drift zegara)."""
    n_out = int(len(audio) * factor)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio)


def add_dc_offset(audio: np.ndarray, offset: float = 0.05) -> np.ndarray:
    """Przesunięcie DC (słaby mikrofon)."""
    return audio + offset


def apply_noise_preset(audio: np.ndarray, preset: str,
                       seed: int | None = None) -> np.ndarray:
    """
    Predefiniowane profile zakłóceń.

    Presety:
      clean      — brak zmian (baseline)
      mild       — lekki szum + małe echo
      moderate   — szum + echo + 50Hz hum
      harsh      — dużo szumu + duży pokój + clipping
      realistic  — szum różowy + echo + band noise + drift
      hell       — wszystko naraz, żeby zobaczyć kiedy dekoder się łamie
    """
    if seed is not None:
        np.random.seed(seed)

    original_power = np.mean(audio ** 2)

    presets = {
        "clean": lambda a: a,

        "mild": lambda a: add_echo(
            add_white_noise(a, snr_db=30),
            delay_ms=20, decay=0.15
        ),

        "moderate": lambda a: add_frequency_interference(
            add_multi_echo(
                add_white_noise(a, snr_db=20),
                room_size="small"
            ),
            freq=50, amplitude=0.05
        ),

        "harsh": lambda a: add_clipping(
            add_multi_echo(
                add_white_noise(a, snr_db=10),
                room_size="large"
            ),
            threshold=0.6
        ),

        "realistic": lambda a: add_time_stretch(
            add_band_noise(
                add_multi_echo(
                    add_pink_noise(a, snr_db=15),
                    room_size="medium"
                ),
                low_hz=800, high_hz=4500, snr_db=25
            ),
            factor=1.005
        ),

        "hell": lambda a: add_dropout(
            add_clipping(
                add_time_stretch(
                    add_frequency_interference(
                        add_multi_echo(
                            add_band_noise(
                                add_white_noise(a, snr_db=5),
                                low_hz=800, high_hz=4500, snr_db=10
                            ),
                            room_size="large"
                        ),
                        freq=50, amplitude=0.1
                    ),
                    factor=1.01
                ),
                threshold=0.5
            ),
            prob=0.005, max_ms=10
        ),
    }

    if preset not in presets:
        print(f"❌ Nieznany preset: '{preset}'")
        print(f"   Dostępne: {', '.join(presets.keys())}")
        sys.exit(1)

    print(f"🔊 Noise preset: {preset}")
    result = presets[preset](audio)

    if preset != "clean":
        # SNR estimation — handle length mismatch from time stretch
        min_len = min(len(result), len(audio))
        if min_len > 0:
            noise_only = result[:min_len] - audio[:min_len]
            noise_power = np.mean(noise_only ** 2)
            if noise_power > 0:
                actual_snr = 10 * np.log10(original_power / noise_power)
                print(f"   Efektywny SNR: {actual_snr:.1f} dB")
        if len(result) != len(audio):
            print(f"   Długość zmieniona: {len(audio)} → {len(result)} "
                  f"({len(result)/len(audio)*100:.1f}%)")
        print(f"   Peak amplitude: {np.max(np.abs(result)):.3f}")

    return result


def write_wav(path: str, data: np.ndarray, sr: int):
    """Zapisz jako 16-bit WAV."""
    audio_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    num_channels = 1
    sample_width = 2
    num_frames = len(audio_int16)
    data_size = num_frames * num_channels * sample_width

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * num_channels * sample_width))
        f.write(struct.pack("<H", num_channels * sample_width))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())


# ══════════════════════════════════════════════════════════════════════
#  DEKODOWANIE
# ══════════════════════════════════════════════════════════════════════

def crc16(data: bytes) -> int:
    """CRC-16-CCITT — identyczny jak w enkoderze."""
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


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Wczytaj WAV (16-bit PCM mono) bez zewnętrznych bibliotek."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Nie jest to plik WAV (brak RIFF)")
        f.read(4)
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Nie jest to plik WAV (brak WAVE)")

        fmt_found = False
        data_bytes = None
        sr = 44100
        num_channels = 1
        sample_width = 2

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                sr = struct.unpack("<I", fmt_data[4:8])[0]
                sample_width = struct.unpack("<H", fmt_data[14:16])[0] // 8
                fmt_found = True
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

        if not fmt_found or data_bytes is None:
            raise ValueError("Nieprawidłowy plik WAV")

        if sample_width == 2:
            samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32767.0
        elif sample_width == 1:
            samples = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float64) - 128) / 128.0
        else:
            raise ValueError(f"Nieobsługiwana głębia: {sample_width * 8} bit")

        if num_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        return samples, sr


def detect_nibble_fft(segment: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, float]:
    """
    Wykryj nibble z segmentu audio za pomocą FFT.
    Zwraca: (nibble_value, confidence)

    >>> TU DOCELOWO PODMIENIC NA SIEĆ NEURONOWĄ <<<
    """
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


def detect_nibble_nn(segment: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, float]:
    """
    PLACEHOLDER — tu wstawicie swoją sieć neuronową.
    """
    return detect_nibble_fft(segment, sr)


detect_nibble = detect_nibble_fft


def find_preamble(audio: np.ndarray, sr: int = SAMPLE_RATE) -> int | None:
    """Znajdź początek preambuły w audio (sliding window)."""
    print("🔍 Szukam preambuły...", flush=True)

    scan_step = STEP_SAMPLES // 4
    max_offset = len(audio) - STEP_SAMPLES * len(PREAMBLE)

    best_offset = None
    best_score = 0

    offset = 0
    while offset < max_offset:
        match_count = 0
        total_conf = 0

        for i, expected in enumerate(PREAMBLE):
            start = offset + i * STEP_SAMPLES
            end = start + SYMBOL_SAMPLES
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

        if (offset // scan_step) % 2000 == 0:
            pct = offset / max_offset * 100
            print(f"\r  Scanning: {pct:5.1f}%", end="", flush=True)

    print()

    if best_offset is not None:
        print(f"✅ Preambuła znaleziona na pozycji {best_offset} "
              f"(t={best_offset/sr:.3f}s)")
    else:
        print("❌ Nie znaleziono preambuły!")

    return best_offset


def decode_symbols(audio: np.ndarray, start_offset: int, num_symbols: int,
                   sr: int = SAMPLE_RATE) -> list[tuple[int, float]]:
    """Dekoduj N symboli od danego offsetu."""
    results = []
    for i in range(num_symbols):
        seg_start = start_offset + i * STEP_SAMPLES
        seg_end = seg_start + SYMBOL_SAMPLES
        if seg_end > len(audio):
            print(f"\n⚠️  Audio za krótkie — brakuje symbolu {i}")
            results.append((0, 0.0))
            continue
        segment = audio[seg_start:seg_end]
        nibble, conf = detect_nibble(segment, sr)
        results.append((nibble, conf))
    return results


def nibbles_to_bytes(nibbles: list[int]) -> bytes:
    """Lista nibble'ów → bajty (high nibble first)."""
    if len(nibbles) % 2 != 0:
        nibbles.append(0)
    result = bytearray()
    for i in range(0, len(nibbles), 2):
        byte = (nibbles[i] << 4) | nibbles[i + 1]
        result.append(byte)
    return bytes(result)


def decode_file(input_path: str, output_path: str, noise_preset: str = "clean",
                save_noisy: str | None = None, noise_seed: int | None = None):
    """Główna funkcja — .wav → plik"""

    print(f"🎵 Wczytuję: {input_path}")
    audio, sr = read_wav(input_path)
    print(f"   Próbki: {len(audio)}, SR: {sr}, Czas: {len(audio)/sr:.2f}s")

    if sr != SAMPLE_RATE:
        print(f"⚠️  Uwaga: SR={sr} != oczekiwane {SAMPLE_RATE}")

    # ── Nakładanie szumu ──
    audio = apply_noise_preset(audio, noise_preset, seed=noise_seed)

    if save_noisy and noise_preset != "clean":
        write_wav(save_noisy, audio, sr)
        print(f"💾 Zaszumione audio zapisane: {save_noisy}")

    # Dekoduj
    _decode_audio(audio, sr, output_path)


def _decode_audio(audio: np.ndarray, sr: int, output_path: str):
    """Wspólna logika dekodowania z tablicy audio."""
    preamble_offset = find_preamble(audio, sr)
    if preamble_offset is None:
        print("💀 Nie mogę zdekodować — brak preambuły")
        return False

    data_offset = preamble_offset + len(PREAMBLE) * STEP_SAMPLES

    print("📏 Dekoduję rozmiar pliku...")
    size_symbols = decode_symbols(audio, data_offset, 8, sr)
    size_nibbles = [s[0] for s in size_symbols]
    size_bytes = nibbles_to_bytes(size_nibbles)
    file_size = struct.unpack(">I", size_bytes)[0]

    avg_conf = np.mean([s[1] for s in size_symbols])
    print(f"   Rozmiar: {file_size} bajtów (confidence: {avg_conf:.2f})")

    if file_size > 10_000_000:
        print("❌ Rozmiar podejrzanie duży — prawdopodobnie błąd dekodowania")
        return False

    data_nibble_count = file_size * 2
    crc_nibble_count = 4
    total_data_nibbles = data_nibble_count + crc_nibble_count

    print(f"📦 Dekoduję {file_size} bajtów danych + CRC16...")

    payload_offset = data_offset + 8 * STEP_SAMPLES
    payload_symbols = decode_symbols(audio, payload_offset, total_data_nibbles, sr)

    data_nibbles = [s[0] for s in payload_symbols[:data_nibble_count]]
    crc_nibbles = [s[0] for s in payload_symbols[data_nibble_count:]]

    file_data = nibbles_to_bytes(data_nibbles)
    received_crc_bytes = nibbles_to_bytes(crc_nibbles)
    received_crc = struct.unpack(">H", received_crc_bytes)[0]

    computed_crc = crc16(file_data)

    print(f"\n🔒 CRC16 odebrany:  0x{received_crc:04X}")
    print(f"🔒 CRC16 obliczony: 0x{computed_crc:04X}")

    crc_ok = received_crc == computed_crc
    if crc_ok:
        print("✅ CRC OK — dane nienaruszone!")
    else:
        print("⚠️  CRC MISMATCH — dane mogą zawierać błędy!")

    all_confs = [s[1] for s in payload_symbols[:data_nibble_count]]
    print(f"\n📊 Confidence statystyki:")
    print(f"   Min:  {min(all_confs):.3f}")
    print(f"   Max:  {max(all_confs):.3f}")
    print(f"   Mean: {np.mean(all_confs):.3f}")
    print(f"   <0.5: {sum(1 for c in all_confs if c < 0.5)} symboli")

    with open(output_path, "wb") as f:
        f.write(file_data)

    print(f"\n✅ Zapisano: {output_path} ({len(file_data)} bajtów)")
    return crc_ok


def apply_custom_noise(audio: np.ndarray, effects: list[str]) -> np.ndarray:
    """Aplikuj łańcuch efektów z CLI."""
    for effect in effects:
        parts = effect.split(":")
        name = parts[0]
        print(f"  + {effect}")

        if name == "white":
            snr = float(parts[1]) if len(parts) > 1 else 20
            audio = add_white_noise(audio, snr)
        elif name == "pink":
            snr = float(parts[1]) if len(parts) > 1 else 20
            audio = add_pink_noise(audio, snr)
        elif name == "echo":
            delay = float(parts[1]) if len(parts) > 1 else 30
            decay = float(parts[2]) if len(parts) > 2 else 0.3
            audio = add_echo(audio, delay, decay)
        elif name == "room":
            size = parts[1] if len(parts) > 1 else "medium"
            audio = add_multi_echo(audio, size)
        elif name == "hum":
            freq = float(parts[1]) if len(parts) > 1 else 50
            amp = float(parts[2]) if len(parts) > 2 else 0.1
            audio = add_frequency_interference(audio, freq, amp)
        elif name == "band":
            lo = float(parts[1]) if len(parts) > 1 else 800
            hi = float(parts[2]) if len(parts) > 2 else 4500
            snr = float(parts[3]) if len(parts) > 3 else 20
            audio = add_band_noise(audio, lo, hi, snr)
        elif name == "clip":
            thresh = float(parts[1]) if len(parts) > 1 else 0.7
            audio = add_clipping(audio, thresh)
        elif name == "dropout":
            prob = float(parts[1]) if len(parts) > 1 else 0.01
            ms = float(parts[2]) if len(parts) > 2 else 20
            audio = add_dropout(audio, prob, ms)
        elif name == "stretch":
            factor = float(parts[1]) if len(parts) > 1 else 1.02
            audio = add_time_stretch(audio, factor)
        elif name == "dc":
            offset = float(parts[1]) if len(parts) > 1 else 0.05
            audio = add_dc_offset(audio, offset)
        else:
            print(f"  ⚠️  Nieznany efekt: {name}, pomijam")

    return audio


def run_sweep(input_path: str, output_base: str, seed: int | None = None):
    """Testuj wszystkie presety szumu i wydrukuj raport porównawczy."""
    presets = ["clean", "mild", "moderate", "harsh", "realistic", "hell"]

    print("=" * 70)
    print("  NOISE SWEEP — testowanie odporności dekodera")
    print("=" * 70)

    audio_orig, sr = read_wav(input_path)
    print(f"🎵 Źródło: {input_path} ({len(audio_orig)/sr:.2f}s)\n")

    results = []

    for preset in presets:
        print(f"\n{'─' * 60}")
        print(f"  Preset: {preset.upper()}")
        print(f"{'─' * 60}")

        audio = apply_noise_preset(np.copy(audio_orig), preset, seed=seed)

        preamble_offset = find_preamble(audio, sr)
        if preamble_offset is None:
            results.append({
                "preset": preset, "preamble": False,
                "crc_ok": False, "mean_conf": 0, "min_conf": 0,
                "low_conf_count": 0, "errors": "no preamble"
            })
            continue

        data_offset = preamble_offset + len(PREAMBLE) * STEP_SAMPLES
        size_symbols = decode_symbols(audio, data_offset, 8, sr)
        size_nibbles = [s[0] for s in size_symbols]
        size_bytes = nibbles_to_bytes(size_nibbles)
        file_size = struct.unpack(">I", size_bytes)[0]

        if file_size > 10_000_000:
            results.append({
                "preset": preset, "preamble": True,
                "crc_ok": False, "mean_conf": 0, "min_conf": 0,
                "low_conf_count": 0, "errors": "bad size"
            })
            continue

        data_nibble_count = file_size * 2
        crc_nibble_count = 4
        total = data_nibble_count + crc_nibble_count

        payload_offset = data_offset + 8 * STEP_SAMPLES
        payload_symbols = decode_symbols(audio, payload_offset, total, sr)

        data_nibbles = [s[0] for s in payload_symbols[:data_nibble_count]]
        crc_nibbles = [s[0] for s in payload_symbols[data_nibble_count:]]

        file_data = nibbles_to_bytes(data_nibbles)
        received_crc = struct.unpack(">H", nibbles_to_bytes(crc_nibbles))[0]
        computed_crc = crc16(file_data)

        confs = [s[1] for s in payload_symbols[:data_nibble_count]]
        low_conf = sum(1 for c in confs if c < 0.5)

        results.append({
            "preset": preset,
            "preamble": True,
            "crc_ok": received_crc == computed_crc,
            "mean_conf": np.mean(confs) if confs else 0,
            "min_conf": min(confs) if confs else 0,
            "low_conf_count": low_conf,
            "errors": "none" if received_crc == computed_crc else "CRC fail"
        })

    # ── Raport ──
    print(f"\n\n{'=' * 70}")
    print("  RAPORT SWEEP")
    print(f"{'=' * 70}")
    print(f"{'Preset':<12} {'Preambuła':<11} {'CRC':<8} {'Conf mean':<11} "
          f"{'Conf min':<10} {'Low(<0.5)':<10} {'Status'}")
    print(f"{'─' * 70}")

    for r in results:
        pre = "✅" if r["preamble"] else "❌"
        crc = "✅" if r["crc_ok"] else "❌"
        mean_c = f"{r['mean_conf']:.3f}" if r['mean_conf'] > 0 else "—"
        min_c = f"{r['min_conf']:.3f}" if r['min_conf'] > 0 else "—"
        low = str(r["low_conf_count"])
        status = "PASS ✅" if r["crc_ok"] else "FAIL ❌"
        print(f"{r['preset']:<12} {pre:<11} {crc:<8} {mean_c:<11} "
              f"{min_c:<10} {low:<10} {status}")

    print(f"{'─' * 70}")
    passed = sum(1 for r in results if r["crc_ok"])
    print(f"\n  Wynik: {passed}/{len(results)} presetów zdekodowanych poprawnie")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Acoustic Modem — Decoder z opcjami szumu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presety szumu:
  clean      Brak zakłóceń (baseline)
  mild       Lekki szum biały + małe echo (SNR ~30dB)
  moderate   Szum + echo pokoju + buczenie 50Hz (SNR ~20dB)
  harsh      Mocny szum + duży pokój + clipping (SNR ~10dB)
  realistic  Szum różowy + echo + band noise + clock drift
  hell       Wszystko naraz — stress test dekodera

Pojedyncze efekty (--noise-custom):
  white:SNR       Biały szum (np. white:15)
  pink:SNR        Różowy szum
  echo:DELAY:DECAY Echo (np. echo:30:0.3)
  room:SIZE       Multi-echo (small/medium/large)
  hum:FREQ:AMP    Interferencja (np. hum:50:0.1)
  band:LO:HI:SNR  Band-limited noise
  clip:THRESHOLD   Clipping
  dropout:PROB:MS  Losowe wypadanie
  stretch:FACTOR   Clock drift
  dc:OFFSET        DC offset

Przykłady:
  python decoder.py signal.wav out.txt --noise moderate
  python decoder.py signal.wav out.txt --noise harsh --save-noisy harsh.wav
  python decoder.py signal.wav out.txt --noise-custom white:15 room:large clip:0.6
  python decoder.py signal.wav out.txt --sweep
        """
    )

    parser.add_argument("input", help="Plik audio WAV")
    parser.add_argument("output", nargs="?", default="decoded_output",
                        help="Plik wyjściowy (domyślnie: decoded_output)")
    parser.add_argument("--noise", default="clean",
                        choices=["clean", "mild", "moderate", "harsh",
                                 "realistic", "hell"],
                        help="Preset szumu (domyślnie: clean)")
    parser.add_argument("--noise-custom", nargs="+", metavar="EFFECT",
                        help="Własna kombinacja efektów")
    parser.add_argument("--save-noisy", metavar="PATH",
                        help="Zapisz zaszumione audio do pliku WAV")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed RNG dla powtarzalności")
    parser.add_argument("--sweep", action="store_true",
                        help="Testuj wszystkie presety i pokaż raport")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Plik nie istnieje: {args.input}")
        sys.exit(1)

    if args.sweep:
        run_sweep(args.input, args.output, seed=args.seed)
        return

    if args.noise_custom:
        print(f"🎵 Wczytuję: {args.input}")
        audio, sr = read_wav(args.input)
        print(f"   Próbki: {len(audio)}, SR: {sr}, Czas: {len(audio)/sr:.2f}s")

        if args.seed is not None:
            np.random.seed(args.seed)

        print("🔧 Custom noise chain:")
        audio = apply_custom_noise(audio, args.noise_custom)

        if args.save_noisy:
            write_wav(args.save_noisy, audio, sr)
            print(f"💾 Zapisano zaszumione audio: {args.save_noisy}")

        _decode_audio(audio, sr, args.output)
        return

    decode_file(args.input, args.output,
                noise_preset=args.noise,
                save_noisy=args.save_noisy,
                noise_seed=args.seed)


if __name__ == "__main__":
    main()
