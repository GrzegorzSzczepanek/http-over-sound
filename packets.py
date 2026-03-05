#!/usr/bin/env python3
"""
packets.py — Warstwa pakietowa z retransmisją (Stop-and-Wait ARQ)

Dzieli dane na małe pakiety, każdy z własnym CRC-16.
ACK/NAK przez specjalne tony poza pasmem FSK (5000 Hz / 400 Hz).
Automatyczna retransmisja do 3 prób na pakiet.

Format pakietu danych:
  [PREAMBLE:8nib] [SEQ:2nib] [FLAGS:2nib] [LEN:2nib] [DATA...] [CRC16:4nib]

  SEQ:   0-255 numer sekwencyjny
  FLAGS: bit0 = MORE (więcej pakietów)
  LEN:   długość payload w bajtach (0-255)
  CRC16: nad SEQ+FLAGS+LEN+DATA

Sygnały ACK/NAK (poza pasmem danych 1000-4000 Hz):
  ACK: podwójny ton 5000 Hz (2×200ms)
  NAK: podwójny ton  400 Hz (2×200ms)

Użycie (live mode):
  session = LiveSession(pa, sr, mic_idx, spk_idx, log_fn)
  session.send_data(frame_bytes)        # nadawca
  data = session.receive_data(audio)    # odbiorca
"""

import numpy as np
import struct
import time

from modem import (
    SAMPLE_RATE, PREAMBLE, AMPLITUDE, FREQS,
    SYMBOL_DURATION, SILENCE_GAP, FREQ_STEP,
    crc16, nibbles_from_bytes, nibbles_to_bytes,
    generate_tone, _nibbles_to_audio,
    find_preamble, decode_nibbles, calc_steps,
)


# ══════════════════════════════════════════════════════════════════════
#  STAŁE PAKIETOWE
# ══════════════════════════════════════════════════════════════════════

PACKET_MAX_PAYLOAD = 32       # max bajtów danych na pakiet
MAX_RETRIES = 3               # max retransmisji per pakiet
ACK_TIMEOUT = 5.0             # sekundy czekania na ACK/NAK
PACKET_LISTEN_TIMEOUT = 15.0  # sekundy na odebranie pakietu
INTER_PACKET_DELAY = 0.4      # pauza między pakietami

# Tony ACK/NAK — poza pasmem danych (1000-4000 Hz)
ACK_FREQ = 5000.0             # Hz — wyraźny "beep"
NAK_FREQ = 400.0              # Hz — niski "boop"
ACKNAK_DURATION = 0.20        # czas trwania jednego tonu
ACKNAK_GAP = 0.05             # przerwa między podwójnym tonem

FLAG_MORE = 0x01              # więcej pakietów w kolejce


# ══════════════════════════════════════════════════════════════════════
#  ACK / NAK — generowanie i detekcja
# ══════════════════════════════════════════════════════════════════════

def generate_ack_signal(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generuj sygnał ACK: podwójny ton 5000 Hz."""
    silence_pad = np.zeros(int(sr * 0.08))
    tone = generate_tone(ACK_FREQ, ACKNAK_DURATION, sr)
    gap = np.zeros(int(sr * ACKNAK_GAP))
    return np.concatenate([silence_pad, tone, gap, tone, silence_pad])


def generate_nak_signal(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generuj sygnał NAK: podwójny ton 400 Hz."""
    silence_pad = np.zeros(int(sr * 0.08))
    tone = generate_tone(NAK_FREQ, ACKNAK_DURATION, sr)
    gap = np.zeros(int(sr * ACKNAK_GAP))
    return np.concatenate([silence_pad, tone, gap, tone, silence_pad])


def detect_ack_or_nak(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str | None:
    """
    Analizuj audio i wykryj ACK lub NAK.
    Zwraca 'ACK', 'NAK', lub None.
    """
    if len(audio) < int(sr * 0.05):
        return None

    windowed = audio * np.hanning(len(audio))
    nfft = max(len(audio), 8192)
    fft_mag = np.abs(np.fft.rfft(windowed, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)

    bw = 300  # Hz bandwidth

    # Energia ACK (5000 Hz)
    ack_mask = (freqs >= ACK_FREQ - bw) & (freqs <= ACK_FREQ + bw)
    ack_energy = np.sum(fft_mag[ack_mask] ** 2) if ack_mask.any() else 0

    # Energia NAK (400 Hz)
    nak_mask = (freqs >= max(50, NAK_FREQ - bw)) & (freqs <= NAK_FREQ + bw)
    nak_energy = np.sum(fft_mag[nak_mask] ** 2) if nak_mask.any() else 0

    # Energia w paśmie danych (1000-4000 Hz) — szum odniesienia
    data_mask = (freqs >= 800) & (freqs <= 4200)
    data_energy = np.sum(fft_mag[data_mask] ** 2) if data_mask.any() else 0

    total = ack_energy + nak_energy + data_energy + 1e-10

    ack_ratio = ack_energy / total
    nak_ratio = nak_energy / total

    if ack_ratio > 0.12 and ack_energy > nak_energy * 2.5:
        return 'ACK'
    if nak_ratio > 0.12 and nak_energy > ack_energy * 2.5:
        return 'NAK'

    return None


# ══════════════════════════════════════════════════════════════════════
#  PAKIETY: split / encode / decode / reassemble
# ══════════════════════════════════════════════════════════════════════

def split_into_packets(data: bytes) -> list[dict]:
    """Podziel dane na pakiety z per-packet CRC."""
    packets = []
    offset = 0
    seq = 0

    if not data:
        frame = bytearray([0, 0, 0])
        checksum = crc16(bytes(frame))
        frame.extend(struct.pack(">H", checksum))
        return [{'seq': 0, 'flags': 0, 'payload': b'',
                 'frame': bytes(frame), 'crc': checksum}]

    while offset < len(data):
        chunk = data[offset:offset + PACKET_MAX_PAYLOAD]
        offset += len(chunk)
        flags = FLAG_MORE if offset < len(data) else 0

        frame = bytearray()
        frame.append(seq & 0xFF)
        frame.append(flags)
        frame.append(len(chunk))
        frame.extend(chunk)
        checksum = crc16(bytes(frame))
        frame.extend(struct.pack(">H", checksum))

        packets.append({
            'seq': seq & 0xFF,
            'flags': flags,
            'payload': chunk,
            'frame': bytes(frame),
            'crc': checksum,
        })
        seq += 1

    return packets


def encode_packet_audio(packet: dict, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zakoduj pakiet jako audio FSK (z preambułą)."""
    nibbles = nibbles_from_bytes(packet['frame'])
    return _nibbles_to_audio(nibbles, sr)


def decode_packet_from_audio(audio: np.ndarray, sr: int = SAMPLE_RATE,
                             strict: bool = True) -> dict | None:
    """Dekoduj pakiet z audio FSK.
    strict=False: akceptuj pakiety nawet z błędnym CRC / payload_len."""
    _, _, step = calc_steps(sr)

    preamble_off = find_preamble(audio, sr)
    if preamble_off is None:
        return None

    pos = preamble_off + len(PREAMBLE) * step

    # SEQ (1B = 2 nibble)
    nibs = decode_nibbles(audio, pos, 2, sr)
    seq = (nibs[0][0] << 4) | nibs[1][0]
    pos += 2 * step

    # FLAGS (1B = 2 nibble)
    nibs = decode_nibbles(audio, pos, 2, sr)
    flags = (nibs[0][0] << 4) | nibs[1][0]
    pos += 2 * step

    # LEN (1B = 2 nibble)
    nibs = decode_nibbles(audio, pos, 2, sr)
    payload_len = (nibs[0][0] << 4) | nibs[1][0]
    pos += 2 * step

    if payload_len > 128:  # sanity check
        if strict:
            return None
        print(f"  [lenient] payload_len={payload_len} > 128, obcinam do 128")
        payload_len = min(payload_len, 128)

    # DATA
    payload = b''
    if payload_len > 0:
        data_nibs = decode_nibbles(audio, pos, payload_len * 2, sr)
        payload = nibbles_to_bytes([n[0] for n in data_nibs])[:payload_len]
        pos += payload_len * 2 * step

    # CRC16 (2B = 4 nibble)
    crc_nibs = decode_nibbles(audio, pos, 4, sr)
    received_crc = struct.unpack(">H", nibbles_to_bytes(
        [n[0] for n in crc_nibs]))[0]

    # Weryfikacja
    frame = bytearray()
    frame.append(seq)
    frame.append(flags)
    frame.append(payload_len)
    frame.extend(payload[:payload_len])
    computed_crc = crc16(bytes(frame))

    return {
        'seq': seq,
        'flags': flags,
        'payload': payload[:payload_len],
        'payload_len': payload_len,
        'crc_ok': received_crc == computed_crc,
        'crc_received': received_crc,
        'crc_computed': computed_crc,
        'more': bool(flags & FLAG_MORE),
    }


def reassemble_packets(packets: list[dict]) -> bytes:
    """Złóż pakiety z powrotem w dane (po SEQ)."""
    return b''.join(p['payload'] for p in sorted(packets, key=lambda p: p['seq']))


# ══════════════════════════════════════════════════════════════════════
#  LIVE SESSION — nadawanie i odbiór z retransmisją
# ══════════════════════════════════════════════════════════════════════

class LiveSession:
    """
    Sesja audio z protokołem pakietowym Stop-and-Wait ARQ.

    Obsługuje:
      - Podział danych na pakiety z CRC
      - Wysyłanie z ACK/NAK i retransmisją
      - Odbiór z weryfikacją CRC i żądaniem retransmisji
    """

    def __init__(self, pa, sr: int, mic_index, speaker_index,
                 log_fn=None, strict: bool = True):
        self.pa = pa
        self.sr = sr
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        self.log = log_fn or (lambda msg: print(f"  [pkt] {msg}"))
        self.strict = strict

    # ── Audio I/O ──

    def _play(self, audio: np.ndarray):
        """Odtwórz audio przez głośnik."""
        import pyaudio
        stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.sr,
            output=True, output_device_index=self.speaker_index,
        )
        out = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        stream.write(out.tobytes())
        stream.close()

    def _record_until_silence(self, timeout: float = 15.0,
                               signal_thresh: float = 0.003,
                               silence_chunks: int = 25) -> np.ndarray:
        """Nagraj z mikrofonu aż do ciszy po wykryciu sygnału."""
        import pyaudio
        CHUNK = int(self.sr * 0.05)  # 50ms
        stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.sr,
            input=True, input_device_index=self.mic_index,
            frames_per_buffer=CHUNK,
        )

        collected = []
        signal_detected = False
        silence_count = 0
        start = time.time()

        try:
            while time.time() - start < timeout:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
                collected.append(chunk)

                rms = np.sqrt(np.mean(chunk ** 2))
                if rms > signal_thresh:
                    signal_detected = True
                    silence_count = 0
                elif signal_detected:
                    silence_count += 1
                    if silence_count >= silence_chunks:
                        break

                # Status co 2s
                if not signal_detected and len(collected) % 40 == 0:
                    remaining = timeout - (time.time() - start)
                    print(f"\r  Czekam na sygnal... ({remaining:.0f}s)   ",
                          end="", flush=True)
        finally:
            stream.close()

        if not collected or not signal_detected:
            return np.array([], dtype=np.float64)
        return np.concatenate(collected)

    def _listen_for_ack(self, timeout: float = ACK_TIMEOUT) -> str | None:
        """Nasłuchuj krótkiego sygnału ACK/NAK."""
        audio = self._record_until_silence(
            timeout=timeout, signal_thresh=0.003, silence_chunks=8
        )
        if len(audio) == 0:
            return None
        return detect_ack_or_nak(audio, self.sr)

    # ── Wysyłanie danych ──

    def send_data(self, data: bytes) -> bool:
        """
        Wyślij dane jako pakiety z retransmisją.
        Zwraca True jeśli wszystkie pakiety potwierdzone (ACK).
        """
        packets = split_into_packets(data)
        total = len(packets)
        self.log(f"Wysylanie {len(data)}B w {total} pakietach "
                 f"(max {PACKET_MAX_PAYLOAD}B/pkt, do {MAX_RETRIES} retransmisji)")

        for pkt in packets:
            audio = encode_packet_audio(pkt, self.sr)
            pkt_dur = len(audio) / self.sr
            success = False

            for attempt in range(MAX_RETRIES + 1):
                more_str = "MORE" if pkt['flags'] & FLAG_MORE else "LAST"
                self.log(f"  PKT #{pkt['seq']}/{total-1} ({len(pkt['payload'])}B) "
                         f"[{more_str}] proba {attempt + 1}/{MAX_RETRIES + 1} "
                         f"({pkt_dur:.1f}s audio)")

                self._play(audio)
                time.sleep(0.2)

                result = self._listen_for_ack()

                if result == 'ACK':
                    self.log(f"  PKT #{pkt['seq']} -> ACK")
                    success = True
                    break
                elif result == 'NAK':
                    self.log(f"  PKT #{pkt['seq']} -> NAK (retransmisja...)")
                    time.sleep(0.3)
                else:
                    self.log(f"  PKT #{pkt['seq']} -> TIMEOUT (retransmisja...)")
                    time.sleep(0.3)

            if not success:
                self.log(f"  PKT #{pkt['seq']} FAILED po {MAX_RETRIES + 1} probach!")
                return False

            if pkt['flags'] & FLAG_MORE:
                time.sleep(INTER_PACKET_DELAY)

        self.log(f"Wszystkie {total} pakietow wyslane pomyslnie!")
        return True

    # ── Odbieranie danych ──

    def receive_data(self, first_audio: np.ndarray | None = None) -> bytes | None:
        """
        Odbierz pakiety z ACK/NAK i retransmisją.
        first_audio: audio pierwszego pakietu (jeśli już nagrane przez serwer).
        Zwraca złożone dane lub None.
        W trybie lenient (strict=False): akceptuje pakiety z CRC FAIL.
        """
        received = []
        expected_seq = 0

        while True:
            # ── Pobierz audio ──
            if first_audio is not None and len(first_audio) > 0:
                audio = first_audio
                first_audio = None  # tylko pierwsze użycie
            else:
                self.log(f"Czekam na pakiet #{expected_seq}...")
                audio = self._record_until_silence(timeout=PACKET_LISTEN_TIMEOUT)
                if len(audio) == 0:
                    self.log("Timeout — brak pakietu")
                    return self._try_reassemble(received)

            # ── Dekoduj pakiet ──
            pkt = decode_packet_from_audio(audio, self.sr, strict=self.strict)

            # Jeśli nie udało się zdekodować — NAK + czekaj na retransmisję
            if pkt is None or (not pkt['crc_ok'] and self.strict):
                reason = "decode fail" if pkt is None else \
                    f"CRC FAIL (recv=0x{pkt['crc_received']:04X} " \
                    f"calc=0x{pkt['crc_computed']:04X})"
                self.log(f"PKT #{expected_seq} {reason} -> NAK")
                self._play(generate_nak_signal(self.sr))
                time.sleep(0.1)

                # Jedna szansa na retransmisję
                self.log(f"Czekam na retransmisje #{expected_seq}...")
                audio = self._record_until_silence(timeout=PACKET_LISTEN_TIMEOUT)
                if len(audio) == 0:
                    # W lenient: jeśli mieliśmy pkt z bad CRC, dodaj go
                    if not self.strict and pkt is not None:
                        self.log(f"  [lenient] akceptuję PKT #{pkt['seq']} mimo CRC FAIL")
                        received.append(pkt)
                    return self._try_reassemble(received)

                pkt = decode_packet_from_audio(audio, self.sr, strict=self.strict)
                if pkt is None or (not pkt['crc_ok'] and self.strict):
                    if not self.strict and pkt is not None:
                        self.log(f"  [lenient] akceptuję PKT #{pkt['seq']} mimo CRC FAIL")
                        received.append(pkt)
                        self._play(generate_ack_signal(self.sr))
                        expected_seq = pkt['seq'] + 1
                        if not pkt['more']:
                            break
                        time.sleep(0.15)
                        continue
                    self.log("Retransmisja tez nieudana — NAK")
                    self._play(generate_nak_signal(self.sr))
                    return self._try_reassemble(received)

            # ── Pakiet zaakceptowany ──
            crc_str = "CRC OK" if pkt['crc_ok'] else "CRC FAIL (lenient)"
            self.log(f"PKT #{pkt['seq']} ({pkt['payload_len']}B) "
                     f"{crc_str} (0x{pkt['crc_received']:04X}) -> ACK")
            received.append(pkt)
            expected_seq = pkt['seq'] + 1

            # Wyślij ACK
            self._play(generate_ack_signal(self.sr))

            # Ostatni pakiet?
            if not pkt['more']:
                total_bytes = sum(p['payload_len'] for p in received)
                self.log(f"Ostatni pakiet. Odebrano {len(received)} pkt, "
                         f"{total_bytes}B danych.")
                break

            time.sleep(0.15)  # krótka pauza przed nasłuchiwaniem

        return reassemble_packets(received)

    def _try_reassemble(self, packets: list[dict]) -> bytes | None:
        """Próbuj złożyć co się da z odebranych pakietów."""
        if not packets:
            return None
        self.log(f"Skladam {len(packets)} odebranych pakietow (czesciowe)")
        return reassemble_packets(packets)
