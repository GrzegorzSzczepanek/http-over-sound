"""
Acoustic Modem — Wspólna konfiguracja

Wszystkie parametry transmisji w jednym miejscu.
Enkoder i dekoder importują stąd.
"""

SAMPLE_RATE = 44100          # Hz
SYMBOL_DURATION = 0.08       # sekundy na nibble
SILENCE_GAP = 0.005          # cisza między symbolami

FREQ_BASE = 1000             # Hz
FREQ_STEP = 200              # Hz
NUM_SYMBOLS = 16             # 4 bity = 16 możliwych wartości

AMPLITUDE = 0.8

# Preambuła: 0xAA55AA55
PREAMBLE = [0xA, 0xA, 0x5, 0x5, 0xA, 0xA, 0x5, 0x5]

# Wyliczone
FREQS = [FREQ_BASE + i * FREQ_STEP for i in range(NUM_SYMBOLS)]
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
GAP_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP)
STEP_SAMPLES = SYMBOL_SAMPLES + GAP_SAMPLES

# Prędkości
NIBBLES_PER_SEC = 1.0 / (SYMBOL_DURATION + SILENCE_GAP)
BYTES_PER_SEC = NIBBLES_PER_SEC / 2

print(f"[config] FSK-16: {FREQS[0]}–{FREQS[-1]} Hz")
print(f"[config] Symbol: {SYMBOL_DURATION*1000:.0f}ms + {SILENCE_GAP*1000:.0f}ms gap")
print(f"[config] Speed: {NIBBLES_PER_SEC:.1f} nibbles/s = {BYTES_PER_SEC:.1f} B/s")
