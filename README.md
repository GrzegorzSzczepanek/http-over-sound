# HTTP over Sound

Protokół HTTP przesyłany przez dźwięk (FSK-16). Klient wysyła żądania HTTP jako sygnał audio przez głośnik, serwer odbiera je mikrofonem, przetwarza i odpowiada dźwiękiem z powrotem.

## Jak to działa

```
Klient (Maszyna B)                     Serwer (Maszyna A)
──────────────────                     ──────────────────
1. Koduje request → FSK
2. Odtwarza przez głośnik  ═══🔊═══►  3. Mikrofon odbiera dźwięk
                                       4. Dekoduje FSK → HTTP request
                                       5. Przetwarza (router)
                                       6. Koduje response → FSK
7. Mikrofon odbiera dźwięk  ◄═══🔊═══ 7. Odtwarza przez głośnik
8. Dekoduje FSK → HTTP response
9. Wyświetla wynik
```

Parametry transmisji (konfigurowalne w `config.py`):
- **FSK-16** — 16 tonów (1000–4000 Hz), każdy koduje 4 bity (nibble)
- **~5.9 B/s** przy domyślnych ustawieniach (80ms/symbol + 5ms gap)
- **Preambuła** `0xAA55AA55` do synchronizacji
- **CRC-16** do weryfikacji integralności

## Instalacja

### Wymagania
- Python 3.10+
- Mikrofon i głośnik (wbudowane w laptopie wystarczą)

### Setup

```bash
git clone <repo-url>
cd http-over-sound
chmod +x setup.sh
./setup.sh
```

Lub ręcznie:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pyaudio
```

> **macOS** — jeśli `pyaudio` się nie instaluje:
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

## Sprawdź urządzenia audio

Na obu maszynach sprawdź dostępne urządzenia:

```bash
python server.py --list-devices
```

Zanotuj indeksy mikrofonu (input) i głośnika (output).

## Uruchomienie — dwie maszyny, prawdziwy dźwięk

### Maszyna A — Serwer

```bash
source venv/bin/activate
python server.py --live
```

Lub z konkretnymi urządzeniami audio:

```bash
python server.py --live --mic 1 --speaker 0
```

Serwer nasłuchuje na mikrofonie i udostępnia endpointy:

| Metoda | Ścieżka  | Opis                    |
|--------|----------|-------------------------|
| GET    | `/`      | Info o serwerze         |
| GET    | `/ping`  | Pong + timestamp        |
| GET    | `/time`  | Aktualny czas i data    |
| POST   | `/echo`  | Echo wysłanego body     |
| POST   | `/msg`   | Wyślij wiadomość        |
| GET    | `/msg`   | Pobierz wiadomości      |
| POST   | `/store` | Zapisz klucz-wartość    |
| GET    | `/store` | Lista kluczy            |

### Maszyna B — Klient

Pojedyncze żądanie:

```bash
source venv/bin/activate
python client.py GET /ping
```

Inne przykłady:

```bash
python client.py GET /time
python client.py POST /echo --body '{"hello":"world"}'
python client.py POST /msg --body '{"from":"Alice","text":"Czesc!"}'
```

Z konkretnymi urządzeniami:

```bash
python client.py GET /ping --mic 2 --speaker 0
```

Tryb interaktywny (REPL):

```bash
python client.py --interactive
# Wpisuj komendy:
#   GET /ping
#   POST /echo {"hello":"world"}
#   quit
```

## Testowanie bez dźwięku (simulate)

Jeśli chcesz przetestować na jednej maszynie bez prawdziwego audio:

```bash
# 1. Klient zapisuje request jako WAV:
python client.py GET /ping --simulate

# 2. Serwer przetwarza WAV i zapisuje odpowiedź:
python server.py --simulate request_get_ping.wav

# 3. Klient dekoduje odpowiedź z WAV:
python client.py --decode-response request_get_ping_response.wav
```

## Testowanie na jednej maszynie (loopback audio)

Użyj wirtualnego kabla audio, aby przekierować wyjście głośnika na wejście mikrofonu.

### macOS — BlackHole

```bash
brew install blackhole-2ch
```

1. Otwórz **Audio MIDI Setup** (`/Applications/Utilities/`)
2. Utwórz **Multi-Output Device** (BlackHole + głośnik) — żebyś słyszał dźwięk
3. Utwórz **Aggregate Device** (BlackHole + mikrofon)
4. Sprawdź indeksy: `python server.py --list-devices`

```bash
# Terminal 1 — serwer (mikrofon = BlackHole):
python server.py --live --mic <blackhole_index>

# Terminal 2 — klient (głośnik = BlackHole):
python client.py GET /ping --speaker <blackhole_index>
```

### Linux — PulseAudio

```bash
pactl load-module module-null-sink sink_name=virtual_speaker
pactl load-module module-loopback source=virtual_speaker.monitor
```

## Wskazówki

- **Głośniki na max** — sygnał FSK musi być wyraźny
- **Maszyny blisko siebie** (< 1m) — im bliżej, tym lepiej
- **Cisza w pomieszczeniu** — hałas tła zakłóca dekodowanie
- Krótkie requesty (np. `/ping`) przechodzą w kilka sekund
- Większe body = dłuższa transmisja (~5.9 B/s)

## Struktura projektu

| Plik              | Opis                                      |
|-------------------|-------------------------------------------|
| `config.py`       | Parametry transmisji FSK                  |
| `encoder.py`      | Enkoder bajtów → audio FSK               |
| `decoder.py`      | Dekoder audio FSK → bajty                |
| `modem.py`        | Warstwa modemowa (encode/decode + I/O)    |
| `protocol.py`     | Protokół ramek (request/response)         |
| `server.py`       | Serwer HTTP-over-Sound                    |
| `client.py`       | Klient HTTP-over-Sound                    |
| `test_roundtrip.py` | Test encode→decode roundtrip            |