#!/bin/bash
#
# setup.sh — Instalacja SoundHTTP (klient lub serwer)
#
# Użycie:
#   chmod +x setup.sh
#   ./setup.sh           # zainstaluj wszystko
#   ./setup.sh server    # uruchom serwer (live)
#   ./setup.sh client    # uruchom klienta (interactive)
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# Kolory
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       SoundHTTP — HTTP over Sound                ║${NC}"
    echo -e "${CYAN}║       Setup & Launcher                           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ── INSTALACJA ──

install() {
    header
    echo -e "${YELLOW}[1/4]${NC} Sprawdzam Python..."
    
    if ! command -v python3 &>/dev/null; then
        echo -e "${RED}[x] Python3 nie znaleziony! Zainstaluj python3.${NC}"
        exit 1
    fi
    
    PYVER=$(python3 --version 2>&1)
    echo -e "  ${GREEN}OK${NC} $PYVER"

    echo -e "${YELLOW}[2/4]${NC} Tworzę virtualenv..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        echo -e "  ${GREEN}OK${NC} Utworzono $VENV_DIR"
    else
        echo -e "  ${GREEN}OK${NC} venv już istnieje"
    fi

    # Aktywacja
    source "$VENV_DIR/bin/activate"

    echo -e "${YELLOW}[3/4]${NC} Instaluję zależności..."
    pip install --upgrade pip -q
    pip install numpy -q
    echo -e "  ${GREEN}OK${NC} numpy"

    # PyAudio — wymaga portaudio na macOS
    echo -e "${YELLOW}[4/4]${NC} Instaluję PyAudio..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS — potrzeba portaudio
        if ! command -v brew &>/dev/null; then
            echo -e "  ${YELLOW}UWAGA${NC}: Homebrew nie znaleziony. PyAudio może wymagać:"
            echo "    brew install portaudio"
            echo "  Próbuję zainstalować PyAudio bez brew..."
        else
            if ! brew list portaudio &>/dev/null 2>&1; then
                echo "  Instaluję portaudio przez Homebrew..."
                brew install portaudio
            else
                echo -e "  ${GREEN}OK${NC} portaudio (brew)"
            fi
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &>/dev/null; then
            echo "  Instaluję portaudio (apt)..."
            sudo apt-get install -y portaudio19-dev python3-pyaudio 2>/dev/null || true
        elif command -v dnf &>/dev/null; then
            echo "  Instaluję portaudio (dnf)..."
            sudo dnf install -y portaudio-devel 2>/dev/null || true
        elif command -v pacman &>/dev/null; then
            echo "  Instaluję portaudio (pacman)..."
            sudo pacman -S --noconfirm portaudio 2>/dev/null || true
        fi
    fi

    pip install pyaudio -q 2>/dev/null && echo -e "  ${GREEN}OK${NC} pyaudio" || {
        echo -e "  ${RED}[x] Nie udało się zainstalować PyAudio.${NC}"
        echo "  Na macOS: brew install portaudio && pip install pyaudio"
        echo "  Na Linux: sudo apt-get install portaudio19-dev && pip install pyaudio"
        echo ""
        echo "  Możesz kontynuować bez PyAudio — tryb --simulate będzie działał."
    }

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Instalacja zakończona!${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Aktywacja venv:"
    echo -e "    ${CYAN}source venv/bin/activate${NC}"
    echo ""
    echo "  Uruchomienie SERWERA (komputer A):"
    echo -e "    ${CYAN}python server.py --live${NC}"
    echo ""
    echo "  Uruchomienie KLIENTA (komputer B):"
    echo -e "    ${CYAN}python client.py --interactive${NC}"
    echo -e "    ${CYAN}python client.py GET /ping${NC}"
    echo -e "    ${CYAN}python client.py POST /echo --body '{\"hello\":\"world\"}'${NC}"
    echo ""
    echo "  Tryb simulate (bez mikrofonu/głośnika):"
    echo -e "    ${CYAN}python client.py GET /ping --simulate${NC}"
    echo -e "    ${CYAN}python server.py --simulate request_get_ping.wav${NC}"
    echo ""
    echo "  Lista urządzeń audio:"
    echo -e "    ${CYAN}python server.py --list-devices${NC}"
    echo ""
}

# ── LAUNCHER ──

run_server() {
    source "$VENV_DIR/bin/activate"
    echo ""
    echo -e "${CYAN}  Uruchamiam serwer...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    python server.py --live "$@"
}

run_client() {
    source "$VENV_DIR/bin/activate"
    echo ""
    echo -e "${CYAN}  Uruchamiam klienta (interactive)...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    python client.py --interactive "$@"
}

# ── MAIN ──

case "${1:-install}" in
    install|setup)
        install
        ;;
    server|serve)
        shift
        run_server "$@"
        ;;
    client)
        shift
        run_client "$@"
        ;;
    devices|list-devices)
        source "$VENV_DIR/bin/activate"
        cd "$PROJECT_DIR"
        python server.py --list-devices
        ;;
    test)
        source "$VENV_DIR/bin/activate"
        cd "$PROJECT_DIR"
        echo ""
        echo -e "${CYAN}  Test: encode request → decode request (round-trip)${NC}"
        echo ""
        python -c "
from modem import encode_request, decode_request, encode_response, decode_response
from modem import write_wav, SAMPLE_RATE
from protocol import METHOD_GET, METHOD_POST, http_to_compact
import json

# Test 1: GET /ping
print('  Test 1: GET /ping')
audio = encode_request(METHOD_GET, '/ping')
result = decode_request(audio)
print(f'    method={result[\"method_name\"]}  path={result[\"path\"]}  crc={result[\"crc_ok\"]}')
assert result['crc_ok'], 'CRC failed!'
assert result['path'] == '/ping'
print('    OK!')

# Test 2: POST /echo z body
print('  Test 2: POST /echo + body')
body = json.dumps({'hello': 'world'}).encode()
audio = encode_request(METHOD_POST, '/echo', body)
result = decode_request(audio)
print(f'    method={result[\"method_name\"]}  path={result[\"path\"]}  body={result[\"body\"]}  crc={result[\"crc_ok\"]}')
assert result['crc_ok'], 'CRC failed!'
assert result['path'] == '/echo'
assert b'hello' in result['body']
print('    OK!')

# Test 3: Response encode/decode
print('  Test 3: Response encode/decode')
resp_audio = encode_response(http_to_compact(200), json.dumps({'pong': True}).encode())
resp = decode_response(resp_audio)
print(f'    status={resp[\"http_code\"]}  body={resp[\"body\"]}  crc={resp[\"crc_ok\"]}')
assert resp['crc_ok'], 'CRC failed!'
assert resp['http_code'] == 200
print('    OK!')

print()
print('  Wszystkie testy przeszly!')
print()
"
        ;;
    help|--help|-h)
        header
        echo "  Uzycie:"
        echo "    ./setup.sh              Zainstaluj zależności"
        echo "    ./setup.sh server       Uruchom serwer (live)"
        echo "    ./setup.sh client       Uruchom klienta (interactive)"
        echo "    ./setup.sh test         Uruchom testy round-trip"
        echo "    ./setup.sh devices      Lista urządzeń audio"
        echo ""
        ;;
    *)
        echo "  Nieznana komenda: $1"
        echo "  Użyj: ./setup.sh [install|server|client|test|devices|help]"
        ;;
esac
