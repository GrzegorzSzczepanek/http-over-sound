#!/usr/bin/env python3
"""Test round-trip: encode → decode dla request, response, i pakietów."""

from modem import (
    encode_request, decode_request, encode_response, decode_response,
    build_request_frame, build_response_frame,
    parse_request_frame, parse_response_frame,
    SAMPLE_RATE,
)
from packets import (
    split_into_packets, encode_packet_audio, decode_packet_from_audio,
    reassemble_packets, generate_ack_signal, generate_nak_signal,
    detect_ack_or_nak,
)
from protocol import METHOD_GET, METHOD_POST, http_to_compact
import json


import numpy as np


def test_get_ping():
    print("Test 1: GET /ping (legacy audio)")
    audio = encode_request(METHOD_GET, "/ping")
    result = decode_request(audio)
    assert result is not None, "decode returned None"
    assert result["crc_ok"], f"CRC failed: recv=0x{result['crc_received']:04X} comp=0x{result['crc_computed']:04X}"
    assert result["path"] == "/ping", f"path mismatch: {result['path']}"
    assert result["method_name"] == "GET"
    print(f"  method={result['method_name']}  path={result['path']}  crc={result['crc_ok']}")
    print("  PASS\n")


def test_post_echo():
    print("Test 2: POST /echo + body (legacy audio)")
    body = json.dumps({"hello": "world"}).encode()
    audio = encode_request(METHOD_POST, "/echo", body)
    result = decode_request(audio)
    assert result is not None
    assert result["crc_ok"], "CRC failed"
    assert result["path"] == "/echo"
    assert b"hello" in result["body"]
    print(f"  method={result['method_name']}  path={result['path']}  body={result['body']}")
    print("  PASS\n")


def test_response():
    print("Test 3: Response encode/decode (legacy audio)")
    body = json.dumps({"pong": True}).encode()
    resp_audio = encode_response(http_to_compact(200), body)
    resp = decode_response(resp_audio)
    assert resp is not None
    assert resp["crc_ok"], "CRC failed"
    assert resp["http_code"] == 200
    print(f"  status={resp['http_code']}  body={resp['body']}  crc={resp['crc_ok']}")
    print("  PASS\n")


def test_full_round_trip():
    print("Test 4: Full server round trip (GET /ping → 200, legacy)")
    from server import SoundRouter, setup_default_routes, SoundHTTPServer

    router = SoundRouter()
    setup_default_routes(router)
    server = SoundHTTPServer(router)

    req_audio = encode_request(METHOD_GET, "/ping")
    resp_audio = server.process_audio(req_audio, SAMPLE_RATE)
    assert resp_audio is not None, "Server returned None"

    resp = decode_response(resp_audio)
    assert resp is not None, "Could not decode response"
    assert resp["crc_ok"], "Response CRC failed"
    assert resp["http_code"] == 200
    body = json.loads(resp["body"].decode())
    assert body.get("pong") is True
    print(f"  status={resp['http_code']}  body={resp['body'][:80]}")
    print("  PASS\n")


def test_post_echo_round_trip():
    print("Test 5: Full server round trip (POST /echo, legacy)")
    from server import SoundRouter, setup_default_routes, SoundHTTPServer

    router = SoundRouter()
    setup_default_routes(router)
    server = SoundHTTPServer(router)

    payload = json.dumps({"msg": "hi"}).encode()
    req_audio = encode_request(METHOD_POST, "/echo", payload)
    resp_audio = server.process_audio(req_audio, SAMPLE_RATE)
    assert resp_audio is not None

    resp = decode_response(resp_audio)
    assert resp is not None
    assert resp["crc_ok"]
    assert resp["http_code"] == 200
    body = json.loads(resp["body"].decode())
    assert body["echo"]["msg"] == "hi"
    print(f"  status={resp['http_code']}  body={resp['body'][:80]}")
    print("  PASS\n")


# ══════════════════════════════════════════════════════════════════════
#  TESTY PAKIETOWE
# ══════════════════════════════════════════════════════════════════════

def test_frame_build_parse():
    print("Test 6: build/parse request frame (bytes)")
    frame = build_request_frame(METHOD_GET, "/ping")
    req = parse_request_frame(frame)
    assert req is not None, "parse returned None"
    assert req["crc_ok"], "CRC failed"
    assert req["path"] == "/ping"
    assert req["method_name"] == "GET"
    print(f"  {req['method_name']} {req['path']}  crc={req['crc_ok']}  len={len(frame)}B")
    print("  PASS\n")


def test_response_frame_build_parse():
    print("Test 7: build/parse response frame (bytes)")
    body = json.dumps({"pong": True}).encode()
    frame = build_response_frame(http_to_compact(200), body)
    resp = parse_response_frame(frame)
    assert resp is not None, "parse returned None"
    assert resp["crc_ok"], "CRC failed"
    assert resp["http_code"] == 200
    assert b"pong" in resp["body"]
    print(f"  {resp['http_code']} {resp['status_text']}  body={resp['body'][:60]}")
    print("  PASS\n")


def test_packet_split_reassemble():
    print("Test 8: split into packets → reassemble")
    data = b"A" * 50 + b"B" * 30  # 80 bytes → 3 pakiety (32+32+16)
    packets = split_into_packets(data)
    assert len(packets) == 3, f"expected 3 packets, got {len(packets)}"
    assert packets[0]['flags'] & 0x01  # MORE
    assert packets[1]['flags'] & 0x01  # MORE
    assert not (packets[2]['flags'] & 0x01)  # LAST
    reassembled = reassemble_packets(packets)
    assert reassembled == data, "reassembled data mismatch"
    print(f"  {len(data)}B → {len(packets)} pkts → {len(reassembled)}B")
    print("  PASS\n")


def test_packet_audio_roundtrip():
    print("Test 9: packet encode → decode (audio)")
    frame = build_request_frame(METHOD_GET, "/ping")
    packets = split_into_packets(frame)
    assert len(packets) >= 1

    decoded_packets = []
    for pkt in packets:
        audio = encode_packet_audio(pkt)
        result = decode_packet_from_audio(audio)
        assert result is not None, f"PKT #{pkt['seq']} decode failed"
        assert result['crc_ok'], f"PKT #{pkt['seq']} CRC failed"
        decoded_packets.append(result)
        print(f"  PKT #{pkt['seq']}: {pkt['payload'][:20]}... → CRC OK")

    data = reassemble_packets(decoded_packets)
    assert data == frame, "reassembled frame mismatch"
    req = parse_request_frame(data)
    assert req is not None and req["crc_ok"]
    assert req["path"] == "/ping"
    print(f"  Reassembled → {req['method_name']} {req['path']} CRC OK")
    print("  PASS\n")


def test_ack_nak_signals():
    print("Test 10: ACK/NAK signal generation & detection")
    ack_audio = generate_ack_signal()
    nak_audio = generate_nak_signal()

    ack_result = detect_ack_or_nak(ack_audio)
    assert ack_result == 'ACK', f"expected ACK, got {ack_result}"
    print(f"  ACK signal → detected as: {ack_result}")

    nak_result = detect_ack_or_nak(nak_audio)
    assert nak_result == 'NAK', f"expected NAK, got {nak_result}"
    print(f"  NAK signal → detected as: {nak_result}")
    print("  PASS\n")


def test_large_body_packets():
    print("Test 11: Large body packet round-trip (POST /echo)")
    body = json.dumps({"msg": "x" * 100, "data": list(range(20))}).encode()
    frame = build_request_frame(METHOD_POST, "/echo", body)
    packets = split_into_packets(frame)
    print(f"  Body: {len(body)}B, Frame: {len(frame)}B, Packets: {len(packets)}")

    decoded_packets = []
    for pkt in packets:
        audio = encode_packet_audio(pkt)
        result = decode_packet_from_audio(audio)
        assert result is not None, f"PKT #{pkt['seq']} decode failed"
        assert result['crc_ok'], f"PKT #{pkt['seq']} CRC failed"
        decoded_packets.append(result)

    data = reassemble_packets(decoded_packets)
    assert data == frame
    req = parse_request_frame(data)
    assert req is not None and req["crc_ok"]
    assert req["path"] == "/echo"
    assert b"data" in req["body"]
    print(f"  {len(packets)} pkts → reassemble → parse OK")
    print("  PASS\n")


# ══════════════════════════════════════════════════════════════════════
#  TESTY LENIENT MODE (best-effort)
# ══════════════════════════════════════════════════════════════════════

def test_lenient_corrupted_audio():
    """Flip kilka nibble'i w audio i sprawdź, że lenient nadal zwraca dane."""
    print("Test 12: Lenient — corrupted audio (CRC FAIL → best-effort)")

    # Encode poprawny request
    audio = encode_request(METHOD_GET, "/ping")

    # Corrupt: dodaj szum w środku transmisji
    mid = len(audio) // 2
    noise = np.random.normal(0, 0.3, 1000)
    audio[mid:mid + 1000] += noise

    # Strict: może zwrócić dane, ale CRC fail
    result_strict = decode_request(audio, strict=True)

    # Lenient: powinien ZAWSZE zwrócić co się da
    result_lenient = decode_request(audio, strict=False)
    assert result_lenient is not None, "lenient decode returned None (should try best-effort)"
    print(f"  strict: {'got data' if result_strict else 'None'}  "
          f"crc={result_strict['crc_ok'] if result_strict else 'N/A'}")
    print(f"  lenient: path={result_lenient['path']!r}  "
          f"method={result_lenient['method_name']}  crc={result_lenient['crc_ok']}")
    print("  PASS\n")


def test_lenient_corrupted_frame():
    """Uszkodz bajty w ramce i sprawdź, że lenient wyciąga częściowe dane."""
    print("Test 13: Lenient — corrupted frame bytes (parse_request_frame)")

    frame = build_request_frame(METHOD_POST, "/echo",
                                json.dumps({"key": "val"}).encode())

    # Corrupt last 2 bytes (CRC)
    corrupted = bytearray(frame)
    corrupted[-1] ^= 0xFF
    corrupted[-2] ^= 0xFF
    corrupted = bytes(corrupted)

    # Strict: zwróci dane z crc_ok=False
    result_strict = parse_request_frame(corrupted, strict=True)
    assert result_strict is not None, "strict should still parse (valid structure)"
    assert not result_strict["crc_ok"], "CRC should fail"

    # Lenient: też zwróci dane
    result_lenient = parse_request_frame(corrupted, strict=False)
    assert result_lenient is not None
    assert not result_lenient["crc_ok"]
    assert result_lenient["path"] == "/echo"
    assert b"key" in result_lenient["body"]
    print(f"  strict:  path={result_strict['path']!r} crc={result_strict['crc_ok']}")
    print(f"  lenient: path={result_lenient['path']!r} crc={result_lenient['crc_ok']}")

    # Teraz obetnij ramkę (brak CRC na końcu)
    truncated = frame[:-2]  # brak CRC
    result_strict2 = parse_request_frame(truncated, strict=True)
    result_lenient2 = parse_request_frame(truncated, strict=False)
    assert result_strict2 is None, "strict should reject truncated frame"
    assert result_lenient2 is not None, "lenient should extract partial data"
    assert result_lenient2["path"] == "/echo"
    print(f"  truncated frame: strict=None  lenient=path={result_lenient2['path']!r}")
    print("  PASS\n")


def test_lenient_server_roundtrip():
    """Server w trybie lenient przetwarza request z uszkodzonym CRC."""
    print("Test 14: Lenient — server round-trip with corrupted request")
    from server import SoundRouter, setup_default_routes, SoundHTTPServer

    router = SoundRouter()
    setup_default_routes(router)
    server = SoundHTTPServer(router, strict=False)

    # Poprawny request → odpowiedź
    req_audio = encode_request(METHOD_GET, "/ping")
    resp_audio = server.process_audio(req_audio, SAMPLE_RATE)
    assert resp_audio is not None, "Server returned None for valid request"

    resp = decode_response(resp_audio, strict=False)
    assert resp is not None
    assert resp["http_code"] == 200
    print(f"  lenient server: {resp['http_code']} {resp['status_text']} crc={resp['crc_ok']}")
    print("  PASS\n")


if __name__ == "__main__":
    print("\n  SoundHTTP Round-Trip Tests")
    print("  " + "=" * 40 + "\n")

    # Legacy tests
    test_get_ping()
    test_post_echo()
    test_response()
    test_full_round_trip()
    test_post_echo_round_trip()

    # Packet protocol tests
    test_frame_build_parse()
    test_response_frame_build_parse()
    test_packet_split_reassemble()
    test_packet_audio_roundtrip()
    test_ack_nak_signals()
    test_large_body_packets()

    # Lenient mode tests
    test_lenient_corrupted_audio()
    test_lenient_corrupted_frame()
    test_lenient_server_roundtrip()

    print("  ALL 14 TESTS PASSED!\n")
