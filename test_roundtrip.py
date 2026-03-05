#!/usr/bin/env python3
"""Test round-trip: encode → decode dla request i response."""

from modem import encode_request, decode_request, encode_response, decode_response
from modem import SAMPLE_RATE
from protocol import METHOD_GET, METHOD_POST, http_to_compact
import json


def test_get_ping():
    print("Test 1: GET /ping")
    audio = encode_request(METHOD_GET, "/ping")
    result = decode_request(audio)
    assert result is not None, "decode returned None"
    assert result["crc_ok"], f"CRC failed: recv=0x{result['crc_received']:04X} comp=0x{result['crc_computed']:04X}"
    assert result["path"] == "/ping", f"path mismatch: {result['path']}"
    assert result["method_name"] == "GET"
    print(f"  method={result['method_name']}  path={result['path']}  crc={result['crc_ok']}")
    print("  PASS\n")


def test_post_echo():
    print("Test 2: POST /echo + body")
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
    print("Test 3: Response encode/decode")
    body = json.dumps({"pong": True}).encode()
    resp_audio = encode_response(http_to_compact(200), body)
    resp = decode_response(resp_audio)
    assert resp is not None
    assert resp["crc_ok"], "CRC failed"
    assert resp["http_code"] == 200
    print(f"  status={resp['http_code']}  body={resp['body']}  crc={resp['crc_ok']}")
    print("  PASS\n")


def test_full_round_trip():
    print("Test 4: Full server round trip (GET /ping → 200)")
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
    print("Test 5: Full server round trip (POST /echo)")
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


if __name__ == "__main__":
    print("\n  SoundHTTP Round-Trip Tests")
    print("  " + "=" * 40 + "\n")
    test_get_ping()
    test_post_echo()
    test_response()
    test_full_round_trip()
    test_post_echo_round_trip()
    print("  ALL TESTS PASSED!\n")
