"""
SoundHTTP — protokół HTTP over Sound

Ramka żądania (Request):
  [PREAMBUŁA] [TYP:1nibble] [METHOD:1nibble] [PATH_LEN:2nibble] [PATH...] [BODY_LEN:4nibble] [BODY...] [CRC16]

  TYP:     0x1 = REQUEST, 0x2 = RESPONSE
  METHOD:  0x0 = GET, 0x1 = POST, 0x2 = PUT, 0x3 = DELETE
  PATH_LEN: długość ścieżki w bajtach (1 bajt = max 255)
  BODY_LEN: długość body w bajtach (2 bajty = max 65535)

Ramka odpowiedzi (Response):
  [PREAMBUŁA] [TYP:1nibble] [STATUS:2nibble] [BODY_LEN:4nibble] [BODY...] [CRC16]

  STATUS: np. 0xC8 = 200, 0x90 = 144 → mapowane na HTTP status codes
          Używamy własnego mapowania 1-bajtowego:
            0x00 = 200 OK
            0x01 = 201 Created
            0x04 = 400 Bad Request
            0x05 = 404 Not Found
            0x06 = 405 Method Not Allowed
            0x10 = 500 Internal Server Error
"""

# Typy ramek
FRAME_REQUEST = 0x1
FRAME_RESPONSE = 0x2

# Metody HTTP
METHOD_GET = 0x0
METHOD_POST = 0x1
METHOD_PUT = 0x2
METHOD_DELETE = 0x3

METHOD_NAMES = {
    METHOD_GET: "GET",
    METHOD_POST: "POST",
    METHOD_PUT: "PUT",
    METHOD_DELETE: "DELETE",
}

METHOD_FROM_NAME = {v: k for k, v in METHOD_NAMES.items()}

# Status codes — kompaktowe mapowanie (1 bajt)
STATUS_MAP = {
    0x00: (200, "OK"),
    0x01: (201, "Created"),
    0x04: (400, "Bad Request"),
    0x05: (404, "Not Found"),
    0x06: (405, "Method Not Allowed"),
    0x10: (500, "Internal Server Error"),
}

STATUS_FROM_HTTP = {v[0]: k for k, v in STATUS_MAP.items()}

# Odwrócone: HTTP code → compact code
def http_to_compact(http_code: int) -> int:
    return STATUS_FROM_HTTP.get(http_code, 0x10)

def compact_to_http(compact: int) -> tuple[int, str]:
    return STATUS_MAP.get(compact, (500, "Internal Server Error"))