"""Raw/JPEG全オフロード実験用の安全な長さプレフィックスTCPプロトコル。

メタデータはJSON、画像本体はバイト列として送る。信頼できないネットワークで
pickleを受信しないため、既存の分割推論プロトコルとは意図的に分離している。
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


# header JSONの長さ(uint32) + payloadの長さ(uint64)
_LENGTHS = struct.Struct("!IQ")
MAX_HEADER_BYTES = 1 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def recvall(sock: socket.socket, size: int) -> bytes:
    """size byteを受信し切る。接続切断時はConnectionErrorを送出する。"""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("メッセージ受信中に接続が閉じました。")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> int:
    """JSON headerと任意のbinary payloadを送信し、wire byte数を返す。"""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("headerが上限を超えました。")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("payloadが上限を超えました。")
    sock.sendall(_LENGTHS.pack(len(header_bytes), len(payload)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)
    return _LENGTHS.size + len(header_bytes) + len(payload)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes, int]:
    """1メッセージを受信し、header、payload、wire byte数を返す。"""
    header_size, payload_size = _LENGTHS.unpack(recvall(sock, _LENGTHS.size))
    if header_size > MAX_HEADER_BYTES or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError("受信メッセージが許容サイズを超えました。")
    try:
        header = json.loads(recvall(sock, header_size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("header JSONを解釈できません。") from exc
    payload = recvall(sock, payload_size) if payload_size else b""
    return header, payload, _LENGTHS.size + header_size + payload_size
