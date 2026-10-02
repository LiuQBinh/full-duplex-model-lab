"""Ask a running Moshi server a question from a wav file.

The server is the web UI from ``python -m moshi_mlx.local_web``
(websocket ``ws://127.0.0.1:8998/api/chat``). The file is resampled to
24 kHz mono, then silence is appended so Moshi has frames left to answer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import aiohttp
import numpy as np
import sphn

SAMPLE_RATE = 24000
FRAME_SIZE = 1920


def load_question(path: str, silence_s: float) -> np.ndarray:
    pcm = np.asarray(sphn.read(path, sample_rate=SAMPLE_RATE)[0], dtype=np.float32)
    if pcm.ndim == 2:
        pcm = pcm[0]
    silence = np.zeros(int(silence_s * SAMPLE_RATE), dtype=np.float32)
    pcm = np.concatenate([pcm, silence])
    pad = (-len(pcm)) % FRAME_SIZE
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.float32)])
    return pcm


def opus_packets(pcm: np.ndarray) -> list[bytes]:
    writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    packets = []
    for offset in range(0, len(pcm), FRAME_SIZE):
        frame = np.ascontiguousarray(pcm[offset : offset + FRAME_SIZE])
        packet = writer.append_pcm(frame)
        if packet:
            packets.append(packet)
    return packets


async def ask(url: str, pcm: np.ndarray, reply_path: str, timeout_s: float) -> str:
    packets = opus_packets(pcm)
    reader = sphn.OpusStreamReader(SAMPLE_RATE)
    reply_chunks: list[np.ndarray] = []
    text_parts: list[str] = []
    audio_frames = 0
    started = time.monotonic()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            handshake = await ws.receive(timeout=30)
            if handshake.type != aiohttp.WSMsgType.BINARY or not handshake.data or handshake.data[0] != 0:
                raise SystemExit(f"expected handshake from server, got {handshake.type} {handshake.data!r}")
            print("connected", flush=True)

            async def send() -> None:
                for index, packet in enumerate(packets, start=1):
                    await ws.send_bytes(b"\x01" + packet)
                    if index == 1 or index % 25 == 0 or index == len(packets):
                        print(f"sent {index}/{len(packets)} frames", flush=True)
                    await asyncio.sleep(0)
                print("question sent", flush=True)

            async def receive() -> None:
                nonlocal audio_frames
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.BINARY or not msg.data:
                        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    kind = msg.data[0]
                    payload = msg.data[1:]
                    if kind == 2 and payload:
                        piece = payload.decode("utf-8", errors="replace")
                        text_parts.append(piece)
                        print(piece, end="", flush=True)
                    elif kind == 1 and payload:
                        decoded = reader.append_bytes(payload)
                        if decoded is not None and len(decoded):
                            audio = np.asarray(decoded, dtype=np.float32).reshape(-1)
                            reply_chunks.append(audio)
                            audio_frames += max(1, len(audio) // FRAME_SIZE)
                            if audio_frames == 1 or audio_frames % 10 == 0:
                                print(
                                    f"\nreply_frames={audio_frames} "
                                    f"t={time.monotonic() - started:.1f}s",
                                    flush=True,
                                )
                    if reply_chunks and sum(len(c) for c in reply_chunks) >= len(pcm) - FRAME_SIZE:
                        break

            send_task = asyncio.create_task(send())
            try:
                await asyncio.wait_for(receive(), timeout=timeout_s)
            except asyncio.TimeoutError:
                print("\ntimed out waiting for the reply", flush=True)
            send_task.cancel()
            await ws.close()

    elapsed = time.monotonic() - started
    reply = np.concatenate(reply_chunks) if reply_chunks else np.zeros(0, dtype=np.float32)
    if reply_path and reply.size:
        sphn.write_wav(reply_path, reply, SAMPLE_RATE)
    print(
        f"\nframes_in={len(packets)} reply_samples={reply.size} "
        f"seconds={elapsed:.1f} text={''.join(text_parts)!r}",
        flush=True,
    )
    return "".join(text_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Question audio. Resampled to 24 kHz mono.")
    parser.add_argument("--url", default="ws://127.0.0.1:8998/api/chat")
    parser.add_argument("--silence", type=float, default=4.0, help="Seconds of silence after the question.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--reply", default="", help="Where to write Moshi's wav reply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pcm = load_question(args.wav, args.silence)
    print(f"question_samples={pcm.size} duration_s={pcm.size / SAMPLE_RATE:.2f}", flush=True)
    asyncio.run(ask(args.url, pcm, args.reply, args.timeout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
