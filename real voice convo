import asyncio
import io
import json
import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import edge_tts
from groq import AsyncGroq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("voice_ai")

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD = 0.008
ZCR_THRESHOLD = 0.15
SILENCE_TRIGGER_S = 1.2
MIN_SPEECH_BYTES = 8000
MAX_AUDIO_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"
PORT = int(os.getenv("PORT", 8000))


class VAD:
    def __init__(self, rate: int = SAMPLE_RATE):
        self.frame_bytes = int(rate * FRAME_DURATION_MS / 1000) * BYTES_PER_SAMPLE

    def is_speech(self, pcm: bytes) -> bool:
        if len(pcm) < self.frame_bytes:
            return False
        s = np.frombuffer(pcm[:self.frame_bytes], dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(s ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(s)))) / 2)
        return rms > SILENCE_THRESHOLD and zcr > ZCR_THRESHOLD


SAMPLE_KNOWLEDGE = """
Voice AI System
================
This is a real-time voice assistant using Groq Whisper for speech-to-text,
LLaMA 3 for language understanding, and edge-tts for speech output.
It runs entirely free with no OpenAI billing required.
""".strip()


class RAGSystem:
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self.kdir = Path(knowledge_dir)
        self._chunks = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._ready = False
        self._load()

    def _load(self):
        chunks = _smart_chunk(SAMPLE_KNOWLEDGE)
        if self.kdir.exists():
            for p in sorted(self.kdir.glob("**/*.txt")):
                try:
                    chunks.extend(_smart_chunk(p.read_text(encoding="utf-8").strip()))
                except Exception:
                    pass
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = model.encode(chunks, convert_to_numpy=True).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            self._embeddings = embs / (norms + 1e-8)
            self._chunks = chunks
            self._model = model
            self._ready = True
        except ImportError:
            pass

    def is_ready(self):
        return self._ready

    async def get_context(self, query: str):
        if not self._ready or not query.strip():
            return ""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve, query)

    def _retrieve(self, query: str):
        q = self._model.encode([query], convert_to_numpy=True).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        scores = (self._embeddings @ q.T).flatten()
        idx = np.argsort(scores)[::-1][:3]
        hits = [self._chunks[i] for i in idx if scores[i] >= 0.30]
        return "\n\n".join(hits) if hits else ""


def _smart_chunk(text: str, max_words: int = 120, overlap: int = 20):
    chunks = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start += max_words - overlap
    return chunks


_SENT_END = re.compile(r'[.!?]["\')\]]*\s*$')


def _is_sentence_end(text: str):
    return bool(_SENT_END.search(text))


class VoicePipeline:
    def __init__(self, session_id: str, websocket: WebSocket, rag: RAGSystem):
        self.sid = session_id
        self.ws = websocket
        self.rag = rag
        self.client = AsyncGroq()
        self.vad = VAD()
        self.audio_buf = bytearray()
        self.is_recording = False
        self.last_speech_t = 0.0
        self.is_ai_speaking = False
        self._interrupted = asyncio.Event()
        self._ai_task: Optional[asyncio.Task] = None
        self.history = []

    async def run(self):
        async for raw in self.ws.iter_bytes():
            if raw[:1] == b"{":
                await self._handle_ctrl(json.loads(raw.decode()))
            else:
                await self._handle_audio(raw)

    async def _handle_ctrl(self, msg):
        if msg.get("type") == "interrupt":
            await self._interrupt()

    async def _handle_audio(self, chunk: bytes):
        voiced = self.vad.is_speech(chunk)
        if voiced:
            self.last_speech_t = time.monotonic()
            if not self.is_recording:
                self.is_recording = True
                await self._json({"type": "status", "status": "listening"})
                if self.is_ai_speaking:
                    await self._interrupt()
            self.audio_buf.extend(chunk)
        else:
            if self.is_recording and time.monotonic() - self.last_speech_t >= 1.2:
                snap = bytes(self.audio_buf)
                self.audio_buf.clear()
                self.is_recording = False
                if len(snap) < 8000:
                    return
                asyncio.create_task(self._process(snap))

    async def _process(self, audio: bytes):
        await self._json({"type": "status", "status": "transcribing"})
        transcript = await self._stt(audio)
        if not transcript:
            return
        await self._json({"type": "transcript", "text": transcript})
        await self._json({"type": "status", "status": "thinking"})
        context = await self.rag.get_context(transcript)
        self._interrupted.clear()
        self.is_ai_speaking = True
        self._ai_task = asyncio.create_task(self._respond(transcript, context))
        await self._ai_task

    async def _stt(self, pcm: bytes):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        buf.seek(0)
        buf.name = "audio.wav"
        resp = await self.client.audio.transcriptions.create(
            model=STT_MODEL,
            file=buf,
            language="en",
        )
        return resp.text.strip()

    async def _respond(self, user_text, context):
        system = "You are a helpful voice assistant."
        messages = [{"role": "system", "content": system}] + [{"role": "user", "content": user_text}]

        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
        )

        reply = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            reply += token

        await self._tts(reply)
        await self._json({"type": "response_text", "text": reply})
        await self._json({"type": "audio_end"})
        self.is_ai_speaking = False

    async def _tts(self, text):
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        audio = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        if audio:
            await self.ws.send_bytes(audio)

    async def _interrupt(self):
        self._interrupted.set()
        if self._ai_task and not self._ai_task.done():
            self._ai_task.cancel()
        self.is_ai_speaking = False
        await self._json({"type": "interrupted"})

    async def cleanup(self):
        await self._interrupt()

    async def _json(self, data):
        await self.ws.send_text(json.dumps(data))


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAGSystem()


@app.get("/")
async def root():
    return HTMLResponse("<h1>Voice AI Running</h1>")


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    pipeline = VoicePipeline(session_id, websocket, rag)
    try:
        await pipeline.run()
    except WebSocketDisconnect:
        pass
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
