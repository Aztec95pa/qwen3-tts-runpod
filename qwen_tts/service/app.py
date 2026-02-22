# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI service for Qwen3-TTS.

Main value:
1. Upload reference audio -> create clone prompt -> immediate synthesis.
2. Save and reuse saved voice prompts.
3. Optional: auto transcribe reference audio (faster-whisper) and optional Firebase upload.

Endpoints:
- POST /api/v1/voices/immediate
- POST /api/v1/voices
- POST /api/v1/speech
- GET /health
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

LOGGER = logging.getLogger("qwen3-tts-service")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "t", "yes", "on"}


def _parse_dtype(value: str) -> torch.dtype:
    key = (value or "bfloat16").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _to_torch_dtype(value: str) -> torch.dtype:
    return _parse_dtype(value)


def _safe_str(v: Optional[str]) -> str:
    if v is None:
        return ""
    return str(v).strip()
 

@dataclass
class ServiceConfig:
    checkpoint: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    use_flash_attention: bool = True

    host: str = "0.0.0.0"
    port: int = 8787

    cors_origins: str = "*"
    profile_dir: Path = Path("./voice_profiles")
    max_profiles: int = 500

    auto_transcribe: bool = False
    asr_model: str = "base"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_language: Optional[str] = None

    firebase_enabled: bool = False
    firebase_credentials: Optional[str] = None
    firebase_bucket: Optional[str] = None
    firebase_collection: str = "qwen3_tts_jobs"

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            checkpoint=os.getenv("QWEN3_TTS_CHECKPOINT", cls.checkpoint),
            device=os.getenv("QWEN3_TTS_DEVICE", cls.device),
            dtype=os.getenv("QWEN3_TTS_DTYPE", cls.dtype),
            use_flash_attention=_env_bool("QWEN3_TTS_FLASH_ATTENTION", cls.use_flash_attention),
            host=os.getenv("HOST", cls.host),
            port=int(os.getenv("PORT", str(cls.port))),
            cors_origins=os.getenv("CORS_ORIGINS", cls.cors_origins),
            profile_dir=Path(os.getenv("VOICE_PROFILE_DIR", str(cls.profile_dir))),
            max_profiles=int(os.getenv("MAX_PROFILES", str(cls.max_profiles))),
            auto_transcribe=_env_bool("QWEN3_TTS_AUTO_TRANSCRIBE", cls.auto_transcribe),
            asr_model=os.getenv("QWEN3_TTS_ASR_MODEL", cls.asr_model),
            asr_device=os.getenv("QWEN3_TTS_ASR_DEVICE", cls.asr_device),
            asr_compute_type=os.getenv("QWEN3_TTS_ASR_COMPUTE_TYPE", cls.asr_compute_type),
            asr_language=os.getenv("QWEN3_TTS_ASR_LANGUAGE", cls.asr_language),
            firebase_enabled=_env_bool("FIREBASE_ENABLED", cls.firebase_enabled),
            firebase_credentials=os.getenv("FIREBASE_CREDENTIALS", cls.firebase_credentials),
            firebase_bucket=os.getenv("FIREBASE_BUCKET", cls.firebase_bucket),
            firebase_collection=os.getenv("FIREBASE_COLLECTION", cls.firebase_collection),
        )


def _decode_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        fp.write(data)
        tmp = fp.name
    try:
        try:
            wav, sr = sf.read(tmp, dtype="float32", always_2d=False)
        except Exception:
            wav, sr = librosa.load(tmp, sr=None, mono=True)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        return wav.astype(np.float32), int(sr)
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def _wav_to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav.astype(np.float32), int(sr), format="WAV")
    return buffer.getvalue()


def _as_list(v, n: int) -> List:
    if isinstance(v, list):
        return v
    return [v] * n


def _build_gen_kwargs(
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    subtalker_top_k: Optional[int] = None,
    subtalker_top_p: Optional[float] = None,
    subtalker_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    if subtalker_top_k is not None:
        kwargs["subtalker_top_k"] = int(subtalker_top_k)
    if subtalker_top_p is not None:
        kwargs["subtalker_top_p"] = float(subtalker_top_p)
    if subtalker_temperature is not None:
        kwargs["subtalker_temperature"] = float(subtalker_temperature)
    return kwargs


def _serialize_prompt_items(items: List[VoiceClonePromptItem], cfg: ServiceConfig) -> str:
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    voice_id = str(uuid.uuid4())
    payload = {
        "version": 1,
        "checkpoint": cfg.checkpoint,
        "items": [],
    }
    for it in items:
        payload["items"].append(
            {
                "ref_code": it.ref_code.cpu() if torch.is_tensor(it.ref_code) else None,
                "ref_spk_embedding": it.ref_spk_embedding.cpu(),
                "x_vector_only_mode": bool(it.x_vector_only_mode),
                "icl_mode": bool(it.icl_mode),
                "ref_text": it.ref_text,
            }
        )
    target = cfg.profile_dir / f"{voice_id}.pt"
    torch.save(payload, target)
    return voice_id


def _load_prompt_items(cfg: ServiceConfig, voice_id: str) -> List[VoiceClonePromptItem]:
    target = cfg.profile_dir / f"{voice_id}.pt"
    if not target.exists():
        raise HTTPException(status_code=404, detail="voice_id not found")
    payload = torch.load(target, map_location="cpu")
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list) or len(raw_items) == 0:
        raise HTTPException(status_code=422, detail="voice prompt is empty")

    out: List[VoiceClonePromptItem] = []
    for i, d in enumerate(raw_items):
        if not isinstance(d, dict):
            raise HTTPException(status_code=422, detail=f"invalid item format at index={i}")

        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)

        ref_spk_embedding = d.get("ref_spk_embedding")
        if ref_spk_embedding is None or not torch.is_tensor(ref_spk_embedding):
            raise HTTPException(
                status_code=422,
                detail=f"invalid ref_spk_embedding format at index={i}",
            )

        out.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text"),
            )
        )
    return out


def _cleanup_profiles(cfg: ServiceConfig):
    files = sorted(
        [p for p in cfg.profile_dir.glob("*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    if len(files) <= cfg.max_profiles:
        return
    for p in files[: len(files) - cfg.max_profiles]:
        try:
            p.unlink()
        except Exception:
            pass


class _FirebaseManager:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.ready = False
        self.bucket = None
        self.firestore = None

    def init(self):
        if not self.cfg.firebase_enabled:
            return
        if not self.cfg.firebase_credentials:
            LOGGER.warning("FIREBASE_ENABLED is on but FIREBASE_CREDENTIALS is not set.")
            return
        try:
            import firebase_admin
            from firebase_admin import credentials
            from firebase_admin import firestore as _firestore
            from firebase_admin import storage as _storage
        except ImportError:
            LOGGER.warning("firebase-admin is not installed. Install with: pip install firebase-admin")
            return

        try:
            if not firebase_admin._apps:
                opts = {}
                if self.cfg.firebase_bucket:
                    opts["storageBucket"] = self.cfg.firebase_bucket
                firebase_admin.initialize_app(
                    credentials.Certificate(self.cfg.firebase_credentials),
                    options=opts,
                )
            self.bucket = _storage.bucket() if self.cfg.firebase_bucket else None
            self.firestore = _firestore.client() if self.cfg.firebase_collection else None
            self.ready = True
        except Exception as e:
            LOGGER.error("failed to init firebase: %s", e, exc_info=True)
            self.ready = False

    def upload_audio(self, voice_id: str, wav_bytes: bytes) -> Optional[Dict[str, Any]]:
        if not self.ready or self.bucket is None:
            return None
        blob_name = f"qwen3-tts/{voice_id}/{uuid.uuid4()}.wav"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(wav_bytes, content_type="audio/wav")
        return {
            "bucket": self.bucket.name,
            "path": blob_name,
            "public_url": getattr(blob, "public_url", None),
            "gs_path": f"gs://{self.bucket.name}/{blob_name}",
        }

    def save_record(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self.ready or self.firestore is None:
            return None
        result = self.firestore.collection(self.cfg.firebase_collection).add(payload)
        if isinstance(result, tuple) and len(result) > 0:
            first = result[0]
            if hasattr(first, "id"):
                return str(first.id)
            if len(result) > 1 and hasattr(result[1], "id"):
                return str(result[1].id)
            return None
        if hasattr(result, "id"):
            return str(result.id)
        return None


_WHISPER_MODEL = None


def _transcribe_audio_bytes(cfg: ServiceConfig, audio_bytes: bytes) -> str:
    global _WHISPER_MODEL
    if not cfg.auto_transcribe:
        return ""
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "auto_transcribe is enabled but faster-whisper is not installed. "
                    "pip install faster-whisper first."
                ),
            )
        _WHISPER_MODEL = WhisperModel(
            model_size_or_path=cfg.asr_model,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
        )

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as fp:
        fp.write(audio_bytes)
        tmp = fp.name
    try:
        segments, info = _WHISPER_MODEL.transcribe(
            tmp,
            beam_size=1,
            language=cfg.asr_language,
            vad_filter=True,
        )
        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
        return "".join(parts).strip()
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


class SpeechRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "Auto"
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None
    save_to_firebase: bool = False


def create_app(cfg: ServiceConfig) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Service API", version="0.1.0")
    origins = _safe_str(cfg.cors_origins) or "*"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if origins == "*" else [x.strip() for x in origins.split(",") if x.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = cfg
    app.state.tts_model = None
    app.state.firebase = None

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

        @app.get("/app", include_in_schema=False)
        async def app_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    @app.on_event("startup")
    async def startup_event():
        logger = logging.getLogger("qwen3-tts-service")
        logger.info("Loading Qwen3-TTS model: %s", cfg.checkpoint)
        try:
            dtype = _to_torch_dtype(cfg.dtype)
            attn = "flash_attention_2" if cfg.use_flash_attention else None
            tts = Qwen3TTSModel.from_pretrained(
                cfg.checkpoint,
                device_map=cfg.device,
                dtype=dtype,
                attn_implementation=attn,
            )
            app.state.tts_model = tts
            logger.info("Qwen3-TTS loaded.")
        except Exception as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            raise

        firebase = _FirebaseManager(cfg)
        firebase.init()
        app.state.firebase = firebase
        cfg.profile_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_profiles(cfg)

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if app.state.tts_model is not None else "booting",
            "checkpoint": cfg.checkpoint,
            "voice_profiles": len(list(cfg.profile_dir.glob("*.pt"))),
        }

    @app.get("/")
    async def index():
        return {"name": "Qwen3-TTS Service API", "docs": "/docs"}

    @app.post("/api/v1/voices/immediate")
    async def create_and_speak(
        ref_audio: UploadFile = File(...),
        text: str = Form(...),
        ref_text: str = Form(""),
        language: str = Form("Auto"),
        x_vector_only: bool = Form(False),
        save_profile: bool = Form(False),
        temperature: Optional[float] = Form(None),
        top_k: Optional[int] = Form(None),
        top_p: Optional[float] = Form(None),
        repetition_penalty: Optional[float] = Form(None),
        max_new_tokens: Optional[int] = Form(None),
        subtalker_top_k: Optional[int] = Form(None),
        subtalker_top_p: Optional[float] = Form(None),
        subtalker_temperature: Optional[float] = Form(None),
        save_to_firebase: bool = Form(False),
    ):
        if app.state.tts_model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="text is required.")

        raw = await ref_audio.read()
        if not raw:
            raise HTTPException(status_code=400, detail="ref_audio is required.")

        wav, sr = _decode_audio_bytes(raw)
        used_ref_text = _safe_str(ref_text)

        if not used_ref_text and not x_vector_only:
            if cfg.auto_transcribe:
                used_ref_text = _transcribe_audio_bytes(cfg, raw)
            if not used_ref_text:
                raise HTTPException(
                    status_code=400,
                    detail="ref_text is required when x_vector_only is false (or enable auto transcribe).",
                )

        prompt_items = app.state.tts_model.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=used_ref_text or None,
            x_vector_only_mode=bool(x_vector_only),
        )
        gen_kwargs = _build_gen_kwargs(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        wavs, sample_rate = app.state.tts_model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            **gen_kwargs,
        )
        wav_bytes = _wav_to_wav_bytes(wavs[0], sample_rate)
        voice_id = None
        if save_profile:
            voice_id = _serialize_prompt_items(prompt_items, cfg)
            _cleanup_profiles(cfg)

        firebase_payload = None
        if save_to_firebase and app.state.firebase is not None and app.state.firebase.ready:
            bucket_file = app.state.firebase.upload_audio(voice_id or str(uuid.uuid4()), wav_bytes)
            firebase_id = app.state.firebase.save_record(
                {
                    "mode": "immediate",
                    "language": language,
                    "text": text,
                    "x_vector_only": bool(x_vector_only),
                    "voice_id": voice_id,
                    "file": bucket_file,
                }
            )
            firebase_payload = {"storage": bucket_file, "record_id": firebase_id}

        return {
            "voice_id": voice_id,
            "ref_text": used_ref_text,
            "sample_rate": sample_rate,
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "firebase": firebase_payload,
        }

    @app.post("/api/v1/voices")
    async def create_voice(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(""),
        x_vector_only: bool = Form(False),
        save_to_firebase: bool = Form(False),
        language: str = Form("Auto"),
    ):
        if app.state.tts_model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        raw = await ref_audio.read()
        wav, sr = _decode_audio_bytes(raw)

        if not _safe_str(ref_text) and not x_vector_only:
            if cfg.auto_transcribe:
                ref_text = _transcribe_audio_bytes(cfg, raw)
            if not _safe_str(ref_text):
                raise HTTPException(
                    status_code=400,
                    detail="ref_text is required when x_vector_only is false (or enable auto transcribe).",
                )

        items = app.state.tts_model.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=_safe_str(ref_text) or None,
            x_vector_only_mode=bool(x_vector_only),
        )
        voice_id = _serialize_prompt_items(items, cfg)
        _cleanup_profiles(cfg)

        firebase_payload = None
        if save_to_firebase and app.state.firebase is not None and app.state.firebase.ready:
            firebase_payload = {
                "record_id": app.state.firebase.save_record(
                    {
                        "mode": "voice_profile",
                        "voice_id": voice_id,
                        "language": language,
                        "ref_text": ref_text,
                        "x_vector_only": bool(x_vector_only),
                    }
                )
            }

        return {"voice_id": voice_id, "firebase": firebase_payload}

    @app.post("/api/v1/speech")
    async def synthesize(req: SpeechRequest):
        if app.state.tts_model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text is required.")
        if not req.voice_id:
            raise HTTPException(status_code=400, detail="voice_id is required.")

        items = _load_prompt_items(cfg, req.voice_id)
        gen_kwargs = _build_gen_kwargs(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
            subtalker_top_k=req.subtalker_top_k,
            subtalker_top_p=req.subtalker_top_p,
            subtalker_temperature=req.subtalker_temperature,
        )

        wavs, sample_rate = app.state.tts_model.generate_voice_clone(
            text=req.text,
            language=req.language,
            voice_clone_prompt=items,
            **gen_kwargs,
        )
        wav_bytes = _wav_to_wav_bytes(wavs[0], sample_rate)

        firebase_payload = None
        if req.save_to_firebase and app.state.firebase is not None and app.state.firebase.ready:
            bucket_file = app.state.firebase.upload_audio(req.voice_id, wav_bytes)
            firebase_payload = {
                "storage": bucket_file,
                "record_id": app.state.firebase.save_record(
                    {
                        "mode": "synthesize",
                        "voice_id": req.voice_id,
                        "language": req.language,
                        "text": req.text,
                    }
                ),
            }

        return {
            "voice_id": req.voice_id,
            "sample_rate": sample_rate,
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "firebase": firebase_payload,
        }

    @app.get("/api/v1/voices/{voice_id}")
    async def get_voice_profile(voice_id: str):
        target = cfg.profile_dir / f"{voice_id}.pt"
        if not target.exists():
            raise HTTPException(status_code=404, detail="voice_id not found")
        return {
            "voice_id": voice_id,
            "file": str(target),
            "size_bytes": target.stat().st_size,
            "modified_time": target.stat().st_mtime,
        }

    @app.delete("/api/v1/voices/{voice_id}")
    async def delete_voice_profile(voice_id: str):
        target = cfg.profile_dir / f"{voice_id}.pt"
        if not target.exists():
            raise HTTPException(status_code=404, detail="voice_id not found")
        target.unlink()
        return {"deleted": voice_id}

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Qwen3-TTS service API.")
    p.add_argument("--checkpoint", default=os.getenv("QWEN3_TTS_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    p.add_argument("--device", default=os.getenv("QWEN3_TTS_DEVICE", "cuda:0"))
    p.add_argument("--dtype", default=os.getenv("QWEN3_TTS_DTYPE", "bfloat16"))
    p.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention 2.")
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8787")))
    p.add_argument("--profile-dir", default=os.getenv("VOICE_PROFILE_DIR", "./voice_profiles"))
    p.add_argument("--max-profiles", type=int, default=int(os.getenv("MAX_PROFILES", "500")))
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    cfg = ServiceConfig.from_env()
    cfg.checkpoint = args.checkpoint
    cfg.device = args.device
    cfg.dtype = args.dtype
    cfg.use_flash_attention = not args.no_flash_attn
    cfg.host = args.host
    cfg.port = args.port
    cfg.profile_dir = Path(args.profile_dir)
    cfg.max_profiles = args.max_profiles
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
