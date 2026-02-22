"""
RunPod Serverless handler for Qwen3-TTS voice cloning.

Accepts JSON input with:
  - ref_audio_base64: base64-encoded reference audio (3+ seconds)
  - text: text to synthesize
  - ref_text: (optional) transcript of reference audio
  - language: (optional, default "Auto")
  - voice_id: (optional) reuse a previously saved voice profile
  - save_profile: (optional, default false) save the voice for reuse
  - x_vector_only: (optional, default false) use x-vector only mode

Returns JSON with:
  - audio_base64: base64-encoded WAV of the synthesized speech
  - sample_rate: sample rate of the audio
  - voice_id: (if save_profile=true) ID for reuse
  - ref_text: the reference text used (may be auto-transcribed)
"""

import base64
import io
import os
import tempfile
import uuid
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import runpod

LOGGER = logging.getLogger("qwen3-tts-serverless")
logging.basicConfig(level=logging.INFO)

# ── Global model (loaded once, reused across requests) ──────────────────

_MODEL = None
_WHISPER = None
_PROFILE_DIR = Path(os.getenv("VOICE_PROFILE_DIR", "/app/voice_profiles"))
_PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _get_model():
    """Lazy-load the Qwen3-TTS model (first request only)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from qwen_tts import Qwen3TTSModel

    checkpoint = os.getenv("QWEN3_TTS_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = os.getenv("QWEN3_TTS_DEVICE", "cuda:0")
    dtype_str = os.getenv("QWEN3_TTS_DTYPE", "bfloat16")
    use_flash = os.getenv("QWEN3_TTS_FLASH_ATTENTION", "true").lower() in ("1", "true", "yes")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16)
    attn = "flash_attention_2" if use_flash else None

    LOGGER.info("Loading Qwen3-TTS model: %s (device=%s, dtype=%s)", checkpoint, device, dtype)
    _MODEL = Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn,
    )
    LOGGER.info("Model loaded successfully.")
    return _MODEL


def _transcribe(audio_bytes: bytes) -> str:
    """Auto-transcribe reference audio using faster-whisper."""
    global _WHISPER
    asr_enabled = os.getenv("QWEN3_TTS_AUTO_TRANSCRIBE", "true").lower() in ("1", "true", "yes")
    if not asr_enabled:
        return ""

    if _WHISPER is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            LOGGER.warning("faster-whisper not installed, skipping transcription")
            return ""
        model_name = os.getenv("QWEN3_TTS_ASR_MODEL", "base")
        device = os.getenv("QWEN3_TTS_ASR_DEVICE", "cuda")
        compute_type = os.getenv("QWEN3_TTS_ASR_COMPUTE_TYPE", "float16")
        LOGGER.info("Loading Whisper model: %s", model_name)
        _WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as fp:
        fp.write(audio_bytes)
        tmp = fp.name
    try:
        segments, _ = _WHISPER.transcribe(tmp, beam_size=1, vad_filter=True)
        return "".join(seg.text.strip() for seg in segments).strip()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _decode_audio(data: bytes):
    """Decode audio bytes to numpy array."""
    import librosa
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
        Path(tmp).unlink(missing_ok=True)


def _save_profile(items, checkpoint):
    """Save voice profile for reuse."""
    voice_id = str(uuid.uuid4())
    payload = {
        "version": 1,
        "checkpoint": checkpoint,
        "items": [],
    }
    for it in items:
        payload["items"].append({
            "ref_code": it.ref_code.cpu() if torch.is_tensor(it.ref_code) else None,
            "ref_spk_embedding": it.ref_spk_embedding.cpu(),
            "x_vector_only_mode": bool(it.x_vector_only_mode),
            "icl_mode": bool(it.icl_mode),
            "ref_text": it.ref_text,
        })
    torch.save(payload, _PROFILE_DIR / f"{voice_id}.pt")
    return voice_id


def _load_profile(voice_id):
    """Load a saved voice profile."""
    from qwen_tts import VoiceClonePromptItem
    target = _PROFILE_DIR / f"{voice_id}.pt"
    if not target.exists():
        raise ValueError(f"voice_id not found: {voice_id}")
    payload = torch.load(target, map_location="cpu")
    out = []
    for d in payload.get("items", []):
        ref_code = d.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        out.append(VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=d["ref_spk_embedding"],
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            icl_mode=bool(d.get("icl_mode", True)),
            ref_text=d.get("ref_text"),
        ))
    return out


# ── Main handler ────────────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler function."""
    try:
        inp = job["input"]
        model = _get_model()

        text = inp.get("text", "").strip()
        if not text:
            return {"error": "text is required"}

        language = inp.get("language", "Auto")
        x_vector_only = inp.get("x_vector_only", False)
        save_profile = inp.get("save_profile", False)

        # ── Mode 1: Reuse existing voice profile ──
        voice_id = inp.get("voice_id")
        if voice_id:
            prompt_items = _load_profile(voice_id)
        else:
            # ── Mode 2: New voice clone from reference audio ──
            ref_audio_b64 = inp.get("ref_audio_base64")
            if not ref_audio_b64:
                return {"error": "ref_audio_base64 or voice_id is required"}

            audio_bytes = base64.b64decode(ref_audio_b64)
            wav, sr = _decode_audio(audio_bytes)

            ref_text = inp.get("ref_text", "").strip()
            if not ref_text and not x_vector_only:
                ref_text = _transcribe(audio_bytes)
                if not ref_text:
                    return {"error": "ref_text is required (auto-transcription returned empty)"}

            prompt_items = model.create_voice_clone_prompt(
                ref_audio=(wav, sr),
                ref_text=ref_text or None,
                x_vector_only_mode=bool(x_vector_only),
            )
            voice_id = None

        # ── Generate speech ──
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
        )

        # Encode result
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0].astype(np.float32), int(sample_rate), format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        result = {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "ref_text": ref_text if not inp.get("voice_id") else None,
        }

        # Save profile if requested
        if save_profile and not inp.get("voice_id"):
            checkpoint = os.getenv("QWEN3_TTS_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            result["voice_id"] = _save_profile(prompt_items, checkpoint)

        return result

    except Exception as e:
        LOGGER.error("Handler error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Entry point ─────────────────────────────────────────────────────────

runpod.serverless.start({"handler": handler})
