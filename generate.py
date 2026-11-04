from pathlib import Path
import time

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "model_cache"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Please check the GPU environment.")

print("GPU:", torch.cuda.get_device_name(0))
print("1/3 Downloading or loading MusicGen small...", flush=True)

model_id = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
)

model = MusicgenForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
    torch_dtype=torch.float16,
).to("cuda")

model.eval()

prompt = "Gentle relaxing piano music with soft ambient pads, instrumental"

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
).to("cuda")

print("2/3 Generating about 5 seconds of music...", flush=True)
start = time.perf_counter()

with torch.inference_mode():
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=256,
    )

audio = audio_values[0, 0].float().cpu().numpy()

if not np.isfinite(audio).all():
    raise RuntimeError("Generated audio contains invalid values.")

sample_rate = model.config.audio_encoder.sampling_rate
pcm_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

output_path = OUTPUT_DIR / "first_music.wav"
write(str(output_path), sample_rate, pcm_audio)

print(f"3/3 Saved: {output_path}")
print(f"Audio duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Generation time: {time.perf_counter() - start:.1f} seconds")