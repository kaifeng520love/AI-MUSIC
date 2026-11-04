"""沿用已验证的 Transformers / CUDA / FP16 / PCM WAV 路线。"""
import math
import time


class MusicService:
    def __init__(self, root):
        self.root = root
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用，请检查当前 Python 环境和显卡驱动。')
        cache = str(self.root / 'model_cache')
        # 复用已经下载成功的缓存；缺文件时报错，不悄悄重新下载。
        processor = AutoProcessor.from_pretrained('facebook/musicgen-small', cache_dir=cache, local_files_only=True)
        model = MusicgenForConditionalGeneration.from_pretrained(
            'facebook/musicgen-small', cache_dir=cache, local_files_only=True,
            torch_dtype=torch.float16).to('cuda')
        model.eval()
        self.processor, self.model = processor, model
        print('MusicGen loaded on', torch.cuda.get_device_name(0), flush=True)

    def generate(self, task):
        import numpy as np
        import torch
        from scipy.io.wavfile import write
        start = time.perf_counter()
        self.load()
        inputs = self.processor(text=[task['prompt']], padding=True, return_tensors='pt').to('cuda')
        # MusicGen 每秒约 50 个音频 token；实际音频长度单独记录。
        with torch.inference_mode():
            values = self.model.generate(**inputs, do_sample=True,
                guidance_scale=task['guidance_scale'], max_new_tokens=math.ceil(task['duration'] * 50))
        audio = values[0, 0].float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError('生成音频包含无效数值。')
        rate = self.model.config.audio_encoder.sampling_rate
        filename = task['id'] + '.wav'
        output = self.root / 'outputs' / filename
        temporary = output.with_suffix('.tmp')
        try:
            write(str(temporary), rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(filename=filename, actual_duration=len(audio) / rate,
                    elapsed_seconds=time.perf_counter() - start)
