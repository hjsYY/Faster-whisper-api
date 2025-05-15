import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time


fasterModel = WhisperModel("large", device="cuda", compute_type="float16")


# 音频参数
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # 每次录制 3 秒
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

print("开始说话吧...\n")

try:
    while True:
        audio_data = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio_array = audio_data.flatten()
        
        segments, info = fasterModel.transcribe(
            audio_array,
            beam_size=5,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )


        for segment in segments:
            print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")

except KeyboardInterrupt:
    print("\n 已退出")
