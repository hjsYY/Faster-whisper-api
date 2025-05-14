import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time

# 初始化模型（可换为 tiny / small / base）
fasterModel = WhisperModel("large", device="cuda", compute_type="float16")


# 音频参数
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # 每次录制 3 秒
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

print("开始说话吧...\n")

try:
    while True:
        # 1. 录制 3 秒音频（注意 channels=1 单声道）
        audio_data = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()  # 等待录音结束

        # 2. 转换为一维 numpy 数组
        audio_array = audio_data.flatten()

        # 3. Whisper 模型转录 + VAD 去静音
        segments, info = fasterModel.transcribe(
            audio_array,
            beam_size=5,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )

        # 4. 打印每一段结果
        for segment in segments:
            print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")

except KeyboardInterrupt:
    print("\n🛑 已退出")
