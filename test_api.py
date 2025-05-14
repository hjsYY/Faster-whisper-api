from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import os
import uuid

app = FastAPI(
    title="Fast-Whisper API",
    description="API for speech recognition using Fast-Whisper",
    version="1.0.0"
)
model_size = "large"
# 初始化模型 (可以调整为更大的模型如 small, medium)
model = WhisperModel(model_size, device="cuda", compute_type="float16")
@app.post("/transcribe")
async def transcribe_audio(
        file: UploadFile = File(..., description="音频文件(WAV, MP3等格式)")
):
    # 验证文件类型
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="不支持的文件类型，请上传WAV或MP3格式音频"
        )

    # 创建临时文件
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{temp_dir}/{str(uuid.uuid4())}.mp3"

    try:
        # 保存上传的文件
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # 使用Fast-Whisper进行转录
        segments, info = model.transcribe(temp_filename, beam_size=5)

        # 拼接所有片段
        full_text = " ".join([segment.text for segment in segments])

        return JSONResponse({
            "language": info.language,
            "duration": round(info.duration, 2),
            "text": full_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "model": "base"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)