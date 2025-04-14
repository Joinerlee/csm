# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import os
import shutil
from typing import Optional
import requests
import json

app = FastAPI(title="TTS API 서버", description="텍스트를 음성으로 변환하는 API 서버")

# 임시 디렉토리 생성 (실제 서비스에서는 영구 저장소를 사용할 수 있음)
TEMP_DIR = tempfile.mkdtemp()
# RunPod 서버 주소 (실제 배포 시 환경 변수로 설정)
RUNPOD_SERVER_URL = os.getenv("RUNPOD_SERVER_URL", "http://localhost:8000")

@app.post("/tts/")
async def text_to_speech(text: str = Form(...)):
    """
    텍스트를 음성으로 변환합니다.
    """
    try:
        # RunPod 서버로 요청 전송
        response = requests.post(
            f"{RUNPOD_SERVER_URL}/process", 
            json={"text": text}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS 처리 중 오류가 발생했습니다.")
        
        # 임시 파일로 저장
        audio_path = os.path.join(TEMP_DIR, f"output_{hash(text)}.wav")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        # 파일로 응답
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 임시 파일 정리"""
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True) 