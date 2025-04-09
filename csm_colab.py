# CSM 음성 생성 API - Google Colab 버전
# 이 코드는 Google Colab에서 CSM API를 실행하기 위한 설정입니다.

# 1. 필요한 패키지 설치
!pip install -q fastapi uvicorn
!pip install -q transformers torch torchaudio
!pip install -q huggingface_hub
!pip install -q python-multipart

# 2. Hugging Face 모델 다운로드
from huggingface_hub import login
# 아래에 Hugging Face 토큰을 입력하세요
# login(token="your_token_here")

# 3. CSM API 서버 코드
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import io
import base64
from typing import Optional

app = FastAPI()

# 모델 및 프로세서 초기화
processor = AutoProcessor.from_pretrained("sesame/csm-1b")
model = AutoModelForSpeechSeq2Seq.from_pretrained("sesame/csm-1b")

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class SpeechRequest(BaseModel):
    text: str
    speaker: Optional[int] = 0

@app.get("/")
async def root():
    return {"status": "CSM API is running"}

@app.post("/generate-speech/")
async def generate_speech(request: SpeechRequest):
    try:
        # 텍스트를 문장으로 분리
        sentences = [s.strip() for s in request.text.split(".") if s.strip()]
        
        # 각 문장에 대한 음성 생성
        audio_segments = []
        for sentence in sentences:
            # 입력 텍스트 처리
            inputs = processor(text=sentence, return_tensors="pt").to(device)
            
            # 음성 생성
            with torch.no_grad():
                output = model.generate(**inputs, max_length=30*16000)  # 30초 제한
            
            # 오디오 데이터 변환
            audio = output.cpu().numpy().squeeze()
            audio_segments.append(audio)
            
            # 1초 묵음 추가
            silence = np.zeros(16000)  # 1초 묵음
            audio_segments.append(silence)
        
        # 모든 세그먼트 결합
        final_audio = np.concatenate(audio_segments)
        
        # MP3로 변환
        buffer = io.BytesIO()
        torchaudio.save(buffer, torch.tensor(final_audio).unsqueeze(0), 16000, format="mp3")
        audio_bytes = buffer.getvalue()
        
        # Base64 인코딩
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return {"audio": audio_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Colab에서 실행하기 위한 코드
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 