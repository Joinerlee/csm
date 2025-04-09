from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import os
import tempfile
from typing import List

# Import CSM generator
from generator import load_csm_1b, Segment

# Create FastAPI app
app = FastAPI(title="CSM 음성 생성 API")

# CORS 미들웨어 설정 - 백엔드 서버에서의 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (실제 운영 환경에서는 백엔드 서버 URL로 제한하세요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model only once when app starts
print(f"Loading CSM model on {device}...")
generator = load_csm_1b(device=device)
print("Model loaded successfully.")

class TextRequest(BaseModel):
    text: str
    speaker: int = 0  # 기본 화자 ID

@app.get("/")
async def root():
    """
    API 상태 확인 엔드포인트
    """
    return {"status": "정상 작동 중", "message": "CSM 음성 생성 API가 준비되었습니다."}

@app.post("/generate-speech/")
async def generate_speech(request: TextRequest):
    """
    텍스트 입력에서 음성을 생성합니다.
    마침표를 기준으로 텍스트를 분할하고 각 문장에 대해 동일한 화자로 오디오를 생성합니다.
    각 문장은 최대 30초로 제한되며, 문장 사이에 1초의 간격이 추가됩니다.
    MP3 파일을 반환합니다.
    """
    try:
        print(f"Received request: text length={len(request.text)}, speaker={request.speaker}")
        
        # 마침표로 텍스트를 분할하고 빈 문장 필터링
        sentences = [s.strip() + "." for s in request.text.split(".") if s.strip()]
        
        if not sentences:
            raise HTTPException(status_code=400, detail="입력 텍스트에서 유효한 문장을 찾을 수 없습니다")
        
        print(f"Processing {len(sentences)} sentences...")
        
        # 오디오 파일을 저장할 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            context_segments = []
            all_audio_segments = []
            
            # 각 문장에 대해 오디오 생성
            for i, sentence in enumerate(sentences):
                print(f"Generating audio for sentence {i+1}/{len(sentences)}: '{sentence}'")
                audio = generator.generate(
                    text=sentence,
                    speaker=request.speaker,
                    context=context_segments,  # 이전 세그먼트를 컨텍스트로 사용
                    max_audio_length_ms=30_000,  # 최대 30초(30,000ms)로 제한
                )
                
                # 오디오 세그먼트 저장
                segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
                torchaudio.save(segment_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
                all_audio_segments.append(audio)
                
                # 세그먼트 객체 생성 및 다음 반복을 위한 컨텍스트에 추가
                segment = Segment(
                    text=sentence,
                    speaker=request.speaker,
                    audio=audio
                )
                context_segments.append(segment)
            
            # 문장 사이에 1초의 간격을 추가
            sample_rate = generator.sample_rate
            silence_duration = 1  # 1초
            silence_samples = int(silence_duration * sample_rate)
            silence = torch.zeros(silence_samples)
            
            # 모든 오디오 세그먼트를 하나의 파일로 결합 (문장 사이에 1초 간격 추가)
            output_path = os.path.join(temp_dir, "output.mp3")
            
            # 첫 번째 세그먼트는 그대로 추가
            combined_segments = [all_audio_segments[0]]
            
            # 나머지 세그먼트는 1초 간격을 먼저 추가한 후 세그먼트 추가
            for segment in all_audio_segments[1:]:
                combined_segments.append(silence)
                combined_segments.append(segment)
            
            # 모든 오디오 세그먼트 연결
            print("Combining all audio segments with 1s silence between sentences...")
            combined_audio = torch.cat([seg.unsqueeze(0) for seg in combined_segments], dim=1)
            
            # MP3로 저장
            torchaudio.save(output_path, combined_audio, sample_rate, format="mp3")
            
            print(f"Audio generation complete. Total duration: {combined_audio.shape[1]/sample_rate:.2f}s")
            print(f"Returning MP3 file.")
            
            # 오디오 파일 반환
            return FileResponse(
                output_path, 
                media_type="audio/mp3",
                filename="generated_speech.mp3"
            )
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"음성 생성 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting CSM speech generation server...")
    # 0.0.0.0으로 변경하여 외부에서 접근 가능하도록 설정
    uvicorn.run(app, host="0.0.0.0", port=8000) 