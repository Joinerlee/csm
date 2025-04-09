"""
Google Colab에서 CSM 음성 생성 API 서버를 실행하는 스크립트
"""

import os
import sys
import subprocess
import time

print("CSM 음성 생성 API - Colab 버전")
print("=" * 50)

# 필요한 라이브러리 설치
print("\n1. 필요한 라이브러리 설치 중...")
subprocess.run(["pip", "install", "fastapi", "uvicorn", "pyngrok", "nest_asyncio", "torchaudio", "-q"], check=True)

# 환경 변수 설정
os.environ["NO_TORCH_COMPILE"] = "1"
print("환경 변수 설정: NO_TORCH_COMPILE=1")

# Hugging Face 토큰 설정 (필요한 경우)
hf_token = input("\nHugging Face 토큰을 입력하세요 (없으면 Enter): ")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("Hugging Face에 로그인되었습니다.")

# CSM 리포지토리 클론
print("\n2. CSM 리포지토리 클론 중...")
if not os.path.exists("csm"):
    subprocess.run(["git", "clone", "https://github.com/SesameAILabs/csm.git"], check=True)
    print("CSM 리포지토리 클론 완료")
else:
    print("CSM 리포지토리가 이미 존재합니다")

# 필요한 패키지 설치
print("\n3. 필요한 패키지 설치 중...")
subprocess.run(["pip", "install", "-r", "csm/requirements.txt", "-q"], check=True)
print("패키지 설치 완료")

# FastAPI 애플리케이션 코드 작성
print("\n4. FastAPI 애플리케이션 코드 작성 중...")
app_code = """
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
    allow_origins=["*"],  # 모든 오리진 허용
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
"""

with open("csm/app.py", "w") as f:
    f.write(app_code)
print("FastAPI 애플리케이션 코드 작성 완료")

# 클라이언트 코드 작성
print("\n5. 클라이언트 코드 작성 중...")
client_code = """
import requests
import base64

# ngrok URL 설정 (실제 URL로 변경하세요)
NGROK_URL = "여기에 ngrok URL을 입력하세요"  # 예: https://xxxx-xx-xxx-xx-xx.ngrok-free.app

def generate_speech(text, speaker=0):
    """
    CSM API를 사용하여 텍스트를 음성으로 변환합니다.
    
    Args:
        text (str): 변환할 텍스트
        speaker (int, optional): 화자 ID. 기본값은 0.
        
    Returns:
        bool: 성공 여부
        bytes or str: 성공 시 MP3 데이터, 실패 시 오류 메시지
    """
    try:
        # 요청 데이터
        payload = {
            "text": text,
            "speaker": speaker
        }
        
        # API 요청
        response = requests.post(
            f"{NGROK_URL}/generate-speech/",
            json=payload,
            timeout=120  # 긴 텍스트의 경우 타임아웃 증가
        )
        
        # 응답 확인
        if response.status_code == 200:
            return True, response.content
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", "Unknown error")
            except:
                pass
                
            return False, f"API 오류: {error_msg}"
    
    except Exception as e:
        return False, f"요청 실패: {str(e)}"

def generate_speech_to_file(text, output_file="output.mp3", speaker=0):
    """
    텍스트를 음성으로 변환하여 파일로 저장합니다.
    
    Args:
        text (str): 변환할 텍스트
        output_file (str): 저장할 파일 경로
        speaker (int, optional): 화자 ID. 기본값은 0.
        
    Returns:
        bool: 성공 여부
        str: 메시지
    """
    success, result = generate_speech(text, speaker)
    
    if success:
        try:
            with open(output_file, "wb") as f:
                f.write(result)
            return True, f"음성 파일이 저장되었습니다: {output_file}"
        except Exception as e:
            return False, f"파일 저장 실패: {str(e)}"
    else:
        return False, result

def generate_speech_base64(text, speaker=0):
    """
    텍스트를 음성으로 변환하여 Base64 인코딩된 문자열로 반환합니다.
    
    Args:
        text (str): 변환할 텍스트
        speaker (int, optional): 화자 ID. 기본값은 0.
        
    Returns:
        bool: 성공 여부
        str: 성공 시 Base64로 인코딩된 MP3 데이터, 실패 시 오류 메시지
    """
    success, result = generate_speech(text, speaker)
    
    if success:
        try:
            base64_audio = base64.b64encode(result).decode('utf-8')
            return True, base64_audio
        except Exception as e:
            return False, f"Base64 인코딩 실패: {str(e)}"
    else:
        return False, result

# 사용 예시
if __name__ == "__main__":
    # 테스트 텍스트
    test_text = "안녕하세요. CSM 음성 생성 모델입니다. 한국어 음성을 생성할 수 있습니다."
    
    # 파일로 저장하는 예시
    success, message = generate_speech_to_file(test_text, "test_output.mp3")
    print(message)
"""

with open("client_sample.py", "w") as f:
    f.write(client_code)
print("클라이언트 코드 작성 완료")

# ngrok 설정 및 서버 실행
print("\n6. ngrok 설정 및 서버 실행...")

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.getcwd())

# nest_asyncio 설정
import nest_asyncio
nest_asyncio.apply()

# pyngrok 임포트 및 터널 설정
from pyngrok import ngrok, conf

# ngrok 인증 토큰 입력 (필요한 경우)
ngrok_token = input("\nngrok 인증 토큰을 입력하세요 (없으면 Enter): ")
if ngrok_token:
    conf.get_default().auth_token = ngrok_token
    print("ngrok 인증 토큰이 설정되었습니다.")

# 서버 프로세스 시작
print("\n서버를 백그라운드에서 시작합니다...")
server_process = subprocess.Popen(["python", "csm/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 서버가 시작될 때까지 잠시 대기
print("서버가 시작될 때까지 10초 대기...")
time.sleep(10)

# ngrok 터널 설정 (포트 8000)
print("ngrok 터널 설정 중...")
ngrok_tunnel = ngrok.connect(8000)
print(f'ngrok 터널 URL: {ngrok_tunnel.public_url}')

# 클라이언트 예제 업데이트
with open("client_sample.py", "r") as f:
    updated_client = f.read().replace("여기에 ngrok URL을 입력하세요", ngrok_tunnel.public_url)

with open("client_sample.py", "w") as f:
    f.write(updated_client)
print(f"클라이언트 예제가 업데이트되었습니다. (ngrok URL: {ngrok_tunnel.public_url})")

# 테스트 방법 안내
print("\n7. API 테스트 방법")
print("-" * 50)
print(f"API 엔드포인트: {ngrok_tunnel.public_url}/generate-speech/")
print("요청 예시:")
print("""
import requests

# API 요청
response = requests.post(
    "{}/generate-speech/",
    json={{"text": "안녕하세요. CSM 음성 생성 모델입니다.", "speaker": 0}}
)

# 응답 처리
if response.status_code == 200:
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    print("음성 파일이 생성되었습니다: output.mp3")
else:
    print(f"오류 발생: {{response.status_code}}")
    print(response.text)
""".format(ngrok_tunnel.public_url))

# 종료 방법 안내
print("\n8. 종료 방법")
print("-" * 50)
print("테스트를 마치면 다음 코드를 실행하여 서버와 ngrok 터널을 종료하세요:")
print("""
from pyngrok import ngrok
import os
import signal

# ngrok 터널 종료
ngrok.kill()

# 서버 프로세스 종료
os.kill(서버_프로세스_ID, signal.SIGTERM)  # 서버_프로세스_ID는 위에서 출력된 PID로 변경하세요
""")

print(f"\n서버 프로세스 ID: {server_process.pid}")
print("\nCSM API 서버와 ngrok 터널이 실행 중입니다. 위 안내에 따라 API를 테스트하세요.")
print("=" * 50)

# 종료 여부 확인
try:
    input("\n서버를 종료하려면 Enter 키를 누르세요...")
    print("서버와 ngrok 터널을 종료합니다...")
    ngrok.kill()
    server_process.terminate()
    print("종료되었습니다.")
except KeyboardInterrupt:
    print("\n중단 신호를 받았습니다. 서버와 ngrok 터널을 종료합니다...")
    ngrok.kill()
    server_process.terminate()
    print("종료되었습니다.") 