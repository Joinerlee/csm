# CSM 음성 생성 API - Google Colab 설정
# 이 코드는 Google Colab에서 CSM API를 실행하기 위한 설정입니다.

# 1. 필요한 패키지 설치
!pip install -q fastapi uvicorn
!pip install -q transformers torch torchaudio
!pip install -q huggingface_hub
!pip install -q python-multipart
!pip install -q pyngrok
!pip install -q nest-asyncio

# 2. CSM 저장소 클론
!git clone https://github.com/Joinerlee/csm.git
%cd csm

# 3. Hugging Face 모델 접근을 위한 로그인
from huggingface_hub import login
# 아래에 Hugging Face 토큰을 입력하세요
# login(token="your_token_here")

# 4. ngrok 설정
from pyngrok import ngrok
import nest_asyncio
nest_asyncio.apply()

# ngrok 인증 토큰 설정 (필요한 경우)
# ngrok.set_auth_token("your_ngrok_auth_token")

# ngrok 터널 생성
public_url = ngrok.connect(8000)
print(f"ngrok 터널 URL: {public_url}")

# 5. 서버 실행
!python app.py 