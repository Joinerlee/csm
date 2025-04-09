# CSM 음성 생성 API

이 프로젝트는 CSM(Conversational Speech Model)을 활용해 텍스트를 자연스러운 음성으로 변환하는 API 서버입니다. 기존 시스템에 음성 생성 기능을 쉽게 통합할 수 있습니다.

## 구성 요소

1. **CSM API 서버 (app.py)**: GPU 서버에서 실행, 텍스트를 받아 음성을 생성합니다.
2. **CSM 클라이언트 (csm_client.py)**: 기존 백엔드에서 CSM API를 호출하는 라이브러리입니다.

## 설치 및 실행

### CSM API 서버 (GPU 서버)

**요구 사항**
- CUDA 호환 GPU
- Python 3.10 이상
- 다음 Hugging Face 모델에 대한 접근 권한:
  - [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  - [CSM-1B](https://huggingface.co/sesame/csm-1b)

**설치**
```bash
git clone https://github.com/Joinerlee/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn

# Mimi에서 지연 컴파일 비활성화
export NO_TORCH_COMPILE=1

# Hugging Face 모델 접근을 위한 로그인
huggingface-cli login
```

**실행**
```bash
# app.py 파일을 csm 디렉토리에 복사한 후
python app.py
```

서버는 `0.0.0.0:8000` 주소에서 실행되며, 외부에서 GPU 서버의 IP 주소를 통해 접근할 수 있습니다.

**네트워크 설정**
1. GPU 서버의 IP 주소(예: 192.168.1.100)와 포트(8000)를 확인합니다.
2. 필요한 경우 방화벽 설정을 변경하여 해당 포트로의 접근을 허용합니다.
3. 보안 연결이 필요한 경우 HTTPS 설정이나 VPN을 고려하세요.

## API 엔드포인트

- **GET /** - API 상태 확인
- **POST /generate-speech/** - 텍스트를 음성으로 변환
  - 요청 본문: `{ "text": "변환할 텍스트입니다.", "speaker": 0 }`
  - 응답: MP3 파일
  - 특징: 
    - 마침표(.)로 구분된 각 문장은 최대 30초로 제한됩니다
    - 문장 사이에 1초의 묵음이 자동으로 추가됩니다

## 기존 백엔드에 통합하는 방법

`csm_client.py` 파일을 기존 백엔드 프로젝트에 복사한 후, 다음과 같이 사용할 수 있습니다:

```python
# CSM 클라이언트 가져오기
from csm_client import CSMClient

# CSM API 클라이언트 초기화 (GPU 서버의 실제 IP 주소로 변경)
csm_client = CSMClient(api_url="http://192.168.1.100:8000")

# 텍스트를 음성으로 변환하는 예시
def convert_text_to_speech(text, speaker=0):
    # 방법 1: Base64 인코딩된 오디오 데이터 얻기
    success, audio_base64, info = csm_client.generate_speech_base64(text, speaker)
    
    if success:
        # Base64 인코딩된 데이터를 프론트엔드로 반환
        return {
            "success": True,
            "audio_base64": audio_base64
        }
    else:
        return {
            "success": False,
            "error": audio_base64  # 오류 메시지
        }
    
    # 방법 2: 파일로 저장
    # success, message, info = csm_client.generate_speech_to_file(
    #     text=text,
    #     output_file="output.mp3",
    #     speaker=speaker
    # )
```

## 주요 특징

- 마침표(.)를 기준으로 문장을 분리하여 자연스러운 음성 생성
- 동일한 화자 ID로 일관된 목소리 유지
- 원어민 수준의 자연스러운 발음
- 문장 사이에 1초의 간격을 두어 자연스러운 호흡 제공
- 각 문장은 최대 30초로 제한되어 안정적인 결과물 생성
- MP3 형식으로 오디오 반환
- 기존 시스템에 쉽게 통합 가능

## 주의사항

1. GPU 메모리 사용량을 모니터링하여 서버 성능을 최적화하세요.
2. 처리 시간은 텍스트 길이와 GPU 성능에 따라 달라질 수 있습니다.
3. 긴 텍스트의 경우 요청 타임아웃을 적절히 설정하세요.
4. 네트워크 지연 시간을 고려하여 백엔드에서 적절한 타임아웃 값을 설정하세요. 

# 기존 서버 종료 후 재시작
cd ~/csm
python app.py 