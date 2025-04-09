# CSM API 클라이언트 - Google Colab 테스트용
import requests
import base64
import io
from IPython.display import Audio

class CSMClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def generate_speech(self, text, speaker=0):
        """텍스트를 음성으로 변환하고 Audio 객체로 반환합니다."""
        try:
            response = requests.post(
                f"{self.api_url}/generate-speech/",
                json={"text": text, "speaker": speaker}
            )
            response.raise_for_status()
            
            # Base64 디코딩
            audio_data = base64.b64decode(response.json()["audio"])
            
            # Audio 객체 생성 및 반환
            return Audio(audio_data, autoplay=False)
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return None

# 사용 예시
if __name__ == "__main__":
    # CSM 클라이언트 초기화
    client = CSMClient()
    
    # 테스트 텍스트
    test_text = "안녕하세요. CSM 음성 생성 API 테스트입니다."
    
    # 음성 생성 및 재생
    audio = client.generate_speech(test_text)
    if audio:
        display(audio)  # Colab에서 오디오 재생 위젯 표시 