import requests
import base64
import os
import logging
from typing import Optional, Dict, Union, Tuple, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CSMClient")

class CSMClient:
    """
    CSM 음성 생성 API 클라이언트 클래스
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 120):
        """
        CSM API 클라이언트 초기화
        
        Args:
            api_url (str): CSM API 서버의 기본 URL (예: http://192.168.1.100:8000)
            timeout (int): API 요청 타임아웃(초), 기본값 120초
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.generate_speech_endpoint = f"{self.api_url}/generate-speech/"
        
        # API 연결 테스트
        try:
            self._check_connection()
            logger.info(f"CSM API 서버에 성공적으로 연결되었습니다: {api_url}")
        except Exception as e:
            logger.warning(f"CSM API 서버 연결 중 경고: {str(e)}")
    
    def _check_connection(self) -> bool:
        """
        API 서버에 연결을 테스트합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)  # 상태 확인은 짧은 타임아웃 사용
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"API 서버가 비정상 응답을 반환했습니다: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"API 서버 연결 실패: {str(e)}")
            return False
    
    def generate_speech(self, text: str, speaker: int = 0) -> Tuple[bool, Union[bytes, str], Dict[str, Any]]:
        """
        텍스트를 음성으로 변환합니다.
        
        Args:
            text (str): 변환할 텍스트
            speaker (int, optional): 화자 ID. 기본값은 0.
            
        Returns:
            Tuple[bool, Union[bytes, str], Dict[str, Any]]: 
                - 성공 여부
                - 성공 시 MP3 오디오 데이터(bytes), 실패 시 오류 메시지(str)
                - 추가 정보 딕셔너리
        """
        try:
            # 텍스트가 비어있는지 확인
            if not text or not text.strip():
                return False, "텍스트가 비어있습니다.", {"error": "empty_text"}
            
            # 요청 데이터
            payload = {
                "text": text,
                "speaker": speaker
            }
            
            # 텍스트 길이에 따라 타임아웃 동적 조정 (선택 사항)
            # 긴 텍스트의 경우 타임아웃을 더 길게 설정
            dynamic_timeout = min(max(self.timeout, len(text) // 20), 600)  # 최대 10분
            
            logger.info(f"CSM API 요청: text_length={len(text)}, speaker={speaker}, timeout={dynamic_timeout}초")
            
            # API 요청
            response = requests.post(
                self.generate_speech_endpoint,
                json=payload,
                timeout=dynamic_timeout
            )
            
            # 응답 처리
            if response.status_code == 200:
                # 성공적으로 음성 생성
                audio_data = response.content
                logger.info(f"음성 생성 성공: {len(audio_data)} 바이트")
                return True, audio_data, {"content_type": "audio/mp3", "size": len(audio_data)}
            else:
                # API 오류
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", "Unknown error")
                except:
                    pass
                
                logger.error(f"API 오류: {response.status_code}, {error_detail}")
                return False, f"CSM API 오류: {error_detail}", {"status_code": response.status_code}
                
        except requests.Timeout:
            error_msg = "API 요청 시간 초과"
            logger.error(error_msg)
            return False, error_msg, {"error": "timeout"}
            
        except requests.RequestException as e:
            error_msg = f"API 요청 실패: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"error": "request_failed"}
            
        except Exception as e:
            error_msg = f"예상치 못한 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"error": "unexpected_error"}
    
    def generate_speech_to_file(self, text: str, output_file: str, speaker: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
        """
        텍스트를 음성으로 변환하여 파일로 저장합니다.
        
        Args:
            text (str): 변환할 텍스트
            output_file (str): 저장할 파일 경로
            speaker (int, optional): 화자 ID. 기본값은 0.
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: 
                - 성공 여부
                - 메시지
                - 추가 정보 딕셔너리
        """
        success, result, info = self.generate_speech(text, speaker)
        
        if success:
            # 결과가 오디오 데이터인 경우
            try:
                # 파일로 저장
                with open(output_file, "wb") as f:
                    f.write(result)
                
                logger.info(f"오디오 파일 저장 완료: {output_file}")
                return True, f"오디오 파일이 저장되었습니다: {output_file}", info
            except Exception as e:
                error_msg = f"파일 저장 실패: {str(e)}"
                logger.error(error_msg)
                return False, error_msg, {"error": "file_save_failed"}
        else:
            # 이미 오류가 발생한 경우
            return success, result, info
    
    def generate_speech_base64(self, text: str, speaker: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
        """
        텍스트를 음성으로 변환하여 Base64 인코딩된 문자열로 반환합니다.
        
        Args:
            text (str): 변환할 텍스트
            speaker (int, optional): 화자 ID. 기본값은 0.
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: 
                - 성공 여부
                - 성공 시 Base64로 인코딩된 MP3 데이터, 실패 시 오류 메시지
                - 추가 정보 딕셔너리
        """
        success, result, info = self.generate_speech(text, speaker)
        
        if success:
            # 바이너리 데이터를 Base64로 인코딩
            try:
                base64_audio = base64.b64encode(result).decode('utf-8')
                return True, base64_audio, info
            except Exception as e:
                error_msg = f"Base64 인코딩 실패: {str(e)}"
                logger.error(error_msg)
                return False, error_msg, {"error": "encoding_failed"}
        else:
            # 이미 오류가 발생한 경우
            return success, result, info


# 사용 예시
if __name__ == "__main__":
    # 클라이언트 초기화 (GPU 서버의 실제 IP 주소로 변경하세요)
    client = CSMClient(api_url="http://192.168.1.100:8000", timeout=180)
    
    # 텍스트 정의
    sample_text = "안녕하세요. CSM 음성 생성 모델입니다. 한국어 음성을 생성할 수 있습니다."
    
    # 1. 파일로 저장하는 예시
    success, message, info = client.generate_speech_to_file(
        text=sample_text,
        output_file="output.mp3",
        speaker=0
    )
    
    if success:
        print(f"성공: {message}")
        print(f"파일 크기: {info['size']} 바이트")
    else:
        print(f"오류: {message}")
    
    # 2. Base64로 인코딩된 문자열로 받는 예시
    success, result, info = client.generate_speech_base64(
        text=sample_text,
        speaker=0
    )
    
    if success:
        print(f"Base64 인코딩 성공: {result[:50]}...")  # 처음 50자만 출력
    else:
        print(f"오류: {result}") 