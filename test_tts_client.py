import requests
import sys
import os
import tempfile
import argparse
from datetime import datetime
import subprocess

def play_audio(file_path):
    """음성 파일 재생"""
    try:
        if sys.platform == "win32":
            # Windows
            os.startfile(file_path)
        elif sys.platform == "darwin":
            # macOS
            subprocess.call(["afplay", file_path])
        else:
            # Linux
            subprocess.call(["aplay", file_path])
        print(f"재생 중: {file_path}")
    except Exception as e:
        print(f"오디오 재생 중 오류 발생: {e}")

def tts_request(server_url, text, language="auto"):
    """TTS API 요청 전송"""
    # 서버 URL 확인
    if not server_url.startswith("http"):
        server_url = f"http://{server_url}"
    
    api_url = f"{server_url}/api/tts"
    
    # 언어 자동 감지 (한국어 포함 여부 확인)
    if language == "auto":
        if any((ord(char) >= 0xAC00 and ord(char) <= 0xD7A3) for char in text):
            language = "ko"
        else:
            language = "en"
    
    # 요청 데이터 준비
    payload = {
        "text": text,
        "language": language
    }
    
    print(f"서버에 요청 전송 중: {api_url}")
    print(f"텍스트: {text[:50]}..." if len(text) > 50 else f"텍스트: {text}")
    
    try:
        # API 요청 전송
        response = requests.post(api_url, json=payload)
        
        # 응답 처리
        if response.status_code == 200:
            # 임시 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(tempfile.gettempdir(), f"tts_output_{timestamp}.wav")
            
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"음성 파일 저장 완료: {output_file}")
            
            # 음성 재생
            play_audio(output_file)
            return output_file
        else:
            print(f"오류 발생 - 상태 코드: {response.status_code}")
            try:
                error_data = response.json()
                print(f"오류 메시지: {error_data.get('error', '알 수 없는 오류')}")
            except:
                print(f"응답: {response.text}")
            return None
    
    except Exception as e:
        print(f"요청 중 오류 발생: {e}")
        return None

def check_server_status(server_url):
    """서버 상태 확인"""
    if not server_url.startswith("http"):
        server_url = f"http://{server_url}"
    
    status_url = f"{server_url}/api/status"
    
    try:
        response = requests.get(status_url)
        if response.status_code == 200:
            status_data = response.json()
            print(f"서버 상태: {status_data.get('status', 'unknown')}")
            print(f"공개 URL: {status_data.get('public_url', 'unknown')}")
            print(f"서버 시간: {status_data.get('time', 'unknown')}")
            return True
        else:
            print(f"서버 상태 확인 실패 - 상태 코드: {response.status_code}")
            return False
    except Exception as e:
        print(f"서버 상태 확인 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="TTS API 테스트 클라이언트")
    parser.add_argument("--url", required=True, help="TTS 서버 URL (예: ngrok URL)")
    parser.add_argument("--text", help="음성으로 변환할 텍스트")
    parser.add_argument("--language", choices=["auto", "en", "ko"], default="auto", 
                        help="텍스트 언어 (기본값: auto)")
    parser.add_argument("--status", action="store_true", help="서버 상태만 확인")
    
    args = parser.parse_args()
    
    # 서버 상태 확인
    if args.status:
        check_server_status(args.url)
        return
    
    # 텍스트 음성 변환 요청
    if not args.text:
        text = input("음성으로 변환할 텍스트를 입력하세요: ")
    else:
        text = args.text
    
    tts_request(args.url, text, args.language)

if __name__ == "__main__":
    main() 