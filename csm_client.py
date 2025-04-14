import requests
import json
import argparse
import os
import tempfile
import time
from datetime import datetime
import sys
import subprocess

def play_audio(file_path):
    """오디오 파일 재생"""
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

def generate_speech(server_url, text):
    """CSM-1B 서버에 음성 생성 요청"""
    if not server_url.startswith("http"):
        server_url = f"http://{server_url}"
    
    api_url = f"{server_url}/csm-1b"
    
    # 한국어 텍스트 감지
    contains_korean = any((ord(char) >= 0xAC00 and ord(char) <= 0xD7A3) for char in text)
    if contains_korean:
        print("한국어 텍스트 감지됨 - 한국어 음성으로 생성합니다.")
    
    # 요청 데이터 준비
    payload = {
        "text": text
    }
    
    print(f"서버에 요청 전송 중: {api_url}")
    print(f"텍스트: {text[:50]}..." if len(text) > 50 else f"텍스트: {text}")
    
    try:
        # 요청 시작 시간
        start_time = time.time()
        
        # API 요청 전송
        response = requests.post(api_url, json=payload, timeout=180)  # 타임아웃 3분으로 늘림
        
        # 요청 완료 시간
        elapsed_time = time.time() - start_time
        
        # 응답 처리
        if response.status_code == 200:
            # 임시 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(tempfile.gettempdir(), "csm_tts")
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일 이름에 한국어 여부 표시
            lang_marker = "ko" if contains_korean else "en"
            output_file = os.path.join(output_dir, f"csm_tts_{lang_marker}_{timestamp}.wav")
            
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            # 결과 출력
            sentences = len([s for s in text.split(".") if s.strip()])
            print(f"음성 파일 저장 완료: {output_file}")
            print(f"처리 시간: {elapsed_time:.2f}초 (약 {sentences}개 문장, 평균: {elapsed_time/max(1,sentences):.2f}초/문장)")
            
            # 음성 재생
            play_audio(output_file)
            return output_file
        else:
            print(f"오류 발생 - 상태 코드: {response.status_code}")
            try:
                error_data = response.json()
                print(f"오류 메시지: {error_data}")
            except:
                print(f"응답: {response.text[:200]}")
            return None
    
    except Exception as e:
        print(f"요청 중 오류 발생: {e}")
        return None

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="CSM-1B TTS 클라이언트")
    parser.add_argument("--url", required=True, help="CSM-1B 서버 URL (예: ngrok URL)")
    parser.add_argument("--text", help="음성으로 변환할 텍스트")
    parser.add_argument("--file", help="텍스트 파일 경로")
    parser.add_argument("--save-dir", help="오디오 파일 저장 디렉토리 (기본: 임시 폴더)")
    
    args = parser.parse_args()
    
    # 저장 디렉토리 설정
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        tempfile.tempdir = args.save_dir
    
    # 텍스트 입력 소스 확인
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            print(f"파일에서 텍스트 로드 완료: {args.file}")
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return
    elif args.text:
        text = args.text
    else:
        print("텍스트를 입력하세요 (종료하려면 빈 줄 입력):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        text = "\n".join(lines)
    
    if not text:
        print("텍스트가 비어있습니다.")
        return
    
    generate_speech(args.url, text)

if __name__ == "__main__":
    main() 