# run_local.py
import os
import subprocess
import time
import signal
import sys

def run_servers():
    """FastAPI 서버와 RunPod 서비스를 실행합니다."""
    try:
        # 환경 변수 설정
        os.environ["NO_TORCH_COMPILE"] = "1"
        
        # 서버 프로세스 시작
        print("RunPod TTS 서비스 시작 중...")
        runpod_process = subprocess.Popen(
            ["python", "tts_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 2초 대기
        time.sleep(2)
        
        print("FastAPI 서버 시작 중...")
        fastapi_process = subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("\n=== TTS 시스템이 실행 중입니다 ===")
        print("FastAPI 서버: http://localhost:8080")
        print("RunPod 서비스: http://localhost:8000")
        print("\n종료하려면 Ctrl+C를 누르세요...\n")
        
        # 서버 실행 중 대기
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n서버 종료 중...")
        # 프로세스 종료
        fastapi_process.terminate()
        runpod_process.terminate()
        print("서버가 종료되었습니다.")
        sys.exit(0)

if __name__ == "__main__":
    run_servers() 