import os
import sys
import subprocess
import json
import time
import requests
from flask import Flask, request, jsonify, send_file
import threading
import logging

# 원래 run_csm.py에서 음성 생성 함수 임포트
from run_csm import generate_with_gTTS, split_text_to_sentences, analyze_emotion

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ngrok 프로세스 저장 변수
ngrok_process = None
public_url = None

def setup_ngrok():
    """ngrok 설정 및 실행"""
    global ngrok_process, public_url
    
    try:
        # ngrok 설치 확인
        try:
            subprocess.check_output(["ngrok", "--version"])
            logger.info("ngrok이 이미 설치되어 있습니다.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("ngrok 설치 중...")
            if sys.platform == "win32":
                subprocess.run(["pip", "install", "pyngrok"], check=True)
                from pyngrok import ngrok as pyngrok
                pyngrok.install_ngrok()
            else:
                subprocess.run(["pip", "install", "pyngrok"], check=True)
                from pyngrok import ngrok as pyngrok
                pyngrok.install_ngrok()
        
        # ngrok 실행
        logger.info("ngrok 시작 중...")
        if sys.platform == "win32":
            # Windows
            from pyngrok import ngrok
            public_url = ngrok.connect(5000).public_url
        else:
            # Linux 또는 macOS
            ngrok_process = subprocess.Popen(
                ["ngrok", "http", "5000"], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # ngrok API에서 URL 가져오기
            time.sleep(2)  # ngrok이 시작될 때까지 대기
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()["tunnels"]
            public_url = tunnels[0]["public_url"]
        
        logger.info(f"ngrok 공개 URL: {public_url}")
        print(f"\n=== 외부에서 접속 가능한 URL ===\n{public_url}\n===========================\n")
        return public_url
    
    except Exception as e:
        logger.error(f"ngrok 설정 중 오류 발생: {e}")
        raise

@app.route('/api/tts', methods=['POST'])
def generate_tts():
    """TTS API 엔드포인트"""
    try:
        # 요청 데이터 검증
        if not request.json or 'text' not in request.json:
            return jsonify({"error": "텍스트가 제공되지 않았습니다"}), 400
        
        text = request.json['text']
        logger.info(f"TTS 요청 받음: {text[:30]}...")
        
        # 문장 분리
        sentences = split_text_to_sentences(text)
        if not sentences:
            sentences = [text]
        
        output_files = []
        
        # 각 문장 처리
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            # 감정 분석
            emotion, _ = analyze_emotion(sentence)
            
            # 파일 경로 설정
            output_path = f"temp_sentence_{i}_{emotion}.wav"
            
            # 한국어 텍스트 감지
            contains_korean = any((ord(char) >= 0xAC00 and ord(char) <= 0xD7A3) for char in sentence)
            lang = "ko" if contains_korean else "en"
            
            # gTTS로 음성 생성
            generate_with_gTTS(sentence, output_path, lang)
            output_files.append(output_path)
        
        # 단일 파일 반환 (여러 파일을 합치는 기능 필요시 추가)
        if output_files:
            return send_file(output_files[0], mimetype="audio/wav", as_attachment=True)
        else:
            return jsonify({"error": "음성 생성 실패"}), 500
    
    except Exception as e:
        logger.error(f"TTS 생성 중 오류: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """API 상태 확인 엔드포인트"""
    return jsonify({
        "status": "online",
        "public_url": public_url,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    })

def start_server():
    """Flask 서버 시작"""
    logger.info("Flask 서버 시작 중...")
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    try:
        # ngrok 설정
        setup_ngrok()
        
        # 서버 시작
        start_server()
    
    except KeyboardInterrupt:
        logger.info("서버 종료 중...")
        # ngrok 프로세스 종료
        if ngrok_process:
            ngrok_process.terminate()
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")
        if ngrok_process:
            ngrok_process.terminate()
        sys.exit(1) 