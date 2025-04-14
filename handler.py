# handler.py
# RunPod에서 실행할 핸들러
import os
import time
import torch
import json
import base64
from urllib.parse import unquote

# 자체 모듈 임포트
from text_preprocessor import preprocess_text
from emotional_analysis import analyze_emotion
from audio_processor import merge_audio_files

# 임시 디렉토리
TEMP_DIR = "/tmp/csm_tts"
os.makedirs(TEMP_DIR, exist_ok=True)

def generate_audio_with_csm(text, output_path):
    """CSM-1B 모델을 사용하여 텍스트를 음성으로 변환합니다."""
    try:
        # CSM-1B 모델 로드 및 음성 생성 로직
        from generator import load_csm_1b, Segment
        import torchaudio
        
        # 장치 선택
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드
        generator = load_csm_1b(device=device)
        
        # 음성 생성
        audio = generator.generate(
            text=text,
            speaker=0,  # 기본 스피커
            context=[],  # 컨텍스트 없음
            max_audio_length_ms=10_000,
        )
        
        # 파일로 저장
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        return True
        
    except Exception as e:
        print(f"CSM-1B 모델 처리 중 오류: {str(e)}")
        return False

def generate_audio_with_gtts(text, output_path):
    """Google TTS를 사용하여 텍스트를 음성으로 변환합니다."""
    try:
        from gtts import gTTS
        
        # gTTS로 음성 생성
        tts = gTTS(text=text, lang='ko', slow=False)
        tts.save(output_path)
        return True
        
    except Exception as e:
        print(f"gTTS 처리 중 오류: {str(e)}")
        return False

def handler(event):
    """
    RunPod 핸들러 함수
    
    Args:
        event: 요청 이벤트 객체
        
    Returns:
        응답 객체
    """
    try:
        # 입력 데이터 추출
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        
        if not text:
            return {
                "status": "error",
                "message": "텍스트가 필요합니다"
            }
        
        # 텍스트 디코딩 (URL 인코딩된 경우)
        try:
            text = unquote(text)
        except:
            pass  # 디코딩 실패 시 원본 사용
        
        # 1. 텍스트 전처리 및 정제
        sentences = preprocess_text(text)
        
        # 2. 각 문장별 감정 분석
        emotion_results = [analyze_emotion(sentence) for sentence in sentences]
        
        # 3-4. 감정 강도에 따른 TTS 모델 선택 및 음성 파일 생성
        start_time = time.time()
        audio_files = []
        
        for i, (sentence, emotion) in enumerate(zip(sentences, emotion_results)):
            output_path = os.path.join(TEMP_DIR, f"audio_{i}.wav")
            
            # 감정 강도에 따라 모델 선택
            success = False
            if emotion["intensity"] >= 0.6:
                # CSM-1B 모델 사용
                success = generate_audio_with_csm(sentence, output_path)
                model_used = "CSM-1B"
            else:
                # gTTS 모델 사용
                success = generate_audio_with_gtts(sentence, output_path)
                model_used = "gTTS"
            
            if success:
                audio_files.append({
                    "path": output_path,
                    "sentence": sentence,
                    "emotion": emotion["emotion"],
                    "intensity": emotion["intensity"],
                    "model": model_used
                })
        
        # 5. 음성 파일 병합
        final_output = os.path.join(TEMP_DIR, "final_output.wav")
        
        if audio_files:
            merge_audio_files([file["path"] for file in audio_files], final_output)
            
            # 6. 파일을 base64로 인코딩하여 반환
            with open(final_output, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "audio": audio_data,
                "format": "wav",
                "processing_time": processing_time,
                "segments": [
                    {
                        "text": file["sentence"],
                        "emotion": file["emotion"],
                        "intensity": file["intensity"],
                        "model": file["model"]
                    } for file in audio_files
                ]
            }
        else:
            return {
                "status": "error",
                "message": "음성 생성에 실패했습니다"
            }
            
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {
            "status": "error",
            "message": str(e),
            "traceback": trace
        }

# 테스트용 코드
if __name__ == "__main__":
    # 테스트용 이벤트
    test_event = {
        "input": {
            "text": "안녕하세요! 오늘은 정말 기분이 좋네요. 새로운 TTS 시스템을 만들어보고 있어요."
        }
    }
    
    # 핸들러 실행
    result = handler(test_event)
    print(json.dumps(result, ensure_ascii=False, indent=2)) 