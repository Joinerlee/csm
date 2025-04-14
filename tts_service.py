# tts_service.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import os
from typing import Dict, Any, List
import torch
import shutil
import json

# 자체 모듈 임포트
from text_preprocessor import preprocess_text
from emotional_analysis import analyze_emotion
from audio_processor import merge_audio_files

app = FastAPI(title="RunPod TTS 서비스", description="텍스트를 음성으로 변환하는 RunPod 서비스")

# 임시 디렉토리 생성
TEMP_DIR = tempfile.mkdtemp()

@app.post("/process")
async def process_text(data: Dict[str, Any]):
    """
    텍스트를 음성으로 변환하는 전체 프로세스를 처리합니다.
    
    처리 단계:
    1. 텍스트 전처리 및 정제
    2. 감정 분석
    3. 감정 강도에 따른 TTS 모델 선택 (CSM-1B 또는 gTTS)
    4. 음성 파일 생성
    5. 음성 파일 병합
    6. 최종 음성 파일 반환
    """
    try:
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="텍스트가 필요합니다")
        
        # 1. 텍스트 전처리 및 정제
        sentences = preprocess_text(text)
        
        # 2. 각 문장별 감정 분석
        emotion_results = [analyze_emotion(sentence) for sentence in sentences]
        
        # 3-4. 감정 강도에 따른 TTS 모델 선택 및 음성 파일 생성
        audio_files = []
        for i, (sentence, emotion) in enumerate(zip(sentences, emotion_results)):
            output_path = os.path.join(TEMP_DIR, f"audio_{i}.wav")
            
            # 감정 강도에 따라 모델 선택
            if emotion["intensity"] >= 0.6:
                # CSM-1B 모델 사용
                generate_audio_with_csm(sentence, output_path)
            else:
                # gTTS 모델 사용
                generate_audio_with_gtts(sentence, output_path)
            
            audio_files.append(output_path)
        
        # 5. 음성 파일 병합
        final_output = os.path.join(TEMP_DIR, "final_output.wav")
        merge_audio_files(audio_files, final_output)
        
        # 6. 최종 음성 파일 반환
        return FileResponse(
            path=final_output,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")

def generate_audio_with_csm(text: str, output_path: str):
    """CSM-1B 모델을 사용하여 텍스트를 음성으로 변환합니다."""
    try:
        # CSM-1B 모델 로드 및 음성 생성 로직
        # 실제 구현에서는 generator.py의 load_csm_1b와 generate 함수를 사용
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
        
    except Exception as e:
        raise RuntimeError(f"CSM-1B 모델 처리 중 오류: {str(e)}")

def generate_audio_with_gtts(text: str, output_path: str):
    """Google TTS를 사용하여 텍스트를 음성으로 변환합니다."""
    try:
        from gtts import gTTS
        
        # gTTS로 음성 생성
        tts = gTTS(text=text, lang='ko', slow=False)
        tts.save(output_path)
        
    except Exception as e:
        raise RuntimeError(f"gTTS 처리 중 오류: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 임시 파일 정리"""
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run("tts_service:app", host="0.0.0.0", port=8000, reload=True) 