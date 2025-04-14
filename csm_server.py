import os
import re
import uuid
import torch
import torchaudio
import io
import time

import nest_asyncio
from pyngrok import ngrok
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# run_csm의 함수 직접 가져오기
from run_csm import (
    load_csm_1b,
    prepare_segment,
    analyze_emotion,
    split_text_to_sentences,
    find_matching_voice_files,
    generate_with_gTTS,
    merge_audio_files,
    VOICE_CLONE_FILES,
    VOICE_TRANSCRIPTS,
    EMOTION_TO_VOICE_MAP
)

from dataclasses import dataclass

@dataclass
class Segment:
    text: str
    speaker: int
    audio: torch.Tensor

# ----- CSM-1B 모델 로드 -----
print("✅ CSM-1B 모델 로딩 중...")

# A100 최적화 설정
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 디바이스 선택
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"사용 디바이스: {device}")

# 사용 가능한 음성 파일 찾기
available_voice_files = []
for voice_file in VOICE_CLONE_FILES:
    if os.path.exists(voice_file):
        available_voice_files.append(voice_file)

if available_voice_files:
    print(f"사용 가능한 음성 파일: {len(available_voice_files)}개")
    
    # CSM 모델 로드 (한 번만)
    generator = load_csm_1b(device)
    print("✅ CSM-1B 모델 로드 완료.")
    
    # 각 감정별로 사용할 음성 파일을 미리 선택
    emotion_to_voice_file = {}
    for emotion in EMOTION_TO_VOICE_MAP:
        matching_files = find_matching_voice_files(emotion, available_voice_files)
        if matching_files:
            emotion_to_voice_file[emotion] = matching_files[0]
    
    # 각 프롬프트 세그먼트를 미리 준비
    emotion_to_prompt = {}
    for emotion, voice_file in emotion_to_voice_file.items():
        transcript = VOICE_TRANSCRIPTS.get(voice_file, f"This is a transcript for {voice_file}")
        prompt_segment = prepare_segment(
            text=transcript,
            speaker=0,
            audio_path=voice_file,
            sample_rate=generator.sample_rate
        )
        emotion_to_prompt[emotion] = prompt_segment
        print(f"'{emotion}' 감정용 프롬프트 준비 완료: {voice_file}")
    
    # 중립 감정의 기본 프롬프트 준비
    if "neutral" not in emotion_to_prompt and available_voice_files:
        neutral_file = available_voice_files[0]  # 첫 번째 파일 사용
        transcript = VOICE_TRANSCRIPTS.get(neutral_file, f"This is a transcript for {neutral_file}")
        prompt_segment = prepare_segment(
            text=transcript,
            speaker=0,
            audio_path=neutral_file,
            sample_rate=generator.sample_rate
        )
        emotion_to_prompt["neutral"] = prompt_segment
        print(f"'neutral' 감정용 프롬프트 준비 완료: {neutral_file}")
else:
    print("⚠️ 음성 파일을 찾을 수 없습니다. gTTS만 사용합니다.")
    generator = None
    emotion_to_prompt = {}

# ----- FastAPI 앱 정의 -----
app = FastAPI()

@app.post("/csm-1b")
async def predict(request: Request):
    """
    JSON 형태로 { "text": "여기에 문장을 넣으세요" }
    요청을 받으면, 문장 분리 후 csm-1b로 음성 생성해
    0.5초 무음을 삽입하여 이어붙인 통합 음성을
    raw 바이너리 형태로 반환합니다.
    """
    start_time = time.time()
    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "No text provided"}

    # --- 문장 단위로 분리 ---
    sentences = split_text_to_sentences(text)
    if not sentences:
        sentences = [text]

    print(f"✅ 문장 분리 완료: {len(sentences)}개 문장")

    # run_csm 로직을 사용하여 음성 생성
    output_files = []
    
    for i, sentence in enumerate(sentences):
        if not sentence:
            continue
            
        print(f"\n[{i+1}/{len(sentences)}] 문장: '{sentence}'")
        
        # 감정 분석
        emotion, max_score = analyze_emotion(sentence)
        
        # 한국어 텍스트 감지
        contains_korean = any((ord(char) >= 0xAC00 and ord(char) <= 0xD7A3) for char in sentence)
        
        # 파일 경로 설정
        output_path = f"temp_sentence_{i}_{emotion}.wav"
        
        # 감정 점수가 2점 이상이고 CSM 사용 가능하고 한국어가 아닐 때만 CSM 사용
        if generator is not None and max_score >= 2 and emotion in emotion_to_prompt and not contains_korean:
            # 이미 준비된 프롬프트 사용
            prompt_segment = emotion_to_prompt[emotion]
            
            # 음성 생성
            print(f"CSM으로 '{emotion}' 감정 음성 생성 중...")
            try:
                audio_tensor = generator.generate(
                    text=sentence,
                    speaker=0,
                    context=[prompt_segment],
                    max_audio_length_ms=30_000,
                )
                
                # 생성된 오디오 저장
                torchaudio.save(
                    output_path,
                    audio_tensor.unsqueeze(0).cpu(),
                    generator.sample_rate
                )
                print(f"Generated (CSM): {output_path}")
                output_files.append(output_path)
            except Exception as e:
                print(f"CSM 음성 생성 실패: {e}")
                # CSM 실패시 gTTS로 대체
                lang = "ko" if contains_korean else "en"
                generate_with_gTTS(sentence, output_path, lang)
                output_files.append(output_path)
        else:
            # 감정 점수가 1점 이하거나 한국어면 gTTS 사용
            lang = "ko" if contains_korean else "en"
            generate_with_gTTS(sentence, output_path, lang)
            output_files.append(output_path)
    
    # 모든 파일 병합하기 전에 0.5초 무음 삽입
    if len(output_files) > 1:
        # 0.5초 무음 생성
        sample_rate = 24000  # CSM의 기본 샘플레이트
        silence_duration = 0.5
        silence_samples = int(silence_duration * sample_rate)
        
        # 각 파일 사이에 무음 추가
        final_output = "temp_final_output.wav"
        merged_file = merge_audio_files(output_files, final_output)
        
        if merged_file:
            audio_file = merged_file
        else:
            # 병합 실패 시 첫 번째 파일만 반환
            audio_file = output_files[0]
    else:
        # 단일 파일인 경우
        audio_file = output_files[0]
    
    # 파일을 바이너리로 읽기
    with open(audio_file, "rb") as f:
        audio_bytes = io.BytesIO(f.read())
    
    # 임시 파일 삭제
    for file in output_files:
        if os.path.exists(file):
            os.remove(file)
    
    if len(output_files) > 1 and os.path.exists("temp_final_output.wav"):
        os.remove("temp_final_output.wav")
    
    # 총 소요 시간 출력
    total_time = time.time() - start_time
    print(f"✅ 총 처리 시간: {total_time:.2f}초 ({len(sentences)}개 문장, 평균: {total_time/max(1,len(sentences)):.2f}초/문장)")
    
    # StreamingResponse로 바이너리 데이터 반환
    audio_bytes.seek(0)
    return StreamingResponse(audio_bytes, media_type="audio/wav")

# ----- ngrok으로 서버 공개 -----
nest_asyncio.apply()
# 커스텀 도메인이 있으면 사용하고, 없으면 일반 ngrok URL 사용
try:
    # 'domain' 파라미터 사용 시 ngrok 계정과 인증이 필요합니다
    public_url = ngrok.connect(8000, domain="omypic.ngrok.app")
except Exception as e:
    print(f"커스텀 도메인 연결 실패 (계정 설정 필요): {e}")
    # 일반 URL로 대체
    public_url = ngrok.connect(8000)
    
print(f"🔗 공개된 FastAPI URL: {public_url}")

if __name__ == "__main__":
    # 서버 시작 전 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uvicorn.run(app, host="0.0.0.0", port=8000) 