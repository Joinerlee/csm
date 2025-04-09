import os
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import io
import base64
from typing import Optional, List
import tempfile
import uuid

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

# CSM 모델 임포트
from generator import load_csm_1b, Segment
from dataclasses import dataclass

app = FastAPI()

# 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
generator = load_csm_1b(device)

# 프롬프트 준비
def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

# 프롬프트 세그먼트 준비
prompt_a = prepare_prompt(
    SPEAKER_PROMPTS["conversational_a"]["text"],
    0,
    SPEAKER_PROMPTS["conversational_a"]["audio"],
    generator.sample_rate
)

prompt_b = prepare_prompt(
    SPEAKER_PROMPTS["conversational_b"]["text"],
    1,
    SPEAKER_PROMPTS["conversational_b"]["audio"],
    generator.sample_rate
)

prompt_segments = [prompt_a, prompt_b]

class SpeechRequest(BaseModel):
    text: str
    speaker: Optional[int] = 0

@app.get("/")
async def root():
    return {"status": "CSM API is running"}

@app.post("/generate-speech/")
async def generate_speech(request: SpeechRequest):
    try:
        # 텍스트를 문장으로 분리
        sentences = [s.strip() for s in request.text.split(".") if s.strip()]
        
        # 각 문장에 대한 음성 생성
        generated_segments = []
        
        for sentence in sentences:
            print(f"Generating: {sentence}")
            audio_tensor = generator.generate(
                text=sentence,
                speaker=request.speaker,
                context=prompt_segments + generated_segments,
                max_audio_length_ms=10_000,
            )
            generated_segments.append(Segment(text=sentence, speaker=request.speaker, audio=audio_tensor))
            
            # 1초 묵음 추가
            silence = torch.zeros(generator.sample_rate)  # 1초 묵음
            silence_segment = Segment(text="", speaker=request.speaker, audio=silence)
            generated_segments.append(silence_segment)
        
        # 모든 세그먼트 결합
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        
        # 임시 WAV 파일로 저장
        temp_filename = f"temp_{uuid.uuid4()}.wav"
        torchaudio.save(
            temp_filename,
            all_audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        # WAV 파일을 바이트로 읽기
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
        
        # 임시 파일 삭제
        os.remove(temp_filename)
        
        # Base64 인코딩
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return {"audio": audio_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 