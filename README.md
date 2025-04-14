# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

This script will generate a conversation between 2 characters, using a prompt for each character.

```bash
python run_csm.py
```

## Usage

If you want to write your own applications with CSM, the following examples show basic usage.

#### Generate a sentence

This will use a random speaker identity, as no prompt or context is provided.

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

#### Generate with context

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

NOTE: The following example is instructional and the audio files do not exist. It is intended as an example for using context with CSM.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.

# CSM Text-to-Speech Client-Server System

This system allows you to run the CSM text-to-speech model as a backend server with a client application that can send text and receive audio.

## Features

- Backend FastAPI server that processes text-to-speech requests
- Client application with a GUI for sending text and playing audio
- Support for both CSM-1B (high quality) and gTTS (fallback) text-to-speech engines
- Emotion analysis to select the right voice for the content
- Sentence splitting for better speech generation
- Korean language interface for the client

## Prerequisites

- Python 3.8 or higher
- pip package manager
- For CSM model: GPU with CUDA support (recommended) or powerful CPU

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the voice sample files for the CSM model (if you want to use CSM):

```bash
python start.py --download-samples
```

## Usage

### Quick Start

The easiest way to start both the server and client:

```bash
python start.py
```

### Starting Only the Backend Server

1. Start the server:

```bash
python start.py --server-only
```

Or directly:

```bash
python server.py
```

The server will run on `http://localhost:5000` by default. It will automatically load the CSM model (if available) or fall back to gTTS.

### Starting Only the Client Application

1. Start the client:

```bash
python start.py --client-only
```

Or directly:

```bash
python client.py
```

2. In the client GUI:
   - Enter the server URL (default: http://localhost:5000)
   - Type your text in the text input area
   - Select the language (English or Korean)
   - Use the "gTTS" checkbox to use gTTS for faster (but lower quality) speech synthesis
   - Click "음성 생성" (Generate Speech) to send the text to the server and generate speech
   - The audio will play automatically, or you can click "최근 음성 재생" (Play Last Speech) to replay it

## API Endpoints

The backend server provides the following API endpoint:

### `/api/tts` (POST)

Generate speech from text.

Request body:
```json
{
  "text": "Text to convert to speech",
  "format": "wav" or "base64",
  "use_gtts": true or false
}
```

Response (when format is "base64"):
```json
{
  "success": true,
  "data": "base64_encoded_audio_data",
  "format": "base64"
}
```

## Troubleshooting

- If the server fails to start, check if all dependencies are installed correctly
- If CSM model fails to load, ensure you have the required voice sample files
- If the client can't connect to the server, check if the server is running and the URL is correct
- For better Korean TTS, use gTTS option as CSM is primarily trained on English

## License

This project uses code from the CSM repository, which is licensed under [MIT License](https://github.com/SesameAILabs/csm/blob/main/LICENSE).

# CSM-1B API 서버

CSM-1B 텍스트음성변환(TTS) 모델을 API 서버로 제공하는 코드입니다. 이 서버는 ngrok을 통해 외부에서 접근 가능합니다.

## 주요 특징

- FastAPI를 사용한 고성능 API 서버
- CSM-1B 모델로 고품질 영어 음성 생성
- gTTS로 한국어 음성 생성 지원
- ngrok을 통한 외부 접근 지원
- 감정 분석 기반 음성 생성
- 문장 간 0.5초 무음 삽입으로 자연스러운 음성 생성

## 필수 요구사항

- Python 3.8+
- PyTorch (CUDA 지원 권장)
- torchaudio
- FastAPI, Uvicorn
- pyngrok
- nest_asyncio

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. CSM-1B 모델 설치 (Hugging Face):

```bash
git clone https://github.com/SesameAILabs/csm.git
cd csm
pip install -e .
```

3. 음성 샘플 파일 다운로드 (CSM 음성 생성에 필요):

```bash
python download_samples.py
```

## 서버 실행

```bash
python csm_server.py
```

실행하면 다음과 같은 출력이 나타납니다:

```
✅ CSM-1B 모델 로딩 중...
✅ CSM-1B 모델 로드 완료.
'joy' 감정용 프롬프트 준비 완료: joy_concert_gift.wav
...
🔗 공개된 FastAPI URL: https://xxxx-xx-xx-xxx-xx.ngrok-free.app
```

ngrok URL을 통해 외부에서 API에 접근할 수 있습니다.

## API 사용법

### 엔드포인트: `/csm-1b`

**메서드**: POST

**요청 형식**:
```json
{
  "text": "변환할 텍스트를 입력하세요. 여러 문장으로 구성할 수 있습니다."
}
```

**응답**: 
- `.wav` 형식의 오디오 바이너리 데이터
- 여러 문장은 0.5초 무음으로 구분됨

## 클라이언트 사용법

테스트 클라이언트를 사용하여 API를 호출할 수 있습니다:

```bash
python csm_client.py --url "https://your-ngrok-url" --text "변환할 텍스트"
```

텍스트 파일을 사용:

```bash
python csm_client.py --url "https://your-ngrok-url" --file "input.txt"
```

대화형 모드로 여러 줄 입력:

```bash
python csm_client.py --url "https://your-ngrok-url"
# 여러 줄 텍스트 입력 후 빈 줄 입력하면 종료
```

저장 디렉토리 지정:

```bash
python csm_client.py --url "https://your-ngrok-url" --text "변환할 텍스트" --save-dir "./outputs"
```

## 음성 생성 로직

1. 텍스트 분석 및 처리
   - 문장 단위로 분리
   - 감정 분석 (키워드 기반)
   - 한국어 텍스트 감지

2. 음성 생성 방식
   - 영어 텍스트 + 높은 감정 점수: CSM-1B 사용
   - 영어 텍스트 + 낮은 감정 점수: gTTS 사용
   - 한국어 텍스트: gTTS 사용 (CSM-1B는 영어 전용)

3. 후처리
   - 문장 간 0.5초 무음 삽입
   - 오디오 파일 병합

## 주의사항

- CSM-1B는 영어로 학습된 모델이므로 영어 텍스트만 제대로 처리합니다.
- 한국어 텍스트는 자동으로 gTTS를 사용하여 처리됩니다.
- ngrok의 무료 계정은 연결 시간에 제한이 있습니다.
- 긴 텍스트의 경우 처리 시간이 길어질 수 있습니다.
- 커스텀 도메인을 사용하려면 ngrok 계정 설정이 필요합니다.
