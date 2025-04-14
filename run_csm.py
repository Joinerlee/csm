import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download, login
import sys
import random
import subprocess

# Disable Triton compilation (필수 설정)
os.environ["NO_TORCH_COMPILE"] = "1"

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab environment")
    
    # Function to set up Hugging Face token
    def setup_huggingface_token():
        from google.colab import output
        from IPython.display import display, HTML
        import getpass
        
        print("Please enter your Hugging Face token to access models:")
        print("(Get a token from https://huggingface.co/settings/tokens)")
        token = getpass.getpass("HF Token: ")
        if token:
            login(token=token)
            print("Successfully logged in to Hugging Face!")
            return True
        else:
            print("No token provided. Models may not be accessible.")
            return False
    
    # Set up HF token at the beginning
    token_setup = setup_huggingface_token()
    
    # Install required packages if running in Colab
    subprocess.run(["pip", "install", "-q", "torch", "torchaudio", "huggingface_hub", "transformers"], check=True)
    
    # Add current directory to path for imports
    if os.path.exists("generator.py"):
        print("Found generator.py in current directory")
    else:
        print("Downloading required model files...")
        subprocess.run(["git", "clone", "https://github.com/SesameAILabs/csm.git"], check=True)
        os.chdir("csm")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        
except ImportError:
    IN_COLAB = False
    print("Not running in Google Colab environment")
    # Make sure to login to huggingface if not in Colab
    # You can uncomment and run this manually:
    # login()

from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define list of available wav files for voice cloning
VOICE_CLONE_FILES = [
    "awkward_failed_joke.wav",
    "excited_first_date.wav",
    "disappointed_movie_night.wav", 
    "confused_firstday_work.wav",
    "emotional_parents_support.wav",
    "dilemma_vacation_choice.wav",
    "embarrassed_presentation.wav",
    "angry_lazy_teammate.wav"
]

# Dictionary to store transcript texts for each wav file
# These should be replaced with accurate transcripts for each audio file
VOICE_TRANSCRIPTS = {
    "awkward_failed_joke.wav": "I was trying to make a joke, but it totally backfired. Everyone just went quiet, and I wanted to disappear. I smiled awkwardly and changed the subject as fast as I could.",
    "excited_first_date.wav": "The night before our first date, I couldn't stop smiling. I kept checking my outfit and replaying our last conversation in my head. I was nervous, but mostly just excited to see him again.",
    "disappointed_movie_night.wav": "I had been waiting months to watch this movie, and it ended up being such a letdown. The plot made no sense, and the characters were flat. I walked out of the theater feeling so disappointed.",
    "confused_firstday_work.wav": "On my first day at the new job, everything felt overwhelming. I kept getting lost in the building and couldn't remember anyone's name. I was smiling, but deep down I was totally confused and anxious.",
    "emotional_parents_support.wav": "When I walked on stage and saw my parents in the audience, I almost cried. They had driven hours just to see me speak. That meant more to me than any award I could've received.",
    "dilemma_vacation_choice.wav": "I can't decide if I should go to the mountains or the beach for vacation. They both sound so appealing for different reasons.",
    "embarrassed_presentation.wav": "In the middle of my presentation to the whole company, I completely blanked out. I couldn't remember any of my talking points.",
    "angry_lazy_teammate.wav": "I've been doing all the work on this group project while my teammate hasn't contributed anything. It's really frustrating."
}

# Default fallback prompt if local files not available
DEFAULT_PROMPT = {
    "filepath": hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    ),
    "text": (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    )
}

def load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """Load audio from file and resample to target sample rate"""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample if needed
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_segment(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    """Create a segment from text, speaker and audio file"""
    audio_tensor = load_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def clone_voice(source_audio_path, source_transcript, target_text, speaker_id=0, device="cpu"):
    """
    Clone a voice from a source audio file to generate speech with the target text
    
    Args:
        source_audio_path: Path to the source audio file
        source_transcript: Transcript of the source audio
        target_text: Text to generate with the cloned voice
        speaker_id: Speaker ID for the voice (default 0)
        device: Device to run the model on
    
    Returns:
        Path to the generated audio file
    """
    print(f"Loading CSM-1B model (this may take a minute)...")
    generator = load_csm_1b(device)
    print(f"Model loaded successfully!")
    
    # Prepare the prompt segment
    prompt_segment = prepare_segment(
        text=source_transcript,
        speaker=speaker_id,
        audio_path=source_audio_path,
        sample_rate=generator.sample_rate
    )
    
    # Generate speech with the target text
    print(f"Generating speech for: {target_text}")
    audio_tensor = generator.generate(
        text=target_text,
        speaker=speaker_id,
        context=[prompt_segment],
        max_audio_length_ms=30_000,  # increased for longer texts
    )
    
    # Save the generated audio
    output_filename = "generated_speech.wav"
    torchaudio.save(
        output_filename,
        audio_tensor.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print(f"Successfully generated {output_filename}")
    
    return output_filename

def main():
    # Select the best available device based on official README
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Find available voice clone source files
    available_voice_files = []
    for voice_file in VOICE_CLONE_FILES:
        if os.path.exists(voice_file):
            available_voice_files.append(voice_file)
    
    if not available_voice_files:
        print("No voice files found. Using default prompt.")
        source_audio = DEFAULT_PROMPT["filepath"]
        source_transcript = DEFAULT_PROMPT["text"]
    else:
        # Use the first available file for voice cloning
        source_audio = available_voice_files[0]
        source_transcript = VOICE_TRANSCRIPTS.get(source_audio, f"This is a transcript for {source_audio}")
        print(f"Using {source_audio} for voice cloning")
    
    # 여기에 생성하고 싶은 텍스트를 입력하세요 (Enter the text you want to generate)
    target_text = "Hello, this is my voice cloned using CSM-1B. I can say anything you want me to say with this same voice. The technology is really impressive."
    
    # Clone the voice to generate the target text
    output_file = clone_voice(
        source_audio_path=source_audio,
        source_transcript=source_transcript,
        target_text=target_text,
        speaker_id=0,  # Fixed speaker ID for consistency
        device=device
    )
    
    if IN_COLAB:
        from google.colab import files
        print("파일 다운로드를 시작합니다... (Starting file download...)")
        files.download(output_file)

if __name__ == "__main__":
    main()

# 사용 방법:
# 1. 필요한 패키지 설치: pip install -r requirements.txt
# 2. Hugging Face 토큰 설정: huggingface-cli login (또는 코드 내에서 login() 함수 사용)
# 3. 이 스크립트 실행: python run_csm.py
#
# 코랩에서 사용하는 방법:
# 1. 구글 코랩에서 새 노트북을 만들고 이 스크립트를 셀에 복사하세요
# 2. 실행 시 Hugging Face 토큰을 입력하세요
# 3. 자동으로 필요한 패키지를 설치하고 음성을 생성합니다
#
# 원하는 텍스트로 변경하려면:
# main() 함수 안에 있는 'target_text' 변수를 수정하세요
#
# 보이스 클론 소스 파일:
# 스크립트는 디렉토리에서 첫 번째로 찾은 .wav 파일을 사용해 음성 특성을 학습합니다
# 원하는 음성의 .wav 파일을 추가하고, 정확한 트랜스크립트를 VOICE_TRANSCRIPTS에 추가하면 됩니다
#
# 주의: 이 모델은 영어로 학습되었으므로 영어 텍스트만 제대로 처리합니다