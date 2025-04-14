import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download, login
import sys
import random
import subprocess
# 직접 모델 사용하지 않도록 변경
import re
from gtts import gTTS
from IPython.display import Audio, display

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
    subprocess.run(["pip", "install", "-q", "torch", "torchaudio", "huggingface_hub", "gtts", "pydub"], check=True)
    
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
    "joy_concert_gift.wav",
    "sad_trip_cancel.wav",
    "angry_lazy_teammate.wav",
    "nervous_interview_eve.wav",
    "touched_by_friend.wav",
    "embarrassed_presentation.wav",
    "dilemma_vacation_choice.wav",
    "emotional_parents_support.wav",
    "surprised_friend_visit.wav",
    "relieved_after_exam.wav",
    "proud_first_dinner.wav",
    "confused_firstday_work.wav",
    "tired_after_overtime.wav",
    "tearful_reunion_friend.wav",
    "disappointed_movie_night.wav",
    "excited_first_date.wav",
    "awkward_failed_joke.wav"
]

# Dictionary to store transcript texts for each wav file
VOICE_TRANSCRIPTS = {
    "joy_concert_gift.wav": "When my friend told me she got us tickets to the concert, I literally screamed. I had wanted to go for so long, and I never expected her to surprise me like that. It was honestly one of the best moments of my year.",
    "sad_trip_cancel.wav": "I was all packed and ready to go when I got the message that our trip was canceled. I just sat there staring at my suitcase—it felt so unreal. I had been looking forward to it for weeks, and I was completely crushed.",
    "angry_lazy_teammate.wav": "One of my teammates literally did nothing the entire project. I ended up doing all the work by myself, and it was incredibly frustrating. I wanted to call them out, but I tried to stay calm and just got it done.",
    "nervous_interview_eve.wav": "The night before my job interview, I couldn't sleep at all. I kept going over my answers in my head, worrying I'd mess something up. When I finally walked into the room, my heart was pounding like crazy.",
    "touched_by_friend.wav": "I was going through a really tough time, and one of my friends stayed by my side the whole way. She didn't try to fix things—she just listened and made me feel seen. I was honestly so touched, and I still think about it often.",
    "embarrassed_presentation.wav": "In the middle of my presentation, I suddenly blanked on the client's name. There was this really awkward silence, and I could feel my face turning red. I laughed it off, but inside I was freaking out.",
    "dilemma_vacation_choice.wav": "We couldn't agree on where to go for our vacation—she wanted the beach, I wanted the mountains. I kept changing my mind because I didn't want to disappoint her. In the end, we compromised, but man, it was so hard to decide.",
    "emotional_parents_support.wav": "When I walked on stage and saw my parents in the audience, I almost cried. They had driven hours just to see me speak. That meant more to me than any award I could've received.",
    "surprised_friend_visit.wav": "I was just chilling at home when someone rang the doorbell. I opened it and saw my best friend standing there with snacks and a big smile. I was totally shocked—but in the best way possible.",
    "relieved_after_exam.wav": "As soon as I walked out of the exam room, I let out a huge sigh of relief. It was finally over, and I felt like a huge weight had been lifted off my shoulders. I could finally relax and breathe again.",
    "proud_first_dinner.wav": "I cooked dinner for my family for the first time, and they loved it. Seeing them smile and enjoy the food made me feel so proud. It wasn't perfect, but I had really given it my all.",
    "confused_firstday_work.wav": "On my first day at the new job, everything felt overwhelming. I kept getting lost in the building and couldn't remember anyone's name. I was smiling, but deep down I was totally confused and anxious.",
    "tired_after_overtime.wav": "I finally got home after a long day of overtime. My body felt like it was made of lead, and even brushing my teeth felt like a chore. I collapsed on my bed and passed out instantly.",
    "tearful_reunion_friend.wav": "When I saw her at the airport after so many years, I couldn't hold back my tears. We just hugged each other without saying a word. It felt like no time had passed at all.",
    "disappointed_movie_night.wav": "I had been waiting months to watch this movie, and it ended up being such a letdown. The plot made no sense, and the characters were flat. I walked out of the theater feeling so disappointed.",
    "excited_first_date.wav": "The night before our first date, I couldn't stop smiling. I kept checking my outfit and replaying our last conversation in my head. I was nervous, but mostly just excited to see him again.",
    "awkward_failed_joke.wav": "I was trying to make a joke, but it totally backfired. Everyone just went quiet, and I wanted to disappear. I smiled awkwardly and changed the subject as fast as I could."
}

# 감정에 따른 음성 파일 매핑
EMOTION_TO_VOICE_MAP = {
    "joy": ["joy_concert_gift.wav", "proud_first_dinner.wav", "excited_first_date.wav"],
    "sadness": ["sad_trip_cancel.wav", "tearful_reunion_friend.wav", "disappointed_movie_night.wav"],
    "anger": ["angry_lazy_teammate.wav"],
    "fear": ["nervous_interview_eve.wav"],
    "surprise": ["surprised_friend_visit.wav"],
    "neutral": ["relieved_after_exam.wav", "dilemma_vacation_choice.wav"],
    "disgust": ["disappointed_movie_night.wav"],
    "embarrassment": ["embarrassed_presentation.wav", "awkward_failed_joke.wav"],
    "confusion": ["confused_firstday_work.wav"],
    "tired": ["tired_after_overtime.wav"],
    "love": ["touched_by_friend.wav", "emotional_parents_support.wav"]
}

# 감정 키워드 사전
EMOTION_KEYWORDS = {
    "joy": ["happy", "joy", "excited", "delighted", "thrilled", "amazing", "wonderful", "favorite", "love", "great", "beautiful", "enjoy", "fun", "lovely", "pleasant", "pleased", "glad", "awesome", "excellent", "fabulous", "fantastic", "perfect", "pretty"],
    "sadness": ["sad", "unhappy", "depressed", "disappointed", "miss", "lost", "alone", "lonely", "sorry", "regret", "upset", "unfortunate", "hurt", "heartbroken", "devastating", "gloomy", "miserable", "hopeless"],
    "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "hate", "terrible", "awful", "horrible", "rage", "outraged", "insulted", "offended", "fed up"],
    "fear": ["scared", "afraid", "nervous", "worried", "anxious", "frightened", "terrified", "panic", "dread", "horrified", "alarmed", "uneasy", "uncomfortable", "stressed"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "suddenly", "wow", "interesting", "interestingly", "incredible", "unbelievable", "remarkable", "extraordinary"],
    "disgust": ["disgusted", "gross", "revolting", "nasty", "horrible", "dirty", "unpleasant", "repulsive", "offensive", "sickening"],
    "love": ["love", "adore", "cherish", "beloved", "dear", "precious", "affection", "caring", "tender", "devoted", "fond", "passion", "treasure", "special"]
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

def analyze_emotion(text):
    """
    텍스트의 감정을 간단한 키워드 기반으로 분석합니다.
    """
    text = text.lower()
    emotion_scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    
    # 각 감정별 키워드 점수 계산
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 1
    
    # 가장 높은 점수의 감정 선택
    max_score = 0
    selected_emotion = "neutral"  # 기본값
    
    for emotion, score in emotion_scores.items():
        if score > max_score:
            max_score = score
            selected_emotion = emotion
    
    print(f"감정 분석 결과: {selected_emotion} (점수: {max_score})")
    return selected_emotion, max_score

def split_text_to_sentences(text):
    """
    텍스트를 문장으로 분리합니다. (마침표 기준)
    """
    # 마침표(.), 느낌표(!), 물음표(?) 기준으로 문장 분리
    sentences = re.split(r'[.!?]+', text)
    
    # 공백 제거 및 빈 문장 필터링
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def find_matching_voice_files(emotion, available_files):
    """
    감정에 맞는 음성 파일을 찾습니다.
    없으면 랜덤하게 선택합니다.
    """
    # 감정에 맞는 음성 파일 목록
    matching_files = []
    
    # 해당 감정의 음성 파일 찾기
    if emotion in EMOTION_TO_VOICE_MAP:
        for voice_file in EMOTION_TO_VOICE_MAP[emotion]:
            if voice_file in available_files:
                matching_files.append(voice_file)
    
    # 매칭되는 파일이 없으면 랜덤하게 선택
    if not matching_files:
        print(f"No matching voice file for emotion '{emotion}'. Selecting randomly.")
        if available_files:
            matching_files = [random.choice(available_files)]
    
    return matching_files

def generate_with_gTTS(text, output_path, lang='en'):
    """
    gTTS를 사용하여 텍스트를 음성으로 변환합니다.
    """
    try:
        # 느린 옵션 사용 안 함 (빠른 음성으로 생성)
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        
        # 음량 정규화와 품질 개선을 위한 후처리 (pydub 사용)
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize
            
            # 파일 불러오기
            audio = AudioSegment.from_file(output_path)
            
            # 음량 정규화 (더 일관된 음량)
            audio = normalize(audio)
            
            # 저장
            audio.export(output_path, format="wav")
        except Exception as e:
            print(f"오디오 후처리 중 경고: {e} (기본 파일이 사용됩니다)")
        
        print(f"Generated (gTTS): {output_path}")
        return True
    except Exception as e:
        print(f"Error generating speech with gTTS: {e}")
        return False

def generate_with_CSM(text, emotion, available_files, device):
    """
    CSM-1B 모델을 사용하여 텍스트를 음성으로 변환합니다.
    """
    try:
        # 감정에 맞는 음성 파일 찾기
        matching_voice_files = find_matching_voice_files(emotion, available_files)
        
        if not matching_voice_files:
            print("No matching voice files found.")
            return None
        
        # 첫 번째 매칭 파일 사용
        source_audio = matching_voice_files[0]
        source_transcript = VOICE_TRANSCRIPTS.get(source_audio, f"This is a transcript for {source_audio}")
        
        # 모델 로드
        generator = load_csm_1b(device)
        
        # 프롬프트 세그먼트 준비
        prompt_segment = prepare_segment(
            text=source_transcript,
            speaker=0,
            audio_path=source_audio,
            sample_rate=generator.sample_rate
        )
        
        # 음성 생성
        audio_tensor = generator.generate(
            text=text,
            speaker=0,
            context=[prompt_segment],
            max_audio_length_ms=30_000,
        )
        
        # 생성된 오디오 저장
        filename_base = os.path.splitext(source_audio)[0]
        output_path = f"generated_{filename_base}_{emotion}.wav"
        
        torchaudio.save(
            output_path,
            audio_tensor.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        print(f"Generated (CSM): {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating speech with CSM: {e}")
        return None

def merge_audio_files(audio_files, output_file):
    """
    여러 오디오 파일을 하나로 합칩니다.
    """
    try:
        from pydub import AudioSegment
        
        # 첫 번째 파일 로드
        try:
            combined = AudioSegment.from_file(audio_files[0], format="wav")
        except Exception as e:
            print(f"첫 번째 파일 로드 중 오류 발생: {e}")
            print(f"개별 파일만 사용합니다.")
            return None
        
        # 나머지 파일 추가
        for audio_file in audio_files[1:]:
            try:
                audio = AudioSegment.from_file(audio_file, format="wav")
                combined += audio
            except Exception as e:
                print(f"파일 '{audio_file}' 병합 중 오류 발생: {e}")
                print(f"이 파일은 건너뜁니다.")
                continue
        
        # 결과 저장
        try:
            combined.export(output_file, format="wav")
            print(f"병합된 오디오가 저장되었습니다: {output_file}")
            return output_file
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
            return None
    
    except Exception as e:
        print(f"오디오 파일 병합 중 오류: {e}")
        return None

def main():
    # 필요한 패키지 설치
    try:
        import pydub
    except ImportError:
        print("Installing pydub for audio merging...")
        subprocess.run(["pip", "install", "pydub"], check=True)
        
    try:
        import gtts
    except ImportError:
        print("Installing gTTS...")
        subprocess.run(["pip", "install", "gtts"], check=True)
    
    # Select the best available device
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
        
        # 여기에 생성하고 싶은 텍스트를 입력하세요
        target_text = input("생성할 텍스트를 입력하세요: ")
        if not target_text:
            target_text = "Hello, this is my voice cloned using CSM-1B. I can say anything you want me to say with this same voice. The technology is really impressive."
        
        # 문장 분리
        sentences = split_text_to_sentences(target_text)
        if not sentences:
            sentences = [target_text]
        
        # 각 문장별로 처리
        output_files = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            output_path = f"sentence_{i}.wav"
            generate_with_gTTS(sentence, output_path)
            output_files.append(output_path)
        
        # 모든 파일 병합
        if len(output_files) > 1:
            final_output = "final_output.wav"
            merged_file = merge_audio_files(output_files, final_output)
            if merged_file:
                print("\n음성 생성 및 병합 완료!")
                print(f"최종 파일: {final_output}")
                
                # Colab에서 다운로드
                if IN_COLAB:
                    try:
                        from google.colab import files
                        print("\n파일 다운로드를 시작합니다...")
                        print(f"병합된 파일 다운로드: {final_output}")
                        files.download(final_output)
                        
                        # 각 문장별 파일도 제공
                        print("\n각 문장별 파일도 다운로드할 수 있습니다:")
                        for file in output_files:
                            print(f"- {file}")
                    except Exception as e:
                        print(f"파일 다운로드 중 오류: {e}")
                        print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                        print("from google.colab import files")
                        print(f"files.download('{final_output}')")
            else:
                print("\n파일 병합에 실패했습니다. 개별 파일을 다운로드해주세요.")
                # 개별 파일 다운로드
                if IN_COLAB:
                    try:
                        from google.colab import files
                        print("\n개별 파일 다운로드:")
                        for file in output_files:
                            print(f"다운로드 중: {file}")
                            files.download(file)
                    except Exception as e:
                        print(f"파일 다운로드 중 오류: {e}")
                        print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                        print("from google.colab import files")
                        for file in output_files:
                            print(f"files.download('{file}')")
        elif len(output_files) == 1:
            final_output = output_files[0]
            print("\n음성 생성 완료!")
            print(f"최종 파일: {final_output}")
            
            # Colab에서 다운로드
            if IN_COLAB:
                try:
                    from google.colab import files
                    print("\n파일 다운로드를 시작합니다...")
                    files.download(final_output)
                except Exception as e:
                    print(f"파일 다운로드 중 오류: {e}")
                    print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                    print("from google.colab import files")
                    print(f"files.download('{final_output}')")
        else:
            print("생성된 오디오 파일이 없습니다.")
            return
    else:
        # 여기에 생성하고 싶은 텍스트를 입력하세요
        target_text = input("생성할 텍스트를 입력하세요: ")
        if not target_text:
            target_text = "I'm really excited about this new technology. It allows us to clone voices and generate natural sounding speech. The possibilities are endless!"
        
        # 문장 분리
        sentences = split_text_to_sentences(target_text)
        if not sentences:
            sentences = [target_text]
        
        print(f"\n문장을 {len(sentences)}개로 분리했습니다.")
        
        # 각 감정별로 사용할 음성 파일을 미리 선택 (고정)
        emotion_to_voice_file = {}
        for emotion in EMOTION_TO_VOICE_MAP:
            matching_files = find_matching_voice_files(emotion, available_voice_files)
            if matching_files:
                # 랜덤으로 선택하지만 한 번 선택하면 고정
                emotion_to_voice_file[emotion] = matching_files[0]
        
        # CSM 모델을 한 번만 로드 (재사용)
        print(f"CSM-1B 모델 로드 중...")
        generator = load_csm_1b(device)
        print(f"모델 로드 완료!")
        
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
            neutral_file = random.choice(available_voice_files)
            transcript = VOICE_TRANSCRIPTS.get(neutral_file, f"This is a transcript for {neutral_file}")
            prompt_segment = prepare_segment(
                text=transcript,
                speaker=0,
                audio_path=neutral_file,
                sample_rate=generator.sample_rate
            )
            emotion_to_prompt["neutral"] = prompt_segment
            print(f"'neutral' 감정용 프롬프트 준비 완료: {neutral_file}")
        
        # 각 문장별로 감정 분석 및 음성 생성
        output_files = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            print(f"\n[{i+1}/{len(sentences)}] 문장: '{sentence}'")
            
            # 감정 분석
            emotion, max_score = analyze_emotion(sentence)
            
            # 감정 점수가 2점 이상일 때만 CSM 사용, 그 외에는 gTTS 사용
            if max_score >= 2 and emotion in emotion_to_prompt:
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
                    output_path = f"sentence_{i}_{emotion}.wav"
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
                    output_path = f"sentence_{i}_fallback.wav"
                    generate_with_gTTS(sentence, output_path)
                    output_files.append(output_path)
            else:
                # 감정 점수가 1점 이하이면 gTTS 사용
                output_path = f"sentence_{i}_neutral.wav"
                generate_with_gTTS(sentence, output_path)
                output_files.append(output_path)
        
        # 모든 파일 병합
        if len(output_files) > 1:
            final_output = "final_output.wav"
            merged_file = merge_audio_files(output_files, final_output)
            if merged_file:
                print("\n음성 생성 및 병합 완료!")
                print(f"최종 파일: {final_output}")
                
                # Colab에서 다운로드
                if IN_COLAB:
                    try:
                        from google.colab import files
                        print("\n파일 다운로드를 시작합니다...")
                        print(f"병합된 파일 다운로드: {final_output}")
                        files.download(final_output)
                        
                        # 각 문장별 파일도 제공
                        print("\n각 문장별 파일도 다운로드할 수 있습니다:")
                        for file in output_files:
                            print(f"- {file}")
                    except Exception as e:
                        print(f"파일 다운로드 중 오류: {e}")
                        print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                        print("from google.colab import files")
                        print(f"files.download('{final_output}')")
            else:
                print("\n파일 병합에 실패했습니다. 개별 파일을 다운로드해주세요.")
                # 개별 파일 다운로드
                if IN_COLAB:
                    try:
                        from google.colab import files
                        print("\n개별 파일 다운로드:")
                        for file in output_files:
                            print(f"다운로드 중: {file}")
                            files.download(file)
                    except Exception as e:
                        print(f"파일 다운로드 중 오류: {e}")
                        print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                        print("from google.colab import files")
                        for file in output_files:
                            print(f"files.download('{file}')")
        elif len(output_files) == 1:
            final_output = output_files[0]
            print("\n음성 생성 완료!")
            print(f"최종 파일: {final_output}")
            
            # Colab에서 다운로드
            if IN_COLAB:
                try:
                    from google.colab import files
                    print("\n파일 다운로드를 시작합니다...")
                    files.download(final_output)
                except Exception as e:
                    print(f"파일 다운로드 중 오류: {e}")
                    print("다음 코드를 실행하여 파일을 다운로드 하세요:")
                    print("from google.colab import files")
                    print(f"files.download('{final_output}')")
        else:
            print("생성된 오디오 파일이 없습니다.")
            return

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
# 실행 시 텍스트를 입력하세요
#
# 주의: 이 모델은 영어로 학습되었으므로 영어 텍스트만 제대로 처리합니다