from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import torch
import torchaudio
from io import BytesIO
import random
import base64
import importlib.util
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import the run_csm functions directly
from run_csm import (
    load_csm_1b, 
    prepare_segment, 
    analyze_emotion,
    split_text_to_sentences,
    find_matching_voice_files,
    generate_with_gTTS,
    merge_audio_files,
    VOICE_CLONE_FILES,
    VOICE_TRANSCRIPTS
)

app = FastAPI(title="CSM TTS API", description="Text-to-Speech API using CSM or gTTS")

# Global variables
generator = None
device = None
available_voice_files = []
emotion_to_prompt = {}

# Define request model
class TTSRequest(BaseModel):
    text: str
    format: str = "wav"  # Default to wav if not specified
    use_gtts: bool = False  # Use CSM by default if available
    language: str = "en"  # Default language is English, but can be 'ko' for Korean

@app.on_event("startup")
async def startup_event():
    global generator, device, available_voice_files, emotion_to_prompt
    
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
    
    # Load CSM model if voice files are available
    if available_voice_files:
        try:
            print("Loading CSM-1B model...")
            generator = load_csm_1b(device)
            print("Model loading complete!")
            
            # Prepare prompt segments for each emotion
            for emotion, voice_files in emotion_to_voice_map().items():
                matching_files = find_matching_voice_files(emotion, available_voice_files)
                if matching_files:
                    voice_file = matching_files[0]
                    transcript = VOICE_TRANSCRIPTS.get(voice_file, f"This is a transcript for {voice_file}")
                    prompt_segment = prepare_segment(
                        text=transcript,
                        speaker=0,
                        audio_path=voice_file,
                        sample_rate=generator.sample_rate
                    )
                    emotion_to_prompt[emotion] = prompt_segment
                    print(f"'{emotion}' emotion prompt ready: {voice_file}")
        except Exception as e:
            print(f"Error loading CSM model: {e}")
            generator = None

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    global generator, device, available_voice_files, emotion_to_prompt
    
    try:
        # Split text into sentences
        sentences = split_text_to_sentences(request.text)
        if not sentences:
            sentences = [request.text]
        
        output_files = []
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            print(f"Processing sentence {i+1}/{len(sentences)}: '{sentence}'")
            
            # Analyze emotion
            emotion, max_score = analyze_emotion(sentence)
            
            output_path = f"temp_sentence_{i}_{emotion}.wav"
            
            # For Korean text, use gTTS regardless of other settings
            use_gtts_for_this = request.use_gtts
            
            # If language is Korean, or contains Korean characters, force using gTTS
            if request.language == "ko" or any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in sentence):
                use_gtts_for_this = True
                print(f"Korean text detected, using gTTS")
            
            # Use CSM if available and emotion is strong enough
            if generator is not None and max_score >= 2 and emotion in emotion_to_prompt and not use_gtts_for_this:
                try:
                    # Use prepared prompt
                    prompt_segment = emotion_to_prompt[emotion]
                    
                    # Generate audio
                    audio_tensor = generator.generate(
                        text=sentence,
                        speaker=0,
                        context=[prompt_segment],
                        max_audio_length_ms=30_000,
                    )
                    
                    # Save audio
                    torchaudio.save(
                        output_path,
                        audio_tensor.unsqueeze(0).cpu(),
                        generator.sample_rate
                    )
                    print(f"Generated with CSM: {output_path}")
                    output_files.append(output_path)
                except Exception as e:
                    print(f"CSM generation failed: {e}")
                    # Fall back to gTTS
                    lang = "ko" if request.language == "ko" or any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in sentence) else "en"
                    generate_with_gTTS(sentence, output_path, lang)
                    output_files.append(output_path)
            else:
                # Use gTTS
                lang = "ko" if request.language == "ko" or any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in sentence) else "en"
                generate_with_gTTS(sentence, output_path, lang)
                output_files.append(output_path)
        
        # Merge audio files if needed
        if len(output_files) > 1:
            final_output = "temp_final_output.wav"
            merged_file = merge_audio_files(output_files, final_output)
            if merged_file:
                audio_file = merged_file
            else:
                # If merging fails, return the first file
                audio_file = output_files[0]
        elif len(output_files) == 1:
            audio_file = output_files[0]
        else:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Return the audio file
        if request.format == 'base64':
            # Convert to base64
            with open(audio_file, 'rb') as file:
                audio_data = file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary files
            for file in output_files:
                if os.path.exists(file):
                    os.remove(file)
            if len(output_files) > 1 and os.path.exists(final_output):
                os.remove(final_output)
                
            return {
                "success": True,
                "data": audio_base64,
                "format": "base64"
            }
        else:
            # Send file directly
            return FileResponse(
                path=audio_file, 
                media_type="audio/wav", 
                filename="generated_speech.wav"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def emotion_to_voice_map():
    """Return the emotion to voice file mapping"""
    return {
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

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000) 