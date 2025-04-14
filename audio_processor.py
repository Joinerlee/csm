# audio_processor.py
from typing import List
from pydub import AudioSegment
import os

def merge_audio_files(audio_files: List[str], output_path: str) -> str:
    """
    여러 오디오 파일을 하나로 병합합니다.
    
    Args:
        audio_files: 병합할 오디오 파일 경로 목록
        output_path: 출력 파일 경로
        
    Returns:
        병합된 오디오 파일 경로
    """
    if not audio_files:
        raise ValueError("병합할 오디오 파일이 없습니다.")
    
    try:
        # 첫 번째 파일 로드
        combined = AudioSegment.from_file(audio_files[0])
        
        # 나머지 파일들 추가
        for audio_file in audio_files[1:]:
            audio = AudioSegment.from_file(audio_file)
            combined += audio
        
        # 결과 파일 저장
        combined.export(output_path, format="wav")
        
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"오디오 파일 병합 중 오류: {str(e)}") 