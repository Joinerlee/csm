# text_preprocessor.py
from typing import List
import re

def preprocess_text(text: str) -> List[str]:
    """
    텍스트를 전처리하고 문장 단위로 분할합니다.
    
    Args:
        text: 처리할 텍스트
        
    Returns:
        문장 리스트
    """
    # 1. 기본 정제 (불필요한 공백 제거)
    text = text.strip()
    
    # 2. 문장 분할 (한국어 문장 구분자: 마침표, 물음표, 느낌표)
    # 정규식을 사용하여 문장 끝으로 간주되는 패턴 찾기
    pattern = r'[.!?]+[\s]*|[\n]+'
    sentences = re.split(pattern, text)
    
    # 빈 문장 제거 및 추가 공백 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 너무 긴 문장 분할 (TTS 모델이 처리하기 쉽도록)
    result = []
    max_length = 200  # 최대 문장 길이
    
    for sentence in sentences:
        if len(sentence) <= max_length:
            result.append(sentence)
        else:
            # 긴 문장을 쉼표나 공백 기준으로 추가 분할
            subparts = re.split(r'[,;]', sentence)
            for part in subparts:
                part = part.strip()
                if part:
                    if len(part) <= max_length:
                        result.append(part)
                    else:
                        # 여전히 긴 경우, 적절한 위치에서 분할
                        words = part.split()
                        current = ""
                        for word in words:
                            if len(current) + len(word) + 1 <= max_length:
                                if current:
                                    current += " " + word
                                else:
                                    current = word
                            else:
                                result.append(current)
                                current = word
                        if current:
                            result.append(current)
    
    return result 