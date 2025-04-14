# emotional_analysis.py
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# EmotionalBERT 모델 로드 (처음 사용 시 다운로드)
# 실제 사용 모델은 한국어 감정 분석에 적합한 모델로 대체해야 함
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"  # 예시 모델

# 글로벌 변수로 모델과 토크나이저 정의
tokenizer = None
model = None

def load_model():
    """모델과 토크나이저를 로드합니다."""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            
            # GPU 사용 가능하면 GPU로 이동
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()  # 평가 모드 설정
        except Exception as e:
            raise RuntimeError(f"감정 분석 모델 로드 중 오류: {str(e)}")

def analyze_emotion(text: str) -> Dict[str, Any]:
    """
    텍스트의 감정을 분석합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        감정 분석 결과를 담은 딕셔너리
        {
            "emotion": "happy/sad/neutral/...",
            "intensity": 0.0-1.0 (감정 강도),
            "scores": {...} (각 감정 클래스별 점수)
        }
    """
    # 모델 로드
    load_model()
    
    try:
        # 토큰화
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # 가장 높은 확률의 감정 클래스와 강도
        label_id = probs.argmax()
        intensity = float(probs[label_id])
        
        # 감정 레이블 매핑 (모델에 따라 다를 수 있음)
        emotion_labels = {
            0: "negative",
            1: "positive"
            # 실제 모델에 맞게 감정 레이블 추가
        }
        
        return {
            "emotion": emotion_labels.get(label_id, "unknown"),
            "intensity": intensity,
            "scores": {emotion: float(probs[i]) for i, emotion in emotion_labels.items()}
        }
    
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return {
            "emotion": "neutral",
            "intensity": 0.5,
            "scores": {"neutral": 1.0}
        } 