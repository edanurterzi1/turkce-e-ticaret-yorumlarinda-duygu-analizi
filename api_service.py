"""
Basit FastAPI servisi: Eğittiğiniz duygu analizi modelini (olumsuz/olumlu)
REST API olarak sunar.

Çalıştırma örneği (Windows, mevcut sanal ortamla):
1) venv_new\Scripts\activate
2) python -m uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload
3) http://localhost:8000/docs


Notlar:
- Varsayılan model klasörü: CONVBERTurk_mC4_uncased2_20251221_1609
  İsterseniz MODEL_DIR ortam değişkeniyle farklı bir klasör verebilirsiniz.
- CORS açık; gerekirse Origins listesini daraltın.
"""
import os
from typing import List, Literal,Annotated

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist,Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------
# Yapılandırma
# ---------------------------
DEFAULT_MODEL_DIR = "CONVBERTurk_mC4_uncased2_20251221_1609"
MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
LABEL_MAP = {0: "olumsuz", 1: "olumlu"}
MAX_SEQ_LEN = 53  # inference sırasında güvenli bir üst sınır

# ---------------------------
# Model + Tokenizer yükleme
# ---------------------------
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_once():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"MODEL_DIR bulunamadı: {MODEL_DIR}. Ortam değişkenini kontrol edin."
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()


# ---------------------------
# FastAPI tanımı
# ---------------------------
app = FastAPI(
    title="Türkçe Duygu Analizi API",
    description="Olumsuz / olumlu sınıflandırma servisi.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ihtiyaca göre daraltın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# İstek/yanıt şemaları
# ---------------------------
class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: Annotated[
        List[str], 
        Field(min_length=1, max_length=256) # min_items -> min_length oldu
    ]


class PredictResponse(BaseModel):
    label: Literal["olumsuz", "olumlu"]
    score: float


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]


# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def _predict_internal(texts: List[str]) -> List[PredictResponse]:
    _load_model_once()

    # Eğitim sırasında uyguladığımız metin temizleme işlemini (clean_text) modele girmeden önce burada da uygulamalıyız.
    from src.data_preprocessing import clean_text
    cleaned_texts = [clean_text(t) for t in texts]

    encoded = _tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        outputs = _model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)
        scores, preds = torch.max(probs, dim=1)

    responses: List[PredictResponse] = []
    for score, pred in zip(scores.tolist(), preds.tolist()):
        label = LABEL_MAP.get(int(pred), "bilinmiyor")
        responses.append(PredictResponse(label=label, score=float(score)))
    return responses


# ---------------------------
# Endpointler
# ---------------------------
@app.get("/health")
def health():
    try:
        _load_model_once()
        return {"status": "ok", "device": str(_device), "model_dir": MODEL_DIR}
    except Exception as exc:  # pragma: no cover - sadece servis kontrolü
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = _predict_internal([req.text])[0]
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    try:
        results = _predict_internal(req.texts)
        return PredictBatchResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root():
    return {
        "message": "Türkçe duygu analizi servisi hazır.",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "predict_batch": "POST /predict/batch",
        "model_dir": MODEL_DIR,
    }

