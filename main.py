"""
SmartFit AI Backend API
FastAPI ile oluşturulmuş RESTful API
Hibrit kalori tahmin motoru (ML + Fuzzy Logic)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from hybrid_engine import SmartFitHybrid
import uvicorn

# ===== FASTAPI UYGULAMASI =====
app = FastAPI(
    title="KaloriBul API",
    description="Akıllı Kalori Tahmin Motoru - Hibrit AI (ML + Fuzzy Logic)",
    version="1.0.0"
)

# ===== CORS AYARLARI =====
# Next.js frontend'inin sorunsuz bağlanabilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm origin'lere izin ver
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
    allow_headers=["*"],  # Tüm header'lara izin ver
)

# ===== GLOBAL MODEL (Uygulama Başlangıcında Bir Kere Eğit) =====
print("=" * 70)
print("🚀 KaloriBul Backend Başlatılıyor...")
print("=" * 70)

# SmartFitHybrid motorunu başlat
engine = SmartFitHybrid()

# Veriyi hazırla
print("\n📊 Veri hazırlanıyor...")
engine.prepare_data()

# ML modelini eğit
print("🤖 ML modeli eğitiliyor...")
engine.train_ml_model()

# Fuzzy sistemini kur
print("🧠 Fuzzy sistemi kuruluyor...")
engine.setup_fuzzy_model()

# Fuzzy yorgunluk sistemini kur
print("💪 Fuzzy yorgunluk sistemi kuruluyor...")
engine.setup_fatigue_fuzzy()

print("\n" + "=" * 70)
print("✅ Backend hazır! API istekleri kabul ediliyor.")
print("=" * 70 + "\n")


# ===== PYDANTIC VERI MODELLERİ =====

class FitnessRequest(BaseModel):
    """
    Fitness tahmin isteği için veri modeli
    """
    weight: float = Field(
        ..., 
        description="Kullanıcının kilosu (kg)",
        gt=0,  # 0'dan büyük olmalı
        le=300,  # 300'den küçük veya eşit olmalı
        example=75.0
    )
    mets: float = Field(
        ..., 
        description="Aktivite MET değeri",
        gt=0,
        le=20,
        example=10.0
    )
    duration: float = Field(
        ..., 
        description="Egzersiz süresi (dakika)",
        gt=0,
        le=300,
        example=45.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "weight": 75.0,
                "mets": 10.0,
                "duration": 45.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Tahmin sonucu için response modeli
    """
    ml_calories: float = Field(..., description="Makine Öğrenimi tahmini (kcal)")
    fuzzy_calories: float = Field(..., description="Fuzzy Logic tahmini (kcal)")
    hybrid_calories: float = Field(..., description="Hibrit tahmin (kcal)")
    ml_fatigue: float = Field(..., description="ML yorgunluk tahmini")
    fuzzy_fatigue: float = Field(..., description="Fuzzy yorgunluk tahmini")
    hybrid_fatigue: float = Field(..., description="Hibrit yorgunluk tahmini")
    fatigue_level: str = Field(..., description="Yorgunluk seviyesi")
    ml_hourly: float = Field(..., description="Saatlik ML tahmini (kcal/saat)")
    burn_factor: float = Field(..., description="Fuzzy burn faktörü")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ml_calories": 605.25,
                "fuzzy_calories": 608.12,
                "hybrid_calories": 606.20,
                "ml_fatigue": 672.5,
                "fuzzy_fatigue": 450.0,
                "hybrid_fatigue": 605.75,
                "fatigue_level": "Orta (Moderate)",
                "ml_hourly": 807.0,
                "burn_factor": 810.83
            }
        }


# ===== API ENDPOINT'LERİ =====

@app.get("/")
async def root():
    """
    Kök endpoint - API durumu
    """
    return {
        "message": "KaloriBul API is Running",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """
    Sağlık kontrolü endpoint'i
    """
    return {
        "status": "healthy",
        "model_loaded": engine.ml_model is not None,
        "fuzzy_loaded": engine.fuzzy_sim is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_calories(request: FitnessRequest):
    """
    Kalori tahmini endpoint'i
    
    Args:
        request (FitnessRequest): Kullanıcı kilosu, MET değeri ve süre
    
    Returns:
        PredictionResponse: ML, Fuzzy ve Hibrit tahmin sonuçları
    
    Raises:
        HTTPException: Model yüklü değilse veya tahmin hatası olursa
    """
    try:
        # Model kontrolü
        if engine.ml_model is None or engine.fuzzy_sim is None:
            raise HTTPException(
                status_code=500,
                detail="Model henüz yüklenmedi. Lütfen daha sonra tekrar deneyin."
            )
        
        # Hibrit tahmin yap
        result = engine.predict_hybrid(
            weight=request.weight,
            mets=request.mets,
            duration_minutes=request.duration
        )
        
        # Sonucu döndür
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz değer: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin hatası: {str(e)}"
        )


@app.get("/activities")
async def get_sample_activities():
    """
    Örnek aktiviteler ve MET değerleri
    """
    return {
        "activities": [
            {"name": "Yürüyüş (Normal)", "mets": 3.8, "intensity": "Düşük"},
            {"name": "Yürüyüş (Hızlı)", "mets": 5.0, "intensity": "Orta"},
            {"name": "Koşu (9.6 km/h)", "mets": 10.5, "intensity": "Yüksek"},
            {"name": "Koşu (12.8 km/h)", "mets": 14.0, "intensity": "Çok Yüksek"},
            {"name": "Bisiklet (Düşük)", "mets": 6.0, "intensity": "Orta"},
            {"name": "Bisiklet (Yüksek)", "mets": 10.0, "intensity": "Yüksek"},
            {"name": "Yüzme (Serbest, Yavaş)", "mets": 6.0, "intensity": "Orta"},
            {"name": "Yüzme (Serbest, Hızlı)", "mets": 10.0, "intensity": "Yüksek"},
            {"name": "Ağırlık Kaldırma", "mets": 6.0, "intensity": "Orta"},
            {"name": "HIIT", "mets": 14.0, "intensity": "Çok Yüksek"},
            {"name": "Yoga", "mets": 2.5, "intensity": "Çok Düşük"},
            {"name": "Pilates", "mets": 3.0, "intensity": "Düşük"},
            {"name": "Basketbol", "mets": 8.0, "intensity": "Yüksek"},
            {"name": "Futbol", "mets": 10.0, "intensity": "Yüksek"},
            {"name": "Tenis", "mets": 7.3, "intensity": "Orta-Yüksek"}
        ]
    }


# ===== ÇALIŞTIRMA =====
if __name__ == "__main__":
    # Uvicorn ile serveri başlat
    print("\n🌐 Server başlatılıyor...")
    print("📍 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔄 API'yi durdurmak için: CTRL+C\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Tüm network interface'lerine dinle
        port=8000,
        log_level="info"
    )
