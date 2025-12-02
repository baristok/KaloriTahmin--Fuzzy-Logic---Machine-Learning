# 🏋️ SmartFit AI - Akıllı Egzersiz Kalori Asistanı

Hibrit (ML + Fuzzy Logic) kalori tahmin motoru ile çalışan akıllı egzersiz asistanı.

## 🎯 Özellikler

- 🤖 **Makine Öğrenimi**: Random Forest ile %92.86 doğruluk
- 🧠 **Fuzzy Logic**: 8 kural ile akıllı mantık sistemi
- ⚡ **Hibrit Tahmin**: ML (%70) + Fuzzy (%30) birleşimi
- 💪 **Yorgunluk Analizi**: Otomatik yorgunluk seviyesi tespiti
- 🌐 **RESTful API**: FastAPI ile yüksek performanslı backend

## 📦 Kurulum

### 1. Sanal Ortamı Aktifleştir

```bash
source venv/bin/activate
```

### 2. Gerekli Paketler (Zaten kurulu)

```bash
pip install pandas numpy scikit-learn scikit-fuzzy scipy networkx packaging fastapi uvicorn
```

## 🚀 Kullanım

### Backend API'yi Başlat

```bash
python main.py
```

API varsayılan olarak http://localhost:8000 adresinde çalışacaktır.

**Alternatif:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoint'leri

#### 1. Ana Sayfa

```bash
GET http://localhost:8000/
```

#### 2. Sağlık Kontrolü

```bash
GET http://localhost:8000/health
```

#### 3. Kalori Tahmini

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "weight": 75.0,
  "mets": 10.0,
  "duration": 45.0
}
```

**Response:**

```json
{
  "ml_calories": 605.25,
  "fuzzy_calories": 608.12,
  "hybrid_calories": 606.2,
  "fatigue_score": 450.0,
  "fatigue_level": "Orta (Moderate)",
  "ml_hourly": 807.0,
  "burn_factor": 810.83
}
```

#### 4. Örnek Aktiviteler

```bash
GET http://localhost:8000/activities
```

### API Dokümantasyonu

API başladıktan sonra otomatik dokümantasyona erişebilirsiniz:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📁 Proje Yapısı

```
akilliegzersizasistani/
├── datasets/
│   ├── Calories.csv          # Ham veri
│   └── cleaned_data.csv      # Temizlenmiş veri
├── venv/                      # Sanal ortam
├── data_cleaning.py          # Veri temizleme scripti
├── hybrid_engine.py          # Hibrit motor (ML + Fuzzy)
├── main.py                   # FastAPI backend
└── README.md                 # Bu dosya
```

## 🧪 Test

### Hibrit Motoru Test Et

```bash
python hybrid_engine.py
```

### Veri Temizleme

```bash
python data_cleaning.py
```

## 📊 Model Performansı

- **ML Model (Random Forest)**

  - R² Score: 0.9286 (%92.86 doğruluk)
  - Mean Absolute Error: 53.61 kcal/saat
  - Feature Importances: METs (%83.3), Weight (%16.7)

- **Fuzzy Logic System**
  - 8 kural
  - 2 girdi: Weight (40-140kg), Intensity (0-18 METs)
  - 1 çıktı: Burn Factor (0-1200)

## 🔧 Teknoloji Stack'i

### Backend

- **FastAPI** 0.123.5 - Modern web framework
- **Uvicorn** 0.38.0 - ASGI server
- **Pydantic** 2.12.5 - Veri validasyonu

### ML & AI

- **scikit-learn** 1.7.2 - Makine öğrenimi
- **scikit-fuzzy** 0.5.0 - Fuzzy logic
- **NumPy** 2.3.5 - Numerical computing
- **Pandas** 2.3.3 - Veri analizi

### Diğer

- **NetworkX** 3.6 - Fuzzy kural ağı
- **SciPy** 1.16.3 - Bilimsel hesaplamalar

## 💡 Örnek Kullanım (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "weight": 75.0,
    "mets": 10.0,
    "duration": 45.0
  }'
```

## 📝 MET Değerleri Örnekleri

| Aktivite         | METs | Yoğunluk   |
| ---------------- | ---- | ---------- |
| Yoga             | 2.5  | Çok Düşük  |
| Yürüyüş (Normal) | 3.8  | Düşük      |
| Yürüyüş (Hızlı)  | 5.0  | Orta       |
| Yüzme (Yavaş)    | 6.0  | Orta       |
| Koşu (9.6 km/h)  | 10.5 | Yüksek     |
| Koşu (12.8 km/h) | 14.0 | Çok Yüksek |
| HIIT             | 14.0 | Çok Yüksek |

## 🎓 Lisans

Bu proje eğitim amaçlıdır.

## 👨‍💻 Geliştirici

Backend API geliştirmesi tamamlandı ve production-ready!
