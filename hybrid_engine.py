"""
SmartFitHybrid - Hibrit Kalori Tahmin Motoru
Random Forest (ML) ve Fuzzy Logic sistemlerini birleştiren akıllı egzersiz kalori tahmini.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class SmartFitHybrid:
    """
    Hibrit Kalori Tahmin Motoru
    ML (Random Forest) ve Fuzzy Logic sistemlerini birleştirerek kalori tahmini yapar.
    """
    
    def __init__(self, data_path='datasets/cleaned_data.csv'):
        """
        Sınıf başlatıcı
        
        Args:
            data_path (str): Temizlenmiş veri dosyasının yolu
        """
        self.data_path = data_path
        self.ml_model = None
        self.fuzzy_sim = None
        self.fatigue_fuzzy_sim = None  # Yorgunluk fuzzy sistemi
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """
        1. Veri Hazırlığı ve Normalizasyon
        - CSV'yi yükler
        - Saatlik kalori yakım hızını hesaplar (Calories_Hourly)
        - Veriyi train/test olarak ayırır (%80/%20)
        """
        print("📊 Veri yükleniyor...")
        df = pd.read_csv(self.data_path)
        
        print(f"Toplam veri sayısı: {len(df)} satır")
        
        # Saatlik kalori yakım hızını hesapla
        # Bu sayede model "ne kadar sürede ne kadar kalori" yakıldığını öğrenir
        df['Calories_Hourly'] = df['Calories'] / (df['Duration'] / 60)
        
        # NaN değerleri temizle
        df = df.dropna(subset=['Weight_kg', 'METs', 'Calories_Hourly'])
        
        # X (Girdiler): Kilo ve METs
        # y (Hedef): Saatlik kalori yakımı
        X = df[['Weight_kg', 'METs']].values
        y = df['Calories_Hourly'].values
        
        # Veriyi %80 train, %20 test olarak ayır
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✅ Veri hazırlandı:")
        print(f"   - Eğitim seti: {len(self.X_train)} örnek")
        print(f"   - Test seti: {len(self.X_test)} örnek")
        print(f"   - Özellikler: Weight_kg, METs")
        print(f"   - Hedef: Calories_Hourly\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_ml_model(self):
        """
        2. Makine Öğrenimi Modeli (Random Forest)
        - Random Forest Regressor ile model eğitir
        - Test seti üzerinde performansı değerlendirir
        """
        if self.X_train is None:
            raise ValueError("Önce prepare_data() fonksiyonunu çalıştırın!")
        
        print("🤖 Makine Öğrenimi Modeli Eğitiliyor (Random Forest)...")
        
        # Random Forest Regressor oluştur ve eğit
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1  # Tüm CPU çekirdeklerini kullan
        )
        
        self.ml_model.fit(self.X_train, self.y_train)
        
        # Test seti üzerinde tahmin yap
        y_pred = self.ml_model.predict(self.X_test)
        
        # Performans metrikleri
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        print(f"✅ ML Model Eğitimi Tamamlandı!")
        print(f"   - ML Model Accuracy (R² Score): {r2:.4f}")
        print(f"   - Mean Absolute Error: {mae:.2f} kcal/hour")
        print(f"   - Feature Importances: Weight={self.ml_model.feature_importances_[0]:.3f}, "
              f"METs={self.ml_model.feature_importances_[1]:.3f}\n")
        
        return self.ml_model
    
    def setup_fuzzy_model(self):
        """
        3. Bulanık Mantık Sistemi (Fuzzy Logic)
        - Girdi ve çıktı değişkenlerini tanımlar
        - Üyelik fonksiyonlarını oluşturur
        - Fuzzy kurallarını tanımlar
        """
        print("🧠 Fuzzy Logic Sistemi Kuruluyor...")
        
        # ===== GİRDİLER (Antecedents) =====
        
        # Girdi 1: Kilo (40-140 kg)
        weight = ctrl.Antecedent(np.arange(40, 141, 1), 'weight')
        weight['Light'] = fuzz.trimf(weight.universe, [40, 40, 70])
        weight['Average'] = fuzz.trimf(weight.universe, [60, 80, 100])
        weight['Heavy'] = fuzz.trimf(weight.universe, [90, 140, 140])
        
        # Girdi 2: Yoğunluk/Şiddet (0-18 METs)
        intensity = ctrl.Antecedent(np.arange(0, 18.1, 0.1), 'intensity')
        intensity['Low'] = fuzz.trimf(intensity.universe, [0, 0, 4])
        intensity['Moderate'] = fuzz.trimf(intensity.universe, [3, 6, 9])
        intensity['High'] = fuzz.trimf(intensity.universe, [8, 12, 15])
        intensity['Extreme'] = fuzz.trimf(intensity.universe, [14, 18, 18])
        
        # ===== ÇIKTI (Consequent) =====
        
        # Çıktı: Kalori Faktörü (0-1200)
        burn_factor = ctrl.Consequent(np.arange(0, 1201, 1), 'burn_factor')
        burn_factor['Low'] = fuzz.trimf(burn_factor.universe, [0, 0, 300])
        burn_factor['Medium'] = fuzz.trimf(burn_factor.universe, [200, 500, 700])
        burn_factor['High'] = fuzz.trimf(burn_factor.universe, [600, 850, 1000])
        burn_factor['VeryHigh'] = fuzz.trimf(burn_factor.universe, [950, 1200, 1200])
        
        # ===== FUZZY KURALLARI =====
        
        rule1 = ctrl.Rule(intensity['Low'], burn_factor['Low'])
        rule2 = ctrl.Rule(intensity['Moderate'] & weight['Light'], burn_factor['Medium'])
        rule3 = ctrl.Rule(intensity['Moderate'] & weight['Average'], burn_factor['Medium'])
        rule4 = ctrl.Rule(intensity['Moderate'] & weight['Heavy'], burn_factor['High'])
        rule5 = ctrl.Rule(intensity['High'] & weight['Light'], burn_factor['Medium'])
        rule6 = ctrl.Rule(intensity['High'] & weight['Average'], burn_factor['High'])
        rule7 = ctrl.Rule(intensity['High'] & weight['Heavy'], burn_factor['VeryHigh'])
        rule8 = ctrl.Rule(intensity['Extreme'], burn_factor['VeryHigh'])
        
        # Kontrol sistemi oluştur
        burn_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
        self.fuzzy_sim = ctrl.ControlSystemSimulation(burn_ctrl)
        
        print("✅ Fuzzy Logic Sistemi Kuruldu!")
        print("   - Girdi 1: weight (Light, Average, Heavy)")
        print("   - Girdi 2: intensity (Low, Moderate, High, Extreme)")
        print("   - Çıktı: burn_factor (Low, Medium, High, VeryHigh)")
        print("   - Kural sayısı: 8\n")
        
        return self.fuzzy_sim
    
    def setup_fatigue_fuzzy(self):
        """
        Yorgunluk için ayrı Fuzzy Logic Sistemi
        - Yorgunluk seviyesini METs ve süreye göre hesaplar
        """
        print("💪 Fuzzy Yorgunluk Sistemi Kuruluyor...")
        
        # ===== GİRDİLER =====
        
        # Girdi 1: Yoğunluk (0-18 METs)
        intensity = ctrl.Antecedent(np.arange(0, 18.1, 0.1), 'intensity')
        intensity['low'] = fuzz.trimf(intensity.universe, [0, 0, 6])
        intensity['moderate'] = fuzz.trimf(intensity.universe, [4, 9, 12])
        intensity['high'] = fuzz.trimf(intensity.universe, [10, 18, 18])
        
        # Girdi 2: Süre (0-180 dakika)
        duration = ctrl.Antecedent(np.arange(0, 181, 1), 'duration')
        duration['short'] = fuzz.trimf(duration.universe, [0, 0, 30])
        duration['medium'] = fuzz.trimf(duration.universe, [20, 60, 90])
        duration['long'] = fuzz.trimf(duration.universe, [80, 180, 180])
        
        # ===== ÇIKTI =====
        
        # Çıktı: Fatigue Score (0-1000)
        fatigue = ctrl.Consequent(np.arange(0, 1001, 1), 'fatigue')
        fatigue['very_low'] = fuzz.trimf(fatigue.universe, [0, 0, 200])
        fatigue['low'] = fuzz.trimf(fatigue.universe, [150, 300, 450])
        fatigue['moderate'] = fuzz.trimf(fatigue.universe, [400, 500, 600])
        fatigue['high'] = fuzz.trimf(fatigue.universe, [550, 700, 850])
        fatigue['very_high'] = fuzz.trimf(fatigue.universe, [800, 1000, 1000])
        
        # ===== FUZZY KURALLARI (9 kombinasyon) =====
        
        rule1 = ctrl.Rule(intensity['low'] & duration['short'], fatigue['very_low'])
        rule2 = ctrl.Rule(intensity['low'] & duration['medium'], fatigue['low'])
        rule3 = ctrl.Rule(intensity['low'] & duration['long'], fatigue['moderate'])
        rule4 = ctrl.Rule(intensity['moderate'] & duration['short'], fatigue['low'])
        rule5 = ctrl.Rule(intensity['moderate'] & duration['medium'], fatigue['moderate'])
        rule6 = ctrl.Rule(intensity['moderate'] & duration['long'], fatigue['high'])
        rule7 = ctrl.Rule(intensity['high'] & duration['short'], fatigue['moderate'])
        rule8 = ctrl.Rule(intensity['high'] & duration['medium'], fatigue['high'])
        rule9 = ctrl.Rule(intensity['high'] & duration['long'], fatigue['very_high'])
        
        # Kontrol sistemi oluştur
        fatigue_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.fatigue_fuzzy_sim = ctrl.ControlSystemSimulation(fatigue_ctrl)
        
        print("✅ Fuzzy Yorgunluk Sistemi Kuruldu!")
        print("   - Girdi 1: intensity (low, moderate, high)")
        print("   - Girdi 2: duration (short, medium, long)")
        print("   - Çıktı: fatigue (very_low, low, moderate, high, very_high)")
        print("   - Kural sayısı: 9\n")
        
        return self.fatigue_fuzzy_sim
    
    def predict_hybrid(self, weight, mets, duration_minutes):
        """
        4. Hibrit Tahmin (ML + Fuzzy)
        
        Args:
            weight (float): Kullanıcının kilosu (kg)
            mets (float): Egzersiz yoğunluğu (METs)
            duration_minutes (float): Egzersiz süresi (dakika)
        
        Returns:
            dict: ML, Fuzzy ve Hibrit tahmin sonuçları + 3 ayrı yorgunluk tahmini
        """
        if self.ml_model is None or self.fuzzy_sim is None:
            raise ValueError("Önce modelleri eğitin (train_ml_model ve setup_fuzzy_model)!")
        
        # ===== ML TAHMİNİ (KALORİ) =====
        ml_hourly = self.ml_model.predict([[weight, mets]])[0]
        ml_calories = ml_hourly * (duration_minutes / 60)
        
        # ===== FUZZY TAHMİNİ (KALORİ) =====
        self.fuzzy_sim.input['weight'] = weight
        self.fuzzy_sim.input['intensity'] = mets
        self.fuzzy_sim.compute()
        burn_factor = self.fuzzy_sim.output['burn_factor']
        fuzzy_calories = burn_factor * (duration_minutes / 60)
        
        # ===== HİBRİT KALORİ =====
        hybrid_calories = (ml_calories * 0.7) + (fuzzy_calories * 0.3)
        
        # ===== ML YORGUNLUK =====
        # Kalori bazlı normalizasyon: Ağır egzersiz ~10 kcal/min yakar
        # Max yorgunluk = 1000
        calories_per_minute = ml_calories / duration_minutes if duration_minutes > 0 else 0
        ml_fatigue = min(calories_per_minute * 100, 1000)
        
        # ===== FUZZY YORGUNLUK =====
        if self.fatigue_fuzzy_sim is not None:
            self.fatigue_fuzzy_sim.input['intensity'] = mets
            self.fatigue_fuzzy_sim.input['duration'] = duration_minutes
            self.fatigue_fuzzy_sim.compute()
            fuzzy_fatigue = self.fatigue_fuzzy_sim.output['fatigue']
        else:
            # Fuzzy yorgunluk sistemi kurulmamışsa basit formül
            fuzzy_fatigue = mets * duration_minutes
        
        # ===== HİBRİT YORGUNLUK =====
        hybrid_fatigue = (ml_fatigue * 0.7) + (fuzzy_fatigue * 0.3)
        
        # ===== YORGUNLUK SEVİYESİ BELİRLEME =====
        if hybrid_fatigue < 200:
            fatigue_level = 'Çok Düşük (Very Low)'
        elif hybrid_fatigue < 400:
            fatigue_level = 'Düşük (Low)'
        elif hybrid_fatigue < 600:
            fatigue_level = 'Orta (Moderate)'
        elif hybrid_fatigue < 800:
            fatigue_level = 'Yüksek (High)'
        else:
            fatigue_level = 'Çok Yüksek (Very High)'
        
        # Sonuçları döndür
        return {
            'ml_calories': round(ml_calories, 2),
            'fuzzy_calories': round(fuzzy_calories, 2),
            'hybrid_calories': round(hybrid_calories, 2),
            'ml_fatigue': round(ml_fatigue, 2),
            'fuzzy_fatigue': round(fuzzy_fatigue, 2),
            'hybrid_fatigue': round(hybrid_fatigue, 2),
            'fatigue_level': fatigue_level,
            'ml_hourly': round(ml_hourly, 2),
            'burn_factor': round(burn_factor, 2)
        }
    
    def batch_predict(self, test_cases):
        """
        Birden fazla test durumu için toplu tahmin
        
        Args:
            test_cases (list): [(weight, mets, duration), ...] formatında liste
        
        Returns:
            pd.DataFrame: Sonuçları içeren DataFrame
        """
        results = []
        
        for weight, mets, duration in test_cases:
            prediction = self.predict_hybrid(weight, mets, duration)
            results.append({
                'Weight (kg)': weight,
                'METs': mets,
                'Duration (min)': duration,
                'ML Calories': prediction['ml_calories'],
                'Fuzzy Calories': prediction['fuzzy_calories'],
                'Hybrid Calories': prediction['hybrid_calories'],
                'Fatigue Level': prediction['fatigue_level']
            })
        
        return pd.DataFrame(results)


# ===== TEST VE ÇALIŞTIRMA =====
if __name__ == "__main__":
    print("=" * 70)
    print("🏋️  SMARTFIT HYBRID - Akıllı Egzersiz Kalori Tahmin Motoru")
    print("=" * 70)
    print()
    
    # Sistem başlat
    engine = SmartFitHybrid()
    
    # 1. Veriyi hazırla
    engine.prepare_data()
    
    # 2. ML modelini eğit
    engine.train_ml_model()
    
    # 3. Fuzzy sistemini kur
    engine.setup_fuzzy_model()
    
    # 4. Örnek tahmin (75kg, 10 METs, 45 dakika)
    print("=" * 70)
    print("🎯 ÖRNEK TAHMİN")
    print("=" * 70)
    
    weight_test = 75  # kg
    mets_test = 10    # METs (örn: koşu)
    duration_test = 45  # dakika
    
    print(f"Girdiler: {weight_test} kg, {mets_test} METs, {duration_test} dakika\n")
    
    result = engine.predict_hybrid(weight_test, mets_test, duration_test)
    
    print("Sonuçlar:")
    print(f"  🤖 ML Tahmini:      {result['ml_calories']:.0f} kcal")
    print(f"  🧠 Fuzzy Tahmini:   {result['fuzzy_calories']:.0f} kcal")
    print(f"  ⚡ Hibrit Tahmin:   {result['hybrid_calories']:.0f} kcal")
    print(f"  💪 Yorgunluk Skoru: {result['fatigue_score']:.0f}")
    print(f"  😰 Yorgunluk Seviyesi: {result['fatigue_level']}")
    print()
    
    # 5. Çoklu test senaryoları
    print("=" * 70)
    print("📊 TOPLU TEST SONUÇLARI")
    print("=" * 70)
    print()
    
    test_scenarios = [
        (50, 3, 30),    # Hafif kişi, düşük yoğunluk, kısa süre
        (70, 6, 45),    # Orta kişi, orta yoğunluk, orta süre
        (90, 12, 60),   # Ağır kişi, yüksek yoğunluk, uzun süre
        (80, 15, 40),   # Orta kişi, çok yüksek yoğunluk, orta süre
        (60, 8, 30),    # Hafif kişi, orta-yüksek yoğunluk, kısa süre
    ]
    
    results_df = engine.batch_predict(test_scenarios)
    print(results_df.to_string(index=False))
    print()
    
    print("=" * 70)
    print("✅ Test tamamlandı!")
    print("=" * 70)
