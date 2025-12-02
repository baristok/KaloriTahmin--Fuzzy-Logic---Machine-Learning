import pandas as pd
import re

def clean_numeric_string(value):
    """
    Sayısal değerlerdeki virgülleri ve diğer işaretleri temizler.
    Örnek: "1,060" -> 1060.0, "600–800" -> ortalama değer
    """
    if pd.isna(value):
        return None
    
    value_str = str(value)
    
    # Eğer değer aralık içeriyorsa (örn: "600–800"), ortalamayı al
    if '–' in value_str or '-' in value_str:
        # Tire veya uzun tire ile ayır
        parts = re.split('–|-', value_str)
        if len(parts) == 2:
            try:
                # Virgülleri temizle ve sayılara çevir
                num1 = float(parts[0].replace(',', '').strip())
                num2 = float(parts[1].replace(',', '').strip())
                return (num1 + num2) / 2
            except:
                pass
    
    # "+" işareti varsa kaldır (örn: "1,100+")
    value_str = value_str.replace('+', '')
    
    # Virgülleri kaldır
    value_str = value_str.replace(',', '')
    
    try:
        return float(value_str)
    except:
        return None

def extract_weight_kg(column_name):
    """
    Sütun adından kilogram değerini çıkarır.
    Örnek: "50 kg (110 lb)" -> 50.0
    """
    match = re.match(r'(\d+)\s*kg', column_name)
    if match:
        return float(match.group(1))
    return None

def main():
    print("CSV dosyası okunuyor...")
    # 1. Dosyayı pandas ile oku (on_bad_lines='skip')
    df = pd.read_csv('datasets/Calories.csv', on_bad_lines='skip')
    
    print(f"Orijinal veri boyutu: {df.shape}")
    print(f"Sütunlar: {df.columns.tolist()}\n")
    
    # 2. Sayısal sütunlardaki binlik ayraçları temizle
    print("Sayısal değerler temizleniyor...")
    
    # Tüm sütunları kontrol et ve sayısal olanları temizle
    for col in df.columns:
        # METs ve weight sütunlarını temizle
        if 'kg' in col or col == 'METs':
            df[col] = df[col].apply(clean_numeric_string)
    
    # 3. Weight sütunlarını Wide Format'tan Long Format'a çevir
    print("Veri Wide Format'tan Long Format'a çevriliyor...")
    
    # Weight sütunlarını belirle (kg içeren sütunlar)
    weight_columns = [col for col in df.columns if 'kg' in col]
    
    # Sabit tutulacak sütunlar (id_vars)
    id_columns = ['Activity', 'Subtype', 'Intensity', 'Duration (min)', 
                  'Distance (km)', 'METs']
    
    # pd.melt() ile long format'a çevir
    df_long = pd.melt(
        df,
        id_vars=id_columns,
        value_vars=weight_columns,
        var_name='Weight_Category',
        value_name='Calories'
    )
    
    # 4. Weight_kg sütununu sadece sayısal yap
    print("Weight_kg sütunu oluşturuluyor...")
    df_long['Weight_kg'] = df_long['Weight_Category'].apply(extract_weight_kg)
    
    # Sütunları yeniden düzenle
    df_long = df_long[[
        'Activity', 'Subtype', 'Intensity', 'Duration (min)', 
        'Distance (km)', 'METs', 'Weight_kg', 'Calories'
    ]]
    
    # Sütun adlarını daha temiz hale getir
    df_long.columns = ['Activity', 'Subtype', 'Intensity', 'Duration', 
                       'Distance', 'METs', 'Weight_kg', 'Calories']
    
    # NaN değerleri olan satırları temizle (isteğe bağlı)
    df_long = df_long.dropna(subset=['Calories', 'Weight_kg'])
    
    print(f"\nTemizlenmiş veri boyutu: {df_long.shape}")
    print(f"Sütunlar: {df_long.columns.tolist()}")
    print(f"\nİlk birkaç satır:")
    print(df_long.head(10))
    
    # Veri tipleri
    print(f"\nVeri tipleri:")
    print(df_long.dtypes)
    
    # İstatistikler
    print(f"\nSayısal sütunların özeti:")
    print(df_long[['Duration', 'METs', 'Weight_kg', 'Calories']].describe())
    
    # 5. Sonucu 'cleaned_data.csv' olarak kaydet
    print("\nTemizlenmiş veri 'cleaned_data.csv' dosyasına kaydediliyor...")
    df_long.to_csv('cleaned_data.csv', index=False)
    
    print("✓ İşlem tamamlandı!")
    print(f"Toplam {len(df_long)} satır kaydedildi.")

if __name__ == '__main__':
    main()
