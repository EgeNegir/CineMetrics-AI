import joblib
import pandas as pd
import numpy as np

# 1. Dondurulmuş Yapay Zeka Beynini ve Hafızalarını Yükle
print("🧠 Yapay zeka beyni ve güncel hafıza dosyaları (Yönetmen + Oyuncu) yükleniyor...")
model = joblib.load('yapay_zeka_modeli.pkl')
model_kolonlari = joblib.load('model_kolonlari.pkl')
yonetmen_hafizasi = joblib.load('yonetmen_skorlari.pkl')
oyuncu_hafizasi = joblib.load('oyuncu_skorlari.pkl') # Yeni hafıza parçası

def imdb_puani_tahmin_et(butce, sure, ana_tur, yonetmen_adi, oyuncu_adi):
    # a. Yönetmen ve Oyuncu puanlarını hafızadan çek (Yoksa ortalama 6.4 kabul et)
    yonetmen_skoru = yonetmen_hafizasi.get(yonetmen_adi, 6.4)
    oyuncu_skoru = oyuncu_hafizasi.get(oyuncu_adi, 6.4)
    
    # b. Girdi verisini hazırla
    yeni_film = pd.DataFrame([{
        'budget': butce,
        'revenue': 0, 
        'runtime': sure,
        'director_avg_score': yonetmen_skoru,
        'actor_avg_score': oyuncu_skoru, # Artık başrolün etkisi devrede!
        'main_genre': ana_tur
    }])
    
    # c. Tür bilgisini One-Hot Encoding yap
    yeni_film_kodlanmis = pd.get_dummies(yeni_film, columns=['main_genre'])
    
    # d. Eksik kolonları 0 ile doldur ve sırayı eşitle
    for col in model_kolonlari:
        if col not in yeni_film_kodlanmis.columns:
            yeni_film_kodlanmis[col] = 0
            
    yeni_film_kodlanmis = yeni_film_kodlanmis[model_kolonlari]
    
    # e. TAHMİN ET!
    tahmin = model.predict(yeni_film_kodlanmis)[0]
    return tahmin

# --- CANLI TEST SENARYOLARI ---
print("\n🎬 --- YAPAY ZEKA CANLI TAHMİN SİMÜLASYONU ---")

# Senaryo 1: Rüya Takımı (Nolan + DiCaprio)
puan1 = imdb_puani_tahmin_et(200000000, 160, 'Action', 'Christopher Nolan', 'Leonardo DiCaprio')
print(f"Senaryo 1 (Nolan & DiCaprio): Tahmini IMDb Puanı: {puan1:.1f}")

# Senaryo 2: Tarantino + Brad Pitt Kapışması
puan2 = imdb_puani_tahmin_et(100000000, 140, 'Drama', 'Quentin Tarantino', 'Brad Pitt')
print(f"Senaryo 2 (Tarantino & Pitt): Tahmini IMDb Puanı: {puan2:.1f}")

# Senaryo 3: Düşük Bütçeli Korku (Tanınmayan Ekip)
puan3 = imdb_puani_tahmin_et(2000000, 85, 'Horror', 'Unknown Director', 'Unknown Actor')
print(f"Senaryo 3 (Bilinmeyen Ekip): Tahmini IMDb Puanı: {puan3:.1f}")

# Senaryo 4: Senin Denemen!
print("\n--- Kendi Senaryonu Test Et ---")
h_butce = 50000000
h_sure = 110
h_tur = 'Comedy'
h_yonetmen = 'Christoper Nolan'
h_oyuncu = 'Brad Pitt'

puan4 = imdb_puani_tahmin_et(h_butce, h_sure, h_tur, h_yonetmen, h_oyuncu)
print(f"Senaryo 4 ({h_yonetmen} & {h_oyuncu}): Tahmini IMDb Puanı: {puan4:.1f}")

# --- GERÇEK FİLM TESTLERİ (SAĞLAMA) ---
print("\n🔍 --- GERÇEK FİLM SAĞLAMASI ---")

# 1. Gerçek Film: Avatar (2009)
# Veritabanındaki Gerçek Puanı: 7.2
avatar_tahmin = imdb_puani_tahmin_et(237000000, 162, 'Action', 'James Cameron', 'Sam Worthington')
print(f"Avatar (2009) -> Tahmin: {avatar_tahmin:.1f} | Gerçek Puan: 7.2")

# 2. Gerçek Film: The Dark Knight Rises (2012)
# Veritabanındaki Gerçek Puanı: 7.6
batman_tahmin = imdb_puani_tahmin_et(250000000, 165, 'Action', 'Christopher Nolan', 'Christian Bale')
print(f"The Dark Knight Rises (2012) -> Tahmin: {batman_tahmin:.1f} | Gerçek Puan: 7.6")

# 3. Gerçek Film: Titanic (1997)
# Veritabanındaki Gerçek Puanı: 7.5
titanic_tahmin = imdb_puani_tahmin_et(200000000, 194, 'Drama', 'James Cameron', 'Leonardo DiCaprio')
print(f"Titanic (1997) -> Tahmin: {titanic_tahmin:.1f} | Gerçek Puan: 7.5")