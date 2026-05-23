import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor  # <-- Karar ağacını doğru adresten çağırdık
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Temizlenmiş veriyi yükle 
df = pd.read_csv('temiz_film_verisi.csv')


# ÖZEL MÜHENDİSLİK: Yönetmen ve Oyuncu Puanlarını Hesaplama

print("Yönetmen başarı skorları hesaplanıyor...")
director_scores = df.groupby('director')['imdb_score'].mean().to_dict()
df['director_avg_score'] = df['director'].map(director_scores)

print("Başrol oyuncusu başarı skorları hesaplanıyor...")
actor_scores = df.groupby('lead_actor')['imdb_score'].mean().to_dict()
df['actor_avg_score'] = df['lead_actor'].map(actor_scores)

# 2. Kategorik Verileri Sayısallaştırma 
print("Film türleri matematiksel matrislere (0 ve 1) dönüştürülüyor...")
df = pd.get_dummies(df, columns=['main_genre'], drop_first=True)

# 3. Model İçin Girdi (X) ve Çıktı (y) Ayarlama

X = df.drop(columns=['movie_id', 'title', 'director', 'lead_actor', 'imdb_score'])
y = df['imdb_score']

# Boşdeğerleri ortalama ile doldurma
X = X.fillna(X.mean())

# 4. Veriyi Bölme (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelleri Tanımlama
modeller = {
    "Çoklu Doğrusal Regresyon (Baseline)": LinearRegression(),
    "Karar Ağaçları": DecisionTreeRegressor(random_state=42),
    "Random Forest (Topluluk Modeli)": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\n🤖 YAPAY ZEKA MODELLERİ (OYUNCU DESTEKLİ) EĞİTİLİYOR VE YARIŞTIRILIYOR...\n")

# 6. Eğit ve Test Et
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    tahminler = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, tahminler))
    mae = mean_absolute_error(y_test, tahminler)
    r2 = r2_score(y_test, tahminler)
    
    print(f"--- {isim} ---")
    print(f"RMSE (Hata Payı): {rmse:.2f} Puan")
    print(f"MAE (Ort. Mutlak Hata): {mae:.2f} Puan")
    print(f"R² (Açıklanabilirlik): %{r2*100:.2f}\n")