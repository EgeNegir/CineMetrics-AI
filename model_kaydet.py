import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression # <-- ARTIK BU ALGORİTMAYI KULLANIYORUZ

print("Veriler yükleniyor...")
df = pd.read_csv('temiz_film_verisi.csv')

# 1. Özellik Mühendisliği (Yönetmen VE Başrol)
dir_scores = df.groupby('director')['imdb_score'].mean().to_dict()
df['director_avg_score'] = df['director'].map(dir_scores)

act_scores = df.groupby('lead_actor')['imdb_score'].mean().to_dict()
df['actor_avg_score'] = df['lead_actor'].map(act_scores)

df = pd.get_dummies(df, columns=['main_genre'], drop_first=True)

X = df.drop(columns=['movie_id', 'title', 'director', 'lead_actor', 'imdb_score'])
y = df['imdb_score']
X = X.fillna(X.mean())

# 2. HASSAS MODEL EĞİTİMİ (Formül Tabanlı)
print("Hassas Matematiksel Model (Linear Regression) eğitiliyor...")
sampiyon_model = LinearRegression() # Model değişti
sampiyon_model.fit(X, y)

# 3. DONDURMA (KAYDETME) İŞLEMİ
print("Model ve hafıza dosyaları diske kaydediliyor...")
joblib.dump(sampiyon_model, 'yapay_zeka_modeli.pkl')
joblib.dump(list(X.columns), 'model_kolonlari.pkl') 
joblib.dump(dir_scores, 'yonetmen_skorlari.pkl') 
joblib.dump(act_scores, 'oyuncu_skorlari.pkl')

print("\n❄️ Zeka (Regresyon) başarıyla donduruldu! Artık 4 adet '.pkl' dosyamız var.")