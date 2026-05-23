import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1. Veri seti yükleniyor...")
df = pd.read_csv('temiz_film_verisi.csv')

#EKSİK VERİLERİN İŞLENMESİ
print("2. Eksik veya mantıksız (0) veriler medyan ile dolduruluyor...")
# Bütçesi veya geliri 0 olarak girilmiş satırları bul ve o sütunun medyanı ile değiştir
df['budget'] = df['budget'].replace(0, df['budget'].median())
df['revenue'] = df['revenue'].replace(0, df['revenue'].median())
df['runtime'] = df['runtime'].replace(0, df['runtime'].median())

#AYKIRI DEĞERLERİN IQR İLE TEMİZLENMESİ 
print("3. Aykırı değerler (Outliers) IQR yöntemi ile temizleniyor...")
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= alt_sinir) & (dataframe[column] <= ust_sinir)]

# Bütçe ve süredeki mantıksız uç değerleri (1 dolarlık filmler vs.) atıyoruz
df = remove_outliers_iqr(df, 'budget')
df = remove_outliers_iqr(df, 'runtime')

#  İLİŞKİSEL ÖZELLİK MÜHENDİSLİĞİ 
print("4. Yönetmen ve Oyuncu geçmiş başarı ortalamaları hesaplanıyor...")
dir_scores = df.groupby('director')['imdb_score'].mean().to_dict()
df['director_avg_score'] = df['director'].map(dir_scores)

act_scores = df.groupby('lead_actor')['imdb_score'].mean().to_dict()
df['actor_avg_score'] = df['lead_actor'].map(act_scores)

# KATEGORİK DEĞİŞKENLERİN SAYISALLAŞTIRILMASI (One-Hot Encoding) 
print("5. Kategorik veriler (Tür) One-Hot Encoding ile sayısallaştırılıyor...")
# Sadece gerekli kolonları seçiyoruz
secili_kolonlar = ['budget', 'runtime', 'main_genre', 'director_avg_score', 'actor_avg_score', 'imdb_score']
df_model = df[secili_kolonlar]

# main_genre kolonunu 1 ve 0'lara dönüştür
df_model = pd.get_dummies(df_model, columns=['main_genre'], drop_first=True)

#  EĞİTİM/TEST AYRIMI (Rapor Bölüm 8 - %80 / %20) 
print("6. Veri seti %80 Eğitim ve %20 Test olarak bölünüyor...")
X = df_model.drop('imdb_score', axis=1)
y = df_model['imdb_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Eğitimde oluşan sütunların listesini kaydedelim (Daha sonra arayüzde kullanmak için)
model_kolonlari = list(X.columns)

# NORMALİZASYON / STANDARTLAŞTIRMA
print("7. Veriler StandardScaler ile aynı ölçeğe çekiliyor...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ADIM 7: MODEL EĞİTİMLERİ VE METRİKLER 

# TEMEL  MODEL: ÇOKLU DOĞRUSAL REGRESYON
print("\n--- ÇOKLU DOĞRUSAL REGRESYON SONUÇLARI (Baseline) ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, lr_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, lr_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.3f}")
print(f"R2 Skoru: {r2_score(y_test, lr_pred):.3f}")

#  ASIL MODEL: RANDOM FOREST
print("\n--- RASTGELE ORMAN (RANDOM FOREST) SONUÇLARI (Ana Model) ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, rf_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, rf_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.3f}")
print(f"R2 Skoru: {r2_score(y_test, rf_pred):.3f}")

# DOSYALARIN KAYDEDİLMESİ
print("\n8. Asıl Model (Random Forest) ve Hafıza Dosyaları Diske Kaydediliyor...")
joblib.dump(rf_model, 'yapay_zeka_modeli.pkl')
joblib.dump(scaler, 'scaler.pkl') # Standartlaştırma motorunu da kaydetmeliyiz!
joblib.dump(model_kolonlari, 'model_kolonlari.pkl')
joblib.dump(dir_scores, 'yonetmen_skorlari.pkl')
joblib.dump(act_scores, 'oyuncu_skorlari.pkl')

print("✅ Kurşun geçirmez Pipeline başarıyla tamamlandı! Tüm dosyalar hazır.")