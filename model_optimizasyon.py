import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Veriler yükleniyor ve hazırlanıyor...")
df = pd.read_csv('temiz_film_verisi.csv')

# Öncekiyle aynı veri hazırlığı: Yönetmen skorlarını ekliyoruz
director_scores = df.groupby('director')['imdb_score'].mean().to_dict()
df['director_avg_score'] = df['director'].map(director_scores)
df = pd.get_dummies(df, columns=['main_genre'], drop_first=True)

X = df.drop(columns=['movie_id', 'title', 'director', 'imdb_score'])
y = df['imdb_score']
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n⚙️ RANDOM FOREST İÇİN EN İYİ AYARLAR ARANIYOR...")
print("(GridSearchCV devrede, bu işlem bilgisayarının hızına göre 1-2 dakika sürebilir)")

# Modelin deneyip en iyisini bulacağı ayar kombinasyonları (Grid)
param_grid = {
    'n_estimators': [100, 200],       # Ormandaki ağaç sayısı
    'max_depth': [10, 20, None],      # Ağaçların maksimum derinliği (Overfitting'i engeller)
    'min_samples_split': [2, 5, 10]   # Bir dalın ikiye ayrılması için gereken minimum veri
}

# Modeli ve Arama Motorunu (GridSearch) tanımlıyoruz
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# Aramayı başlat!
grid_search.fit(X_train, y_train)

# En iyi ayarları ve o ayarlarla çıkan test sonucunu alalım
best_rf = grid_search.best_estimator_
tahminler = best_rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, tahminler))
r2 = r2_score(y_test, tahminler)

print("\n🏆 OPTİMİZASYON TAMAMLANDI!")
print(f"Bulunan En İyi Ayarlar: {grid_search.best_params_}")
print(f"Yeni RMSE (Hata Payı): {rmse:.2f} Puan")
print(f"Yeni R² (Açıklanabilirlik): %{r2*100:.2f}")