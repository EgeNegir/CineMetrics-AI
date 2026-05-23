import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veriyi yükle
df = pd.read_csv('temiz_film_verisi.csv')

# 2.İsimleri kariyer puanlarına çeviriyoruz
# Bu sayede Isı Haritası bu kolonları görebilecek
df['director_avg_score'] = df.groupby('director')['imdb_score'].transform('mean')
df['actor_avg_score'] = df.groupby('lead_actor')['imdb_score'].transform('mean')

# 3. Sadece anlamlı sayısal kolonları seç
sayisal_kolonlar = ['budget', 'revenue', 'runtime', 'imdb_score', 'director_avg_score', 'actor_avg_score']
corr_matrix = df[sayisal_kolonlar].corr()

# 4. Isı Haritasını Çiz 
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Film Başarı Faktörleri Korelasyon Haritası (Gelişmiş)')
plt.show()