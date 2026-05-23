import pandas as pd
import ast
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:senin_sifren@localhost:5432/imdb_proje')
df = pd.read_sql("SELECT * FROM v_master_movie_data", engine)

def bilgiyi_ayikla(obj, tip='Director'):
    liste = ast.literal_eval(obj)
    if tip == 'Director':
        for i in liste:
            if i.get('job') == 'Director': return i.get('name')
    elif tip == 'Lead':
        # Cast listesindeki ilk oyuncuyu (order 0) alıyoruz
        if len(liste) > 0: return liste[0].get('name')
    return "Unknown"

print("Yönetmen ve Başrol Oyuncusu bilgileri ayıklanıyor...")
df['director'] = df['crew'].apply(lambda x: bilgiyi_ayikla(x, 'Director'))
df['lead_actor'] = df['cast'].apply(lambda x: bilgiyi_ayikla(x, 'Lead'))

# Tür bilgisini de alalım
def ilk_tur(obj):
    liste = ast.literal_eval(obj)
    return liste[0]['name'] if len(liste) > 0 else "Unknown"

df['main_genre'] = df['genres'].apply(ilk_tur)

# Temiz tabloyu kaydedelim
df_clean = df[['movie_id', 'title', 'budget', 'revenue', 'runtime', 'imdb_score', 'main_genre', 'director', 'lead_actor']]
df_clean.to_csv('temiz_film_verisi.csv', index=False)
print("Başrol oyuncusu eklendi ve 'temiz_film_verisi.csv' güncellendi!")