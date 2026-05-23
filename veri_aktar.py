import pandas as pd
from sqlalchemy import create_engine, text

# 1. Veritabanı bağlantımızı kuruyoruz (Şifreni buraya tekrar yaz)
engine = create_engine('postgresql://postgres:senin_sifren@localhost:5432/imdb_proje')

# 2. Dosya yolların (Aynı kalacak)
movies_path = r"C:\Users\Ege\Desktop\IMDb_yapayzeka\archive\tmdb_5000_movies.csv"
credits_path = r"C:\Users\Ege\Desktop\IMDb_yapayzeka\archive\tmdb_5000_credits.csv"

# 3. YENİ EKLENEN KISIM: Eski hatalı tabloları birbirine olan bağlarıyla (CASCADE) birlikte zorla siliyoruz
print("Eski tablolar ve Foreign Key bağları temizleniyor...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS credits, movies CASCADE;"))
    conn.commit()

# 4. Verileri okuma ve aktarma
print("Pandas, o bozuk CSV dosyalarını temizleyerek okuyor...")
df_movies = pd.read_csv(movies_path)
df_credits = pd.read_csv(credits_path)

print("Movies tablosu veritabanına mermi gibi çakılıyor...")
df_movies.to_sql('movies', engine, if_exists='replace', index=False)

print("Credits tablosu veritabanına mermi gibi çakılıyor...")
df_credits.to_sql('credits', engine, if_exists='replace', index=False)

print("GÖREV TAMAMLANDI! pgAdmin'e gidip verilerini kontrol edebilirsin.")