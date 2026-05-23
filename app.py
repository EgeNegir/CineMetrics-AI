import streamlit as st
import joblib
import pandas as pd
import time

# Sayfa ayarları (Kurumsal başlık ve geniş ekran modu)
st.set_page_config(page_title="CineMetrics AI - IMDb Puan Tahmin Sistemi", page_icon="🎬", layout="wide")

# --- ÖZEL CSS ENJEKSİYONU (Karanlık ve Fütüristik Tema) ---
st.markdown("""
<style>
    /* Ana Arka Plan, Metinler ve Sinematik Görsel */
    .stApp {
        background-image: linear-gradient(rgba(5, 5, 5, 0.60), rgba(5, 5, 5, 0.60)), url("https://semt77.com/wp-content/uploads/2021/03/Sinema-Salonu.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #e0e3e5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sol Menü (Sidebar) Yapısı ve Üst Boşluğu Silme */
    [data-testid="stSidebar"] {
        background-color: #101415 !important;
        border-right: 1px solid #3c494e;
    }
    
    /* Kenar çubuğunun tepesindeki o devasa boşluğu yok ediyoruz */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important; 
    }
    
    /* Girdi Kutuları (Modern ve Neon Odaklı) */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #121212 !important;
        color: #a8e8ff !important;
        border: 1px solid #3c494e !important;
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #a8e8ff !important;
        box-shadow: 0 0 10px rgba(168, 232, 255, 0.2) !important;
    }
    
    /* Canlı Mavi Aksiyon Butonu */
    .stButton>button {
        background-color: #a8e8ff !important;
        color: #003642 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: #dfb7ff !important;
        color: #2d004f !important;
        box-shadow: 0 0 20px rgba(223, 183, 255, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Puan Gösterge Paneli */
    [data-testid="stMetricValue"] {
        color: #dfb7ff !important;
        font-size: 4rem !important;
        text-shadow: 0 0 20px rgba(223, 183, 255, 0.3);
    }
    
    h1, h2, h3 {
        color: #a8e8ff !important;
    }
</style>
""", unsafe_allow_html=True)


# --- ARKA PLAN: VERİ VE MODEL YÜKLEMESİ ---
@st.cache_resource 
def model_yukle():
    model = joblib.load('yapay_zeka_modeli.pkl')
    scaler = joblib.load('scaler.pkl')
    kolonlar = joblib.load('model_kolonlari.pkl')
    yonetmenler = joblib.load('yonetmen_skorlari.pkl')
    oyuncular = joblib.load('oyuncu_skorlari.pkl')
    return model, scaler, kolonlar, yonetmenler, oyuncular

model, scaler, model_kolonlari, yonetmen_hafizasi, oyuncu_hafizasi = model_yukle()

yonetmen_isimleri = ["✨ Farklı Bir İsim Gireceğim"] + sorted(list(yonetmen_hafizasi.keys()))
oyuncu_isimleri = ["✨ Farklı Bir İsim Gireceğim"] + sorted(list(oyuncu_hafizasi.keys()))

# --- SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    sol_bosluk, orta_sutun, sag_bosluk = st.columns([1, 1.5, 1])
    with orta_sutun:
        st.image("logo.png", use_container_width=True)
    
    st.markdown("<h2 style='text-align: center; color: #a8e8ff; margin-top: -15px;'>CineMetrics AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("📡 **Sistem Durumu:** Çevrimiçi")
    st.markdown("🧠 **Algoritma:** Rastgele Orman (Random Forest)")
    st.markdown("📊 **Veri Kümesi:** 5000+ Sinema Filmi")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("© 2026 CineMetrics AI Tahmin Altyapısı.")

# --- ANA EKRAN ---
st.title("CineMetrics AI Film Başarı Tahmin Motoru")
st.markdown("Sinema projelerine ait teknik parametreleri girerek gerçek zamanlı yapay zeka IMDb tahminleri üretebilirsiniz.")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🗄️ Proje Girdi Parametreleri")
    secilen_yonetmen = st.selectbox("Yönetmen Adı", yonetmen_isimleri, index=yonetmen_isimleri.index("Christopher Nolan") if "Christopher Nolan" in yonetmen_isimleri else 0)
    if secilen_yonetmen == "✨ Farklı Bir İsim Gireceğim":
        yonetmen_adi = st.text_input("Lütfen Yönetmen Adını Yazın:", "Ege")
    else:
        yonetmen_adi = secilen_yonetmen

    secilen_oyuncu = st.selectbox("Başrol Oyuncusu", oyuncu_isimleri, index=oyuncu_isimleri.index("Leonardo DiCaprio") if "Leonardo DiCaprio" in oyuncu_isimleri else 0)
    if secilen_oyuncu == "✨ Farklı Bir İsim Gireceğim":
        oyuncu_adi = st.text_input("Lütfen Oyuncu Adını Yazın:", "Halil")
    else:
        oyuncu_adi = secilen_oyuncu

    ana_tur = st.selectbox("Filmin Ana Türü", ['Action', 'Drama', 'Comedy', 'Horror', 'Sci-Fi', 'Romance', 'Thriller'])
    
    col_b, col_s = st.columns(2)
    with col_b:
        butce = st.number_input("Tahmini Bütçe ($)", min_value=10000, value=150000000, step=1000000)
    with col_s:
        sure = st.number_input("Süre (Dakika)", min_value=60, max_value=240, value=140)

with col2:
    st.markdown("### ✨ Yapay Zekâ Projeksiyonu")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Görüntünün zıplamaması için dinamik bir içerik alanı (placeholder) oluşturduk
    icerik_alani = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Butonu sağ sütunun en altına kalıcı olarak yerleştirdik
    tahmin_baslat = st.button("🚀 TAHMİN PROTOKOLÜNÜ BAŞLAT", use_container_width=True)
    
    # Eğer butona basıldıysa içerik alanına sonuçları yazdır
    if tahmin_baslat:
        with icerik_alani.container():
            with st.spinner('CineMetrics Core Analiz Ediyor... 🧠'):
                time.sleep(1.2)
                
                yonetmen_temiz = yonetmen_adi.strip()
                oyuncu_temiz = oyuncu_adi.strip()
                
                yonetmen_skoru = yonetmen_hafizasi.get(yonetmen_temiz, 6.40)
                oyuncu_skoru = oyuncu_hafizasi.get(oyuncu_temiz, 6.40)

                yeni_film = pd.DataFrame([{
                    'budget': butce, 'runtime': sure,
                    'director_avg_score': yonetmen_skoru, 'actor_avg_score': oyuncu_skoru, 'main_genre': ana_tur
                }])
                
                yeni_film_kodlanmis = pd.get_dummies(yeni_film, columns=['main_genre'])
                for col in model_kolonlari:
                    if col not in yeni_film_kodlanmis.columns:
                        yeni_film_kodlanmis[col] = 0
                yeni_film_kodlanmis = yeni_film_kodlanmis[model_kolonlari]
                yeni_film_olcekli = scaler.transform(yeni_film_kodlanmis)
                
                tahmin = model.predict(yeni_film_olcekli)[0]
                
                st.metric(label="Tahmin Edilen IMDb Puanı", value=f"{tahmin:.3f}", delta="Model Kararlılığı Yüksek")
                
                with st.expander("⚙️ Sistem Analitik Raporu (Faktör Etkileri)"):
                    st.write("Modelin karar mekanizmasında kullandığı ağırlıklı ortalamalar:")
                    
                    if yonetmen_temiz in yonetmen_hafizasi:
                        st.success(f"🎥 **Yönetmen:** {yonetmen_temiz} (Kariyer Başarı Oranı: {yonetmen_skoru:.2f})")
                    else:
                        st.warning(f"⚠️ **Yönetmen:** '{yonetmen_temiz}' yeni kayıt. Sektör taban puanı (6.40) atandı.")

                    if oyuncu_temiz in oyuncu_hafizasi:
                        st.success(f"⭐ **Oyuncu:** {oyuncu_temiz} (Kariyer Başarı Oranı: {oyuncu_skoru:.2f})")
                    else:
                        st.warning(f"⚠️ **Oyuncu:** '{oyuncu_temiz}' yeni kayıt. Sektör taban puanı (6.40) atandı.")
                        
                    st.info(f"💸 **Finansal Ölçek:** ${butce:,} | ⏱️ **Zaman Kısıtı:** {sure} Dakika")
                    
    # Eğer butona henüz basılmadıysa içerik alanına bilgi mesajını yazdır
    else:
        with icerik_alani.container():
            st.info("Projeksiyonu başlatmak için sol taraftaki formu doldurup aşağıdaki butona tıklayın.")