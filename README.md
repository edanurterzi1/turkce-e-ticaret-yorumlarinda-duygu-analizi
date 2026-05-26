# E-Ticaret Ürün Yorumları Duygu Analizi ve Web API Entegrasyonu


<a id="türkçe"></a>

Bu proje, e-ticaret sitelerinden elde edilen Türkçe müşteri yorumlarını Doğal Dil İşleme (NLP) yöntemleriyle analiz ederek **olumlu** veya **olumsuz** olarak sınıflandırmayı amaçlayan bir tez projesidir.

## 📌 Proje Hakkında
Bu çalışmada, derin öğrenme tabanlı Doğal Dil İşleme (NLP) yöntemleri kullanılarak API destekli bir duygu analizi (sentiment analysis) sistemi geliştirilmiştir. Projenin çekirdeğini Transformer mimarisi (BERT tabanlı modeller) oluşturmaktadır.
Proje geliştirme sürecinde çeşitli dil modelleri (farklı BERT varyasyonları) ve farklı hiperparametre kombinasyonları (öğrenme hızı, epoch sayısı, batch size vb.) ile kapsamlı deneyler yapılmıştır. Elde edilen sonuçlar karşılaştırılarak, **en yüksek F1 skoru oranını veren** model mimarisi seçilmiş ve canlı sisteme (API) entegre edilmiştir.

**📈 Model Başarısı ve Yeniden Üretilebilirlik (Reproducibility) Notu:**
Bu resmi tez çalışması kapsamında optimize edilen en iyi model **%98.77** doğruluk (accuracy) oranına ulaşmıştır.

## 📊 Veri Seti (Dataset)
Projede kullanılan veri seti Kaggle üzerinde yayınlanmıştır. Veri setine buradan ulaşabilirsiniz:
🔗 **(https://www.kaggle.com/datasets/mujdatcabuk/eticaret-urun-yorumlari)**

## 🚀 Özellikler
- **Veri Ön İşleme:** HTML etiketleri, URL'ler, e-posta adresleri temizlenir ve tekrarlayan harfler (örneğin "çooook" -> "çook") NLP'ye uygun şekilde normalize edilir.
- **Veri Seti Yönetimi:** Veri setindeki olumlu/olumsuz sınıf eşitsizliklerini gidermek için ağırlıklandırma (Class Weighting) yöntemi kullanılmıştır.
- **Modüler Mimari:** Veri yükleme, metin temizleme, model eğitimi ve API servis işlemleri profesyonel yazılım mühendisliği prensiplerine uygun olarak farklı modüllere ayrılmıştır.
- **Kapsamlı Değerlendirme:** ROC Eğrisi, Confusion Matrix, Sınıflandırma Raporları ve hata analizi (yanlış bilinen örneklerin tespiti) ile modelin performansı detaylıca incelenir.
- **REST API Desteği:** `FastAPI` kullanılarak geliştirilen servis, tekli veya çoklu (batch) tahmin isteklerine yanıt verebilecek şekilde tasarlanmıştır.

## 📂 Proje Yapısı
```text
.
├── src/
│   ├── data_loader.py        # Veri setini yükleme ve formatlama
│   ├── data_preprocessing.py # Metin temizleme ve sınıf dengesizliği hesaplamaları
│   └── utils.py              # Eğitim loglama callback'leri ve rastgelelik sabitleyiciler
├── main.py                   # Modelin ana eğitim betiği
├── api_service.py            # Eğitilen modelin sunulduğu FastAPI servisi
├── requirements.txt          # Gerekli Python kütüphaneleri
└── README.md                 # Proje tanıtım dosyası
```

## 🛠️ Kurulum
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz. Bir sanal ortam (virtual environment) kullanmanız önerilir:

```bash
# 1. Depoyu bilgisayarınıza klonlayın
git clone <repo_url>
cd <repo_klasoru>

# 2. Sanal ortam oluşturun ve aktifleştirin (Windows için)
python -m venv venv
venv\Scripts\activate

# 3. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

## 🧠 Model Eğitimi (Training)
Eğitim işlemini başlatmak için veri setinizin yolunu belirterek aşağıdaki komutu çalıştırabilirsiniz. Model eğitim bitiminde performans grafiklerini ve sonuç raporlarını otomatik oluşturacaktır.

```bash
python train.py --data_csv "e-ticaret_urun_yorumlari.csv" --epochs 4 --batch_size 64 --lr 2e-5
```

## 🌐 API Kullanımı (Serving)
Eğittiğiniz en iyi modeli canlıya almak için FastAPI sunucusunu aşağıdaki komutla başlatın:

```bash
python -m uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload
```
Sunucu çalıştıktan sonra tarayıcınızda `http://localhost:8000/docs` adresine giderek API'yi anında test edebilirsiniz.

## 📝 Teşekkür & Referans
Bu çalışma bir lisans/yüksek lisans tez projesi kapsamında **[Edanur Terzi]** ve **[Melike Karaman]** tarafından ortaklaşa geliştirilmiş olup, doğal dil işleme (NLP) alanında Türkçe metinlerin sınıflandırılmasına yönelik deneysel bulgular içermektedir.
