import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # HTML etiketlerini kaldır
    text = re.sub(r'<[^>]+>', '', text)
    
    # URL'leri kaldır
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Email adreslerini kaldır
    text = re.sub(r'\S+@\S+', '', text)
    
    # Tekrarlı harfleri normalize et (ör: "çokkkk" -> "çok")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Küçük harfe çevir
    text = text.lower().strip()
    
    return text

def preprocess_text_column(df, text_col="metin"):
    # """
    # DataFrame'deki metin sütununu temizle ve normalize et
    # """
    print(f"Veri ön işleme başlıyor: {len(df)} örnek")
    df[text_col] = df[text_col].apply(clean_text)
    
    # Boş metinleri kaldır
    before = len(df)
    df = df[df[text_col].str.len() > 0]
    after = len(df)
    print(f"Boş metinler kaldırıldı: {before - after} örnek")
    
    return df

def check_class_imbalance(df, label_col="durum"):
    """
    İP2 - Sınıf dengesizliği kontrolü
    """
    class_counts = df[label_col].value_counts().sort_index()
    total = len(df)
    
    print("\n" + "="*50)
    print("SINIF DENGESİZLİĞİ ANALİZİ")
    print("="*50)
    for label, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"Sınıf {label}: {count} örnek ({percentage:.2f}%)")
    
    # Dengesizlik oranı (en çok / en az)
    if len(class_counts) > 1:
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\nDengesizlik Oranı: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 2.0:
            print("⚠️  UYARI: Sınıf dengesizliği tespit edildi!")
            return True, class_counts
        else:
            print("✓ Sınıf dengesi kabul edilebilir seviyede.")
            return False, class_counts
    
    return False, class_counts

def balance_dataset(df, label_col="durum", method="class_weight"):
    """
    İP2 - Sınıf dengesizliği düzeltme
    method: "class_weight" (model eğitiminde), "oversample" (Random Oversampling)
    """
    class_counts = df[label_col].value_counts()
    
    if method == "oversample":
        # Random Oversampling
        from sklearn.utils import resample
        
        max_count = class_counts.max()
        balanced_dfs = []
        
        for label in class_counts.index:
            label_df = df[df[label_col] == label]
            if len(label_df) < max_count:
                # Oversample
                label_df_resampled = resample(
                    label_df,
                    replace=True,
                    n_samples=max_count,
                    random_state=42
                )
                balanced_dfs.append(label_df_resampled)
            else:
                balanced_dfs.append(label_df)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nRandom Oversampling uygulandı:")
        print(f"Önceki örnek sayısı: {len(df)}")
        print(f"Sonraki örnek sayısı: {len(df_balanced)}")
        print(f"Yeni sınıf dağılımı:\n{df_balanced[label_col].value_counts().sort_index()}")
        
        return df_balanced
    else:
        # class_weight kullanılacak (model eğitiminde)
        return df

def calculate_class_weights(df, label_col="durum"):
    """
    Sınıf ağırlıklarını hesapla (class_weight için)
    """
    class_counts = df[label_col].value_counts().sort_index()
    total = len(df)
    
    weights = {}
    for label in class_counts.index:
        weights[label] = total / (len(class_counts) * class_counts[label])
    
    print(f"\nHesaplanan sınıf ağırlıkları: {weights}")
    return weights
