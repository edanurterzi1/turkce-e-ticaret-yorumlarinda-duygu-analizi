import csv
import pandas as pd
from datasets import Dataset

# Proje içi bağımlılıklar
from .data_preprocessing import preprocess_text_column, check_class_imbalance, calculate_class_weights

def sniff_separator(sample_bytes):
    try:
        sample = sample_bytes.decode("utf-8")
    except:
        sample = sample_bytes.decode("cp1254", errors="replace")
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        if "\t" in sample:
            return "\t"
        if ";" in sample:
            return ";"
        return None


def try_read_with_seps(path, seps):
    last_exc = None
    for sep in seps:
        try:
            try:
                df = pd.read_csv(path, sep=sep, encoding="utf-8", engine='python', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(path, sep=sep, encoding="cp1254", engine='python', on_bad_lines='skip')
            print(f"Başarıyla okundu, ayraç: {repr(sep)}")
            return df
        except Exception as e:
            print(f"Ayraç {repr(sep)} ile denendi, hata: {e}")
            last_exc = e
    try:
        print("Standart ayraçlar başarısız, 'sep=None' ile deneniyor...")
        try:
            df = pd.read_csv(path, sep=None, engine='python', encoding="utf-8", on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(path, sep=None, engine='python', encoding="cp1254", on_bad_lines='skip')
        print("Başarıyla okundu, ayraç: Otomatik (sep=None)")
        return df
    except Exception as e:
        raise last_exc if last_exc is not None else e


def robust_load_csv_to_dataset(path):
    with open(path, "rb") as f:
        sample = f.read(8192)
    guessed = sniff_separator(sample)

    seps_to_try = []
    if guessed:
        seps_to_try.append(guessed)
    for sep in [";", "\t", ",", "|"]:
        if sep not in seps_to_try:
            seps_to_try.append(sep)

    print("Denenecek ayraçlar (öncelik sırasıyla):", [repr(s) for s in seps_to_try])
    df = try_read_with_seps(path, seps_to_try)

    df.columns = [c.strip().lower() for c in df.columns.astype(str)]
    print("DataFrame başarıyla yüklendi. Şekil:", df.shape)
    print("Bulunan sütunlar (normalize edilmiş):", list(df.columns[:50]))

    if df.shape[1] == 1:
        col0 = df.columns[0]
        for sep in ["\t",";",",","|"]:
            splitted = df[col0].astype(str).str.split(sep, n=1, expand=True)
            if splitted.shape[1] > 1:
                df = splitted
                if splitted.shape[1] == 2:
                    df.columns = ["metin", "durum"]
                else:
                    df.columns = [f"c{i}" for i in range(splitted.shape[1])]
                print("Tek sütun otomatik bölündü", sep, "-> yeni sütunlar:", df.columns.tolist())
                break

    cols = [c.lower() for c in df.columns]

    text_col = None
    label_col = None

    if 'metin' in cols:
        text_col = df.columns[cols.index('metin')]
    else:
        for alt in ["text","yorum","review","comment","yorum_text","yorumlar"]:
            if alt in cols:
                text_col = df.columns[cols.index(alt)]
                break

    if 'durum' in cols:
        label_col = df.columns[cols.index('durum')]
    else:
        for alt in ["label","labels","sentiment","rating","score","dur"]:
            if alt in cols:
                label_col = df.columns[cols.index(alt)]
                break

    if text_col is None or label_col is None:
        raise ValueError(f"Metin ('metin') veya Etiket ('durum') sütunları bulunamadı. Bulunan sütunlar: {list(df.columns)}")

    print(f"Metin sütunu olarak '{text_col}' bulundu.")
    print(f"Etiket sütunu olarak '{label_col}' bulundu.")

    df = df.rename(columns={text_col: "metin", label_col: "durum"})

    df = df.dropna(subset=["metin","durum"])
    df["metin"] = df["metin"].astype(str)
    df = df[df["metin"].str.len() > 0]

    try:
        df["durum"] = df["durum"].astype(int)
    except Exception:
        uniques = sorted(df["durum"].unique().tolist())
        mapping = {v:i for i,v in enumerate(uniques)}
        df["durum"] = df["durum"].map(mapping)
        print("Metin etiketleri sayısala çevrildi:", mapping)

    print(f"Filtreleme öncesi örnek sayısı: {len(df)}")
    df = df[df["durum"] != 2]
    print(f"Nötr (durum == 2) veriler kaldırıldı. Kalan (binary) örnek sayısı: {len(df)}")

    # Veri Ön İşleme: Metin temizleme
    df = preprocess_text_column(df, text_col="metin")
    
    # Sınıf dengesizliği kontrolü
    is_imbalanced, class_counts = check_class_imbalance(df, label_col="durum")
    
    # Sınıf ağırlıklarını hesapla (model eğitiminde kullanılacak)
    class_weights_dict = calculate_class_weights(df, label_col="durum")

    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return ds, class_weights_dict
