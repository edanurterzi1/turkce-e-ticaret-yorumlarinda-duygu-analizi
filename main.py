import argparse
import os
import random
import re
import numpy as np
import torch
import csv
import pandas as pd
from datetime import datetime
from collections import Counter

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
import evaluate
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

matplotlib.use('Agg')


from src.utils import TrainingLoggerCallback, set_seed
from src.data_loader import robust_load_csv_to_dataset

# ---------------------- MAIN ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, 
                        default="C:/Users/edanu/Desktop/tez/e-ticaret_urun_yorumlari.csv") 
    parser.add_argument("--model", type=str, default="dbmdz/convbert-base-turkish-mc4-uncased")
    parser.add_argument("--output", type=str, default=f"./ConvBert_uncased1{datetime.now().strftime('%Y%m%d_%H%M')}")
    parser.add_argument("--max_length", type=int, default=53)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output, exist_ok=True)

    print(f"CSV (Robust) yükleniyor: {args.data_csv}")
    full_ds, class_weights_dict = robust_load_csv_to_dataset(args.data_csv)
    print("Toplam örnek sayısı (sadece 0 ve 1'ler):", len(full_ds))

    # Veri seti dağılım grafiği oluştur
    print("\n" + "="*50)
    print("VERİ SETİ DAĞILIM GRAFİĞİ OLUŞTURULUYOR")
    print("="*50)
    
    # Tüm veri setindeki sınıf dağılımı
    full_labels = [int(x) for x in full_ds["durum"]]
    class_counts_full = Counter(full_labels)
    
    plt.figure(figsize=(10, 6))
    labels = ['Olumsuz (0)', 'Olumlu (1)']
    counts = [class_counts_full.get(0, 0), class_counts_full.get(1, 0)]
    colors = ['#ff6b6b', '#51cf66']
    
    bars = plt.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Yorum Sayısı', fontsize=12, fontweight='bold')
    plt.xlabel('Sınıf', fontsize=12, fontweight='bold')
    plt.title('Veri Seti Sınıf Dağılımı (Tüm Veri)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Değerleri çubukların üzerine yaz
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    distribution_path = os.path.join(args.output, "dataset_distribution.png")
    plt.savefig(distribution_path, dpi=150)
    plt.close()
    print(f"Veri seti dağılım grafiği kaydedildi: {distribution_path}")
    print(f"Olumsuz (0) yorum sayısı: {class_counts_full.get(0, 0)}")
    print(f"Olumlu (1) yorum sayısı: {class_counts_full.get(1, 0)}")

    ds_train_temp = full_ds.train_test_split(test_size=0.2, seed=42)
    ds_val_test = ds_train_temp["test"].train_test_split(test_size=0.5, seed=42)
    train_ds = ds_train_temp["train"]
    val_ds = ds_val_test["train"]
    test_ds = ds_val_test["test"]
    
    # Tokenize edilmemiş test_ds'i sakla (inference time için)
    test_ds_original = test_ds

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def preprocess(batch):
        return tokenizer(batch["metin"], truncation=True, padding="max_length", max_length=args.max_length)
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=[])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=[])
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=[])

    if "labels" not in train_tok.column_names:
        train_tok = train_tok.rename_column("durum", "labels")
        val_tok = val_tok.rename_column("durum", "labels")
        test_tok = test_tok.rename_column("durum", "labels")

    cols_to_keep = ["input_ids","attention_mask","labels"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in cols_to_keep])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in cols_to_keep])
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in cols_to_keep])

    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")
    test_tok.set_format(type="torch")

    label_values = sorted(set(int(x) for x in full_ds["durum"]))
    num_labels = max(label_values) + 1
    print("Tespit edilen etiket sayısı:", num_labels)
    
    # Model Seçimi ve Eğitimi
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    
    # Sınıf ağırlıklarını model loss'una uygula
    if class_weights_dict:
        label_ids = sorted(set(int(x) for x in full_ds["durum"]))

        class_weights = torch.tensor(
            [class_weights_dict[label] for label in label_ids],
            dtype=torch.float32
        )
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        print(f"Sınıf ağırlıkları model loss'una uygulanıyor: {class_weights_dict}")
    else:
        class_weights = None

    # Model Değerlendirme Metrikleri
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        precision_weighted = precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"]
        recall_weighted = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]
        
        # Binary için per-class metrikler
        precision_per_class = precision_metric.compute(predictions=preds, references=labels, average=None)["precision"]
        recall_per_class = recall_metric.compute(predictions=preds, references=labels, average=None)["recall"]
        f1_per_class = f1_metric.compute(predictions=preds, references=labels, average=None)["f1"]
        
        return {
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "precision_0": precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
            "precision_1": precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
            "recall_0": recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
            "recall_1": recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
            "f1_0": f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
            "f1_1": f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        }

    # Model Eğitimi: Optimize edilmiş hiperparametreler
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        weight_decay=0.01,  # L2 regularization
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_steps=100,  # Learning rate warmup
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",  # En iyi modeli F1'e göre seç
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=0, 
        push_to_hub=False,
        
    )

    # CALLBACK'LER
    logger_callback = TrainingLoggerCallback(save_dir=args.output)
    
    # Early Stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # 3 epoch iyileşme olmazsa durdur
        early_stopping_threshold=0.001,  # Minimum iyileşme eşiği
    )

    # Custom loss function with class weights
    if class_weights is not None:
        from torch.nn import CrossEntropyLoss
        
        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[logger_callback, early_stopping],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[logger_callback, early_stopping],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eğitim başlıyor... Cihaz: {device}")
    trainer.train()

    print("Eğitim tamamlandı. En iyi model (validation setine göre) yüklendi.")

    print("\n" + "="*50)
    print("İP4 - NİHAİ DEĞERLENDİRME (TEST SETİ)")
    print("="*50)

    test_results = trainer.evaluate(eval_dataset=test_tok)
    print("Test seti sonuçları (Binary):")
    print(test_results)

    # Confusion Matrix ve Classification Report
    print("\n" + "="*50)
    print("CONFUSION MATRIX VE DETAYLI METRİKLER")
    print("="*50)
    
    # Test seti üzerinde tahmin yap
    predictions = trainer.predict(test_tok)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # ROC Eğrisi ve AUC
    print("\n" + "="*50)
    print("İP4 - ROC EĞRİSİ VE AUC HESAPLANIYOR")
    print("="*50)
    # Softmax ile pozitif sınıf (1) olasılığını al
    logits = predictions.predictions
    logits = logits - np.max(logits, axis=1, keepdims=True)  # numerik stabilite

    y_scores = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    if y_scores.shape[1] != 2:
        raise ValueError("ROC-AUC sadece binary sınıflandırma için hesaplanır.")

    y_prob_positive = y_scores[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (TPR - Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Eğrisi', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.5)

    roc_path = os.path.join(args.output, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    print(f"AUC (Area Under the Curve) değeri: {roc_auc:.4f}")
    print(f"ROC Eğrisi görseli kaydedildi: {roc_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Confusion Matrix görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Olumsuz (0)', 'Olumlu (1)'],
                yticklabels=['Olumsuz (0)', 'Olumlu (1)'])
    plt.title('Confusion Matrix - Test Seti', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix kaydedildi: {cm_path}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, 
                                  target_names=['Olumsuz (0)', 'Olumlu (1)'],
                                  output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Olumsuz (0)', 'Olumlu (1)']))
    
    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Sınıf bazlı accuracy hesapla (her sınıf için doğru tahmin / toplam)
    accuracy_per_class = []
    for i in range(len(support)):
        correct = cm[i, i]  # Confusion matrix'te diagonal değerler
        total = support[i]
        class_accuracy = correct / total if total > 0 else 0.0
        accuracy_per_class.append(class_accuracy)
    
    print("\nSınıf Bazlı Metrikler:")
    print(f"Olumsuz (0) -  Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Support: {support[0]}")
    print(f"Olumlu (1)  -  Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Support: {support[1]}")
    
    # Hata analizi: Yanlış sınıflandırılan örnekler
    errors = []
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
           
            errors.append({
                'index': i,
                'true_label': int(true_label),
                'pred_label': int(pred_label),
                'error_type': 'Olumlu->Olumsuz' if true_label == 1 else 'Olumsuz->Olumlu'
            })
    
    print(f"\nToplam Hata Sayısı: {len(errors)}")
    error_types = Counter([e['error_type'] for e in errors])
    print("Hata Türleri:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    # Sonuçları dosyaya kaydet
    results_path = os.path.join(args.output, "final_test_results_binary.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("İP4 - MODEL DEĞERLENDİRME RAPORU\n")
        f.write("="*60 + "\n\n")
        
        f.write("GENEL METRİKLER:\n")
        f.write("-" * 60 + "\n")
        for key, value in test_results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("ROC EĞRİSİ VE AUC DEĞERİ:\n")
        f.write("-" * 60 + "\n")
        f.write(f"AUC (Area Under the Curve): {roc_auc:.4f}\n")
        f.write(f"ROC eğrisi görseli: {roc_path}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 60 + "\n")
        f.write(f"                Tahmin\n")
        f.write(f"              Olumsuz  Olumlu\n")
        f.write(f"Gerçek Olumsuz   {cm[0][0]:4d}    {cm[0][1]:4d}\n")
        f.write(f"       Olumlu    {cm[1][0]:4d}    {cm[1][1]:4d}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SINIF BAZLI METRİKLER:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Olumsuz (0) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Support: {support[0]}\n")
        f.write(f"Olumlu (1)  - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Support: {support[1]}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("HATA ANALİZİ:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Toplam Hata Sayısı: {len(errors)}\n")
        f.write("Hata Türleri:\n")
        for error_type, count in error_types.items():
            f.write(f"  {error_type}: {count}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(y_true, y_pred, 
                                      target_names=['Olumsuz (0)', 'Olumlu (1)']))
    
    print(f"\nDetaylı değerlendirme raporu kaydedildi: {results_path}")
    print(f"Confusion Matrix görseli kaydedildi: {cm_path}")

    # Inference Time Hesaplama (tek bir yorum için)
    print("\n" + "="*50)
    print("ÇIKARIM SÜRESİ (INFERENCE TIME) HESAPLAMA")
    print("="*50)
    
    import time 
    model.eval()
    device = next(model.parameters()).device
    
    sample_text = "Bu ürün çok güzel ve kaliteli, kesinlikle tavsiye ederim."
    if len(test_ds_original) > 0:
        sample_text = test_ds_original[0]["metin"]
    
    sample_inputs = tokenizer(sample_text, truncation=True, padding="max_length", 
                               max_length=args.max_length, return_tensors="pt")
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    
    num_runs = 100 
    
    print(f"Isınma turu (20 iterasyon) başlatılıyor...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(**sample_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    print(f"Gerçek ölçüm ({num_runs} iterasyon) başlatılıyor...")
    inference_times = []
    
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(**sample_inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000) 

    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    print(f"\nÖrnek yorum: \"{sample_text[:100]}...\"")
    print(f"Çıkarım süresi ölçümü ({num_runs} çalıştırma):")
    print(f"  Ortalama: {avg_inference_time:.4f} ms")
    print(f"  Standart Sapma: {std_inference_time:.4f} ms")
    print(f"  Minimum: {min_inference_time:.4f} ms")
    print(f"  Maksimum: {max_inference_time:.4f} ms")
    print(f"  Tek bir yorum için çıkarım süresi: {avg_inference_time:.4f} ms")
    
    # Sonuçları dosyaya ekle
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("ÇIKARIM SÜRESİ (INFERENCE TIME):\n")
        f.write("-" * 60 + "\n")
        f.write(f"Örnek yorum: \"{sample_text[:100]}...\"\n")
        f.write(f"Çıkarım süresi ölçümü ({num_runs} çalıştırma):\n")
        f.write(f"  Ortalama: {avg_inference_time:.4f} ms\n")
        f.write(f"  Standart Sapma: {std_inference_time:.4f} ms\n")
        f.write(f"  Minimum: {min_inference_time:.4f} ms\n")
        f.write(f"  Maksimum: {max_inference_time:.4f} ms\n")
        f.write(f"  Tek bir yorum için çıkarım süresi: {avg_inference_time:.4f} ms\n")
    
    print(f"\nÇıkarım süresi bilgileri rapora eklendi: {results_path}")

    trainer.save_model(args.output)
    print(f"\nEğitim bitti, model şuraya kaydedildi: {args.output}")


if __name__ == "__main__":
    main()