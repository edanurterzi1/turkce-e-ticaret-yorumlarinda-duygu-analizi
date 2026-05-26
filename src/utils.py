import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainerCallback

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TrainingLoggerCallback(TrainerCallback):

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # epoch bazlı: [epoch, train_loss_at_epoch_end, eval_loss, eval_accuracy, eval_f1_weighted]
        self.epoch_data = []

        # step bazlı: [step, train_loss, maybe_eval_loss]
        self.step_data = []

        # yardımcı değişkenler
        self.current_epoch = 0
        self.last_train_loss = None  # on_log'ta güncellenir, on_evaluate'de epoch için kullanılır
        self.state_ref = None

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        # state referansını sakla ki on_train_end'de log_history erişelim
        self.state_ref = state

        # global adım
        step = getattr(state, "global_step", None)
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)

        # son train loss'u sakla (epoch sonu için)
        if train_loss is not None:
            self.last_train_loss = train_loss

        # step verisini ekle
        if step is not None and train_loss is not None:
            self.step_data.append([step, train_loss, eval_loss])

            # adım bazlı CSV kaydı (her logta güncellenir)
            df_steps = pd.DataFrame(self.step_data, columns=["step", "train_loss", "eval_loss"])
            steps_csv = os.path.join(self.save_dir, "training_log_steps.csv")
            try:
                df_steps.to_csv(steps_csv, index=False)
            except Exception:
                pass  # I/O hatası eğitim sürecini bozmasın

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        self.state_ref = state

        # Tüm olası metric isimlerini kontrol et
        eval_loss = logs.get("eval_loss", None)
        if eval_loss is None:
            eval_loss = logs.get("loss", None)
        
        eval_accuracy = logs.get("eval_accuracy", None)
        if eval_accuracy is None:
            eval_accuracy = logs.get("accuracy", None)
        
        # F1 için tüm olası isimleri kontrol et
        eval_f1 = logs.get("eval_f1_weighted", None)
        if eval_f1 is None:
            eval_f1 = logs.get("f1_weighted", None)
        if eval_f1 is None:
            eval_f1 = logs.get("eval_f1", None)
        if eval_f1 is None:
            eval_f1 = logs.get("f1", None)

        # Train loss için: önce logs'tan kontrol et, yoksa last_train_loss kullan
        train_loss = logs.get("train_loss", None)
        if train_loss is None:
            train_loss = self.last_train_loss
        
        # Eğer hala None ise, log_history'den bu epoch için train_loss'u bul
        if (train_loss is None or (isinstance(train_loss, float) and np.isnan(train_loss))) and getattr(state, "log_history", None):
            epoch_val = getattr(state, "epoch", None)
            if epoch_val is not None:
                # Bu epoch için log_history'den train_loss'u bul
                for entry in reversed(state.log_history):
                    entry_epoch = entry.get("epoch")
                    if entry_epoch is not None and abs(entry_epoch - epoch_val) < 0.001:
                        # "loss" var ve "eval_loss" yoksa bu train loss'tur
                        if "loss" in entry and "eval_loss" not in entry:
                            potential_train_loss = entry.get("loss")
                            if potential_train_loss is not None:
                                train_loss = potential_train_loss
                                self.last_train_loss = train_loss
                                break
                        # Veya açıkça "train_loss" olarak belirtilmişse
                        if "train_loss" in entry:
                            potential_train_loss = entry.get("train_loss")
                            if potential_train_loss is not None:
                                train_loss = potential_train_loss
                                self.last_train_loss = train_loss
                                break

        # epoch numarası olarak state.epoch (float) varsa onu kullan, yoksa current_epoch
        epoch_idx = None
        if getattr(state, "epoch", None) is not None:
            epoch_idx = state.epoch
        else:
            epoch_idx = self.current_epoch

        # epoch için kayıt ekle
        self.epoch_data.append([epoch_idx, train_loss, eval_loss, eval_accuracy, eval_f1])

        # epoch bazlı CSV kaydı
        df_epoch = pd.DataFrame(self.epoch_data, columns=[
            "epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_f1_weighted"
        ])
        epoch_csv = os.path.join(self.save_dir, "training_log.csv")
        try:
            df_epoch.to_csv(epoch_csv, index=False)
        except Exception as e:
            print(f"CSV yazma hatası: {e}")  # Hata mesajını görelim

    def on_epoch_end(self, args, state, control, **kwargs):

        # Son train_loss'u mevcut epoch için kaydet (eğer henüz kaydedilmemişse)
        if self.last_train_loss is not None and getattr(state, "epoch", None) is not None:
            epoch_val = state.epoch
            # Bu epoch için kayıt var mı kontrol et
            found = False
            for i, row in enumerate(self.epoch_data):
                if abs(row[0] - epoch_val) < 0.001 if isinstance(row[0], (int, float)) else row[0] == epoch_val:
                    # Eğer train_loss None ise güncelle
                    if row[1] is None or (isinstance(row[1], float) and (pd.isna(row[1]) if hasattr(pd, 'isna') else np.isnan(row[1]))):
                        self.epoch_data[i][1] = self.last_train_loss
                    found = True
                    break
            
            # Eğer bu epoch için kayıt yoksa, oluştur (eval henüz çağrılmamış olabilir)
            if not found:
                self.epoch_data.append([epoch_val, self.last_train_loss, None, None, None])
        
        # epoch sayaçını artır; on_evaluate genelde epoch sonunda çağrıldığı için
        # current_epoch'i burada artırıyoruz ki bir sonraki eval doğru epoch indexine sahip olsun.
        self.current_epoch += 1

    def on_train_end(self, args, state, control, **kwargs):

        # Eğitim bittiğinde:
        # - epoch bazlı tüm metrikleri tek grafikte (loss/accuracy/f1) kaydet
        # - step bazlı train loss (ve varsa eval_loss) grafiğini kaydet
   
        # Eğer epoch verisi boşsa veya eksikse, log_history'den toparlamayı dene (robust)
        try:
            if getattr(state, "log_history", None):
                # Logları epoch bazında bir sözlükte topla, en son logları sakla
                logs_by_epoch = {}
                for entry in state.log_history:
                    epoch_val = entry.get("epoch")
                    if epoch_val is not None:
                        if epoch_val not in logs_by_epoch:
                            logs_by_epoch[epoch_val] = {}
                        
                        # Aynı epoch'a ait tüm anahtarları birleştir (güncel olanı korur)
                        logs_by_epoch[epoch_val].update(entry)
                
                # Yeni ve güncel epoch verilerini oluştur
                self.epoch_data = []
                for epoch_val, logs in sorted(logs_by_epoch.items()):
                    
                    # Eval logları
                    eval_loss = logs.get("eval_loss")
                    eval_acc = logs.get("eval_accuracy") or logs.get("accuracy")
                    eval_f1 = logs.get("eval_f1_weighted") or logs.get("f1_weighted") or logs.get("eval_f1") or logs.get("f1")
                    
                    # Train logları
                    # Önce 'train_loss' anahtarını kontrol et, yoksa 'loss' (son adımı temsil eden) anahtarını kullan.
                    train_loss = logs.get("train_loss")
                    if train_loss is None or (isinstance(train_loss, float) and np.isnan(train_loss)):
                        train_loss = logs.get("loss")
                        
                    self.epoch_data.append([
                        epoch_val,
                        train_loss,
                        eval_loss,
                        eval_acc,
                        eval_f1,
                    ])
                    
        except Exception as e:
            print(f"log_history'den veri toplama hatası: {e}")

        # EPOCH BAZLI GRAFİK (loss, accuracy, f1)
        try:
            if len(self.epoch_data) > 0:
                df = pd.DataFrame(self.epoch_data, columns=[
                    "epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_f1_weighted"
                ])
                
                # NaN değerleri temizle ve epoch'a göre sırala
                df = df.sort_values("epoch")
                df = df.dropna(subset=["epoch"])  # epoch NaN olanları at

                plt.figure(figsize=(12, 7))
                plotted_any = False
                
                # Train Loss - NaN olmayan değerleri çiz
                if df["train_loss"].notna().any():
                    valid = df[df["train_loss"].notna()]
                    if len(valid) > 0:
                        plt.plot(valid["epoch"], valid["train_loss"], marker="o", label="Train Loss", linewidth=2)
                        plotted_any = True
                
                # Eval Loss - NaN olmayan değerleri çiz
                if df["eval_loss"].notna().any():
                    valid = df[df["eval_loss"].notna()]
                    if len(valid) > 0:
                        plt.plot(valid["epoch"], valid["eval_loss"], marker="s", label="Eval Loss", linewidth=2)
                        plotted_any = True
                
                # Eval Accuracy - NaN olmayan değerleri çiz
                if df["eval_accuracy"].notna().any():
                    valid = df[df["eval_accuracy"].notna()]
                    if len(valid) > 0:
                        plt.plot(valid["epoch"], valid["eval_accuracy"], marker="^", label="Eval Accuracy", linewidth=2)
                        plotted_any = True
                
                # Eval F1 - NaN olmayan değerleri çiz
                if df["eval_f1_weighted"].notna().any():
                    valid = df[df["eval_f1_weighted"].notna()]
                    if len(valid) > 0:
                        plt.plot(valid["epoch"], valid["eval_f1_weighted"], marker="d", label="Eval F1 (Weighted)", linewidth=2)
                        plotted_any = True

                if plotted_any:
                    plt.xlabel("Epoch", fontsize=12)
                    plt.ylabel("Value", fontsize=12)
                    plt.title("Training Logs (Loss - Accuracy - F1)", fontsize=14, fontweight="bold")
                    plt.legend(loc="best", fontsize=10)
                    plt.grid(True, alpha=0.3)
                    
                    # CSV'yi de kaydet
                    epoch_csv = os.path.join(self.save_dir, "training_log.csv")
                    df.to_csv(epoch_csv, index=False)
                    print(f"Epoch CSV kaydedildi: {epoch_csv}")

                    plot_path = os.path.join(self.save_dir, "training_plot_epoch.png")
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print(f"Epoch bazlı log grafiği kaydedildi: {plot_path}")
                    
                    # Training Loss vs Test Loss grafiği (overfit kontrolü için)
                    if df["train_loss"].notna().any() and df["eval_loss"].notna().any():
                        plt.figure(figsize=(10, 6))
                        train_valid = df[df["train_loss"].notna()]
                        eval_valid = df[df["eval_loss"].notna()]
                        
                        if len(train_valid) > 0:
                            plt.plot(train_valid["epoch"], train_valid["train_loss"], 
                                    marker="o", label="Training Loss", linewidth=2, color='#339af0')
                        if len(eval_valid) > 0:
                            plt.plot(eval_valid["epoch"], eval_valid["eval_loss"], 
                                    marker="s", label="Test Loss (Validation)", linewidth=2, color='#ff6b6b')
                        
                        plt.xlabel("Epoch", fontsize=12, fontweight='bold')
                        plt.ylabel("Loss", fontsize=12, fontweight='bold')
                        plt.title("Training Loss vs Test Loss (Overfit Kontrolü)", fontsize=14, fontweight="bold")
                        plt.legend(loc="best", fontsize=11)
                        plt.grid(True, alpha=0.3, linestyle='--')
                        
                        plot_loss_path = os.path.join(self.save_dir, "training_vs_test_loss.png")
                        plt.tight_layout()
                        plt.savefig(plot_loss_path, dpi=150)
                        plt.close()
                        print(f"Training vs Test Loss grafiği kaydedildi: {plot_loss_path}")
                else:
                    print("Uyarı: Çizilecek geçerli metrik bulunamadı!")

        except Exception as e:
            print(f"Epoch grafiği oluşturulurken hata: {e}")
            import traceback
            traceback.print_exc()
