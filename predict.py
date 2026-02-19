"""
predict.py — Toplu Tahmin Modülü
==================================
Bu dosya, eğitilmiş modeli kullanarak tüm ön işlenmiş veri seti
üzerinde toplu tahmin (batch prediction) yapar.

Kullanım senaryosu:
  Eğitimden sonra modelin genel veri setindeki performansını görmek
  ve tüm örnekler için tahmin dosyası oluşturmak istediğinizde kullanılır.

Çıktı:
  models/predictions.csv — Her satır bir kristale karşılık gelir.
  Sütunlar: tahmin edilen 4 hedef + kararlılık + elektronik tip etiketleri.

Not:
  Tek bir yeni kristal için tahmin yapmak istiyorsanız
  → python predict_crystal.py  kullanın.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ============================================================
# Sinir Ağı Mimarisi — train.py ile tamamen aynı olmalı
# ============================================================
# Model .pt dosyasından yüklenirken Python bu sınıfı tanıması gerekir.
# Bu yüzden mimari tanımı her tahmin/değerlendirme dosyasında tekrarlanır.
class NeuralNetwork(nn.Module):
    """
    Eğitimde kullanılan tam bağlı çok çıktılı sinir ağı.
    Giriş → 512 → 512 → 256 → 128 → 64 → 32 → Çıkış
    """
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# Sınıflandırıcı Fonksiyonlar
# ============================================================

def classify_stability(e_hull):
    """
    Hull üstü enerji (energy above hull) değerinden kararlılık etiketi üretir.

    Kararlılık eşikleri (Materials Project standardı):
      ≤ 0.025 eV/atom → Kararlı   : Sentezlenebilir, termodinamik olarak stabil
      ≤ 0.100 eV/atom → Meta-Kararlı: Belirli koşullarda sentezlenebilir
      > 0.100 eV/atom → Kararsız  : Bozunma eğilimli

    Parametreler:
        e_hull (float): Hull üstü enerji değeri (eV/atom)

    Döndürür:
        str: Kararlılık etiketi
    """
    if e_hull <= 0.025:
        return "Kararli (Stable)"
    elif e_hull <= 0.100:
        return "Meta-Kararli (Metastable)"
    else:
        return "Kararli Degil (Unstable)"


def classify_electronic(band_gap):
    """
    Bant aralığı değerinden elektronik tip etiketi üretir.

    Sınıflandırma kriterleri:
      < 0.01 eV  → Metal      : Valans ve iletim bandı üst üste gelir
      0.01–1.5 eV → Yarıiletken: Isıyla aktive olabilen küçük bant aralığı
      > 1.5 eV   → Yalıtkan   : Geniş bant aralığı, elektrik iletmez

    Parametreler:
        band_gap (float): Bant aralığı değeri (eV)

    Döndürür:
        str: Elektronik tip etiketi
    """
    if band_gap < 0.01:
        return "Metal"
    elif band_gap < 1.5:
        return "Yariiletken (Semiconductor)"
    else:
        return "Yalitkan (Insulator)"


# ============================================================
# BÖLÜM 1 — Ortam ve Model Yükleme
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Sıfıra bölme koruması
eps = 1e-8

# --- Normalizasyon istatistiklerini yükle ---
# Bu dosya train.py tarafından üretilir ve X ile y'nin ortalama/std değerlerini
# içerir. Tahmin için giriş verisini aynı ölçeğe getirmek zorunludur.
print("\n[1/5] Normalizasyon istatistikleri yükleniyor...")
norm_stats   = torch.load("models/normalization_stats.pth", map_location=device)
X_mean       = norm_stats["X_mean"].to(device)
X_std        = norm_stats["X_std"].to(device)
y_mean       = norm_stats["y_mean"].to(device)   # Şekil: (4,) — her hedef için
y_std        = norm_stats["y_std"].to(device)    # Şekil: (4,)
target_names = norm_stats.get("target_names", ["formation_energy_per_atom"])
n_targets    = len(target_names)

print(f"  Tahmin edilecek hedefler: {target_names}")

# --- Eğitilmiş modeli yükle ---
# safe_globals: PyTorch 2.6+ güvenlik gereksinimi.
# weights_only=False ile tam model (mimari + ağırlıklar) yüklenir.
print("\n[2/5] Eğitilmiş model yükleniyor...")
safe_classes = [NeuralNetwork, torch.nn.modules.container.Sequential]
with torch.serialization.safe_globals(safe_classes):
    model = torch.load(
        "models/best_model_full.pt",
        map_location=device,
        weights_only=False
    )
# eval() modu: Dropout katmanları kapatılır, deterministik tahmin yapılır
model.eval()
print("  Model başarıyla yüklendi.")

# ============================================================
# BÖLÜM 2 — Girdi Verisini Hazırla ve Normalize Et
# ============================================================
# Bu örnek, mevcut eğitim verisi üzerinde toplu tahmin yapar.
# Farklı bir veri seti için "data/X_preprocessed.csv" yerine
# kendi CSV dosyanızı belirtin. Sütun sırasının aynı olması gerekir.
print("\n[3/5] Girdi verisi yükleniyor ve normalize ediliyor...")
X_new = pd.read_csv("data/X_preprocessed.csv")
print(f"  Tahmin edilecek örnek sayısı: {len(X_new):,}")

# pandas DataFrame → PyTorch tensörü → GPU/CPU
X_tensor = torch.tensor(X_new.values, dtype=torch.float32).to(device)

# Eğitimde kullanılan normalizasyon istatistiklerini uygula.
# Aynı dönüşüm yapılmazsa model yanlış aralıkta girdi alır → hatalı tahmin.
X_tensor = (X_tensor - X_mean) / (X_std + eps)

# ============================================================
# BÖLÜM 3 — Tahmin ve Ters Normalizasyon
# ============================================================
print("\n[4/5] Tahminler yapılıyor...")
with torch.no_grad():  # Gradyan hesaplama gerekmez → bellek tasarrufu
    y_pred_norm = model(X_tensor)   # Model normalize uzayda tahmin üretir

    # Ters normalizasyon: tahminleri orijinal ölçeğe geri çevir
    # z = (x - ortalama) / std  →  x = z * std + ortalama
    y_pred = y_pred_norm * (y_std + eps) + y_mean

# PyTorch tensörü → NumPy dizisi (pandas ile çalışmak için)
y_pred_np = y_pred.cpu().numpy()  # Şekil: (N_örnek, 4)

# ============================================================
# BÖLÜM 4 — Tahmin DataFrame'ini Oluştur ve Sınıflandır
# ============================================================
print("\n[5/5] Sonuçlar düzenleniyor ve kaydediliyor...")

# Tahmin matrisini sütun adlarıyla DataFrame'e dönüştür
pred_df = pd.DataFrame(y_pred_np, columns=target_names)

# Fiziksel kısıtlamaları uygula:
#   - Band gap negatif olamaz (enerji farkıdır)
#   - Hull üstü enerji negatif olamaz (convex hull tanımı gereği)
if "band_gap" in pred_df.columns:
    pred_df["band_gap"] = pred_df["band_gap"].clip(lower=0.0)
if "energy_above_hull" in pred_df.columns:
    pred_df["energy_above_hull"] = pred_df["energy_above_hull"].clip(lower=0.0)

# Hull üstü enerjiden kararlılık etiketi üret
if "energy_above_hull" in pred_df.columns:
    pred_df["kararlılık"] = pred_df["energy_above_hull"].apply(classify_stability)

# Bant aralığından elektronik tip etiketi üret
if "band_gap" in pred_df.columns:
    pred_df["elektronik_tip"] = pred_df["band_gap"].apply(classify_electronic)

# Sonuçları CSV olarak kaydet
pred_df.to_csv("models/predictions.csv", index=False)
print("  Tahminler kaydedildi → models/predictions.csv")

# ============================================================
# BÖLÜM 5 — Özet Çıktısı
# ============================================================
print("\n" + "=" * 65)
print("  İLK 5 TAHMİN")
print("=" * 65)
print(pred_df.head().to_string())

# Sayısal hedefler için istatistikler
print("\n" + "=" * 65)
print("  TAHMİN İSTATİSTİKLERİ (tüm örnekler)")
print("=" * 65)
numeric_cols = [c for c in target_names if c in pred_df.columns]
print(pred_df[numeric_cols].describe().round(4).to_string())

# Kararlılık dağılımı
if "kararlılık" in pred_df.columns:
    print("\n" + "=" * 65)
    print("  KARARLILIIK DAĞILIMI")
    print("=" * 65)
    dist = pred_df["kararlılık"].value_counts()
    for label, count in dist.items():
        print(f"  {label:<35}: {count:>8,} ({count/len(pred_df)*100:.1f}%)")

# Elektronik tip dağılımı
if "elektronik_tip" in pred_df.columns:
    print("\n" + "=" * 65)
    print("  ELEKTRONİK TİP DAĞILIMI")
    print("=" * 65)
    dist = pred_df["elektronik_tip"].value_counts()
    for label, count in dist.items():
        print(f"  {label:<35}: {count:>8,} ({count/len(pred_df)*100:.1f}%)")
