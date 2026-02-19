"""
evaluate.py — Model Değerlendirme Modülü
==========================================
Bu dosya, eğitilmiş modelin gerçek dünya performansını ölçer ve
yayın kalitesinde görselleştirmeler üretir.

Her tahmin hedefi (formasyon enerjisi, bant aralığı, CBM, hull enerjisi)
için aşağıdaki metrikler hesaplanır:
  - MAE  (Mean Absolute Error)         : Ortalama mutlak hata
  - RMSE (Root Mean Squared Error)     : Karekök ortalama kare hata
  - R²   (Coefficient of Determination): Belirleme katsayısı (1.0 = mükemmel)

Çıktılar:
  - Konsol: metrik tablosu
  - figures/true_vs_predicted_plot.eps — vektör grafik (yayın için)
  - figures/true_vs_predicted_plot.svg — vektör grafik (web/sunum için)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Sinir Ağı Mimarisi — train.py ile tamamen aynı olmalı
# ============================================================
# Modeli .pt dosyasından yüklerken PyTorch bu sınıfın tanımını arar.
class NeuralNetwork(nn.Module):
    """
    Tam bağlı çok çıktılı sinir ağı (train.py ile özdeş mimari).
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
            nn.Linear(32, 1),    # Yer tutucu; gerçek çıkış sayısı .pt'den gelir
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# BÖLÜM 1 — Ortam Kurulumu
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Grafik ve sonuç dizinlerini oluştur
os.makedirs("figures", exist_ok=True)

# Sıfıra bölme koruması
eps = 1e-8

# ============================================================
# BÖLÜM 2 — Normalizasyon İstatistiklerini Yükle
# ============================================================
# train.py bu dosyayı oluşturur; X ve y'nin ortalama/std değerlerini içerir.
# Tahminden sonra geri normalizasyon (denormalizasyon) için gereklidir.
print("\n[1/6] Normalizasyon istatistikleri yükleniyor...")
norm_stats   = torch.load("models/normalization_stats.pth", map_location=device)
X_mean       = norm_stats["X_mean"].to(device)
X_std        = norm_stats["X_std"].to(device)
y_mean       = norm_stats["y_mean"].to(device)
y_std        = norm_stats["y_std"].to(device)
target_names = norm_stats.get("target_names", ["formation_energy_per_atom"])
n_targets    = len(target_names)

print(f"  Değerlendirilecek hedefler: {target_names}")

# ============================================================
# BÖLÜM 3 — Modeli Yükle
# ============================================================
# safe_globals: PyTorch 2.6+ ile gelen güvenli sınıf yükleme mekanizması.
# weights_only=False: tam model nesnesini (mimari + ağırlıklar) yükler.
print("\n[2/6] Eğitilmiş model yükleniyor...")
safe_classes = [NeuralNetwork, torch.nn.modules.container.Sequential]
with torch.serialization.safe_globals(safe_classes):
    model = torch.load(
        "models/best_model_full.pt",
        map_location=device,
        weights_only=False
    )
model.eval()  # Dropout kapatılır, deterministik mod
print("  Model yüklendi.")

# ============================================================
# BÖLÜM 4 — Veri Setini Yükle ve Normalize Et
# ============================================================
print("\n[3/6] Değerlendirme verisi yükleniyor...")
X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_preprocessed.csv")
print(f"  Örnek sayısı: {len(X):,}")

# pandas → PyTorch tensörü
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# Eğitimdeki normalizasyonu aynen uygula
# (modelin beklediği ölçek ile tutarlı olmak için)
X_norm = (X_tensor - X_mean) / (X_std + eps)

# ============================================================
# BÖLÜM 5 — Tahmin Yap ve Ters Normalizasyon Uygula
# ============================================================
print("\n[4/6] Tahminler hesaplanıyor...")
with torch.no_grad():
    # Model normalize uzayda tahmin üretir
    y_pred_norm = model(X_norm)

    # Ters normalizasyon: z → orijinal ölçek
    # z = (x - μ) / σ  →  x = z·σ + μ
    y_pred = y_pred_norm * (y_std + eps) + y_mean
    y_true = y_tensor                      # Gerçek değerler (orijinal ölçek)

# GPU tensörlerini CPU NumPy dizilerine çevir (sklearn ve matplotlib için)
y_true_np = y_true.cpu().numpy()   # Şekil: (N, 4)
y_pred_np = y_pred.cpu().numpy()   # Şekil: (N, 4)

# Fiziksel kısıtlamaları uygula
for i, name in enumerate(target_names):
    if name in ("band_gap", "energy_above_hull"):
        # Bu değerler negatif olamaz (fiziksel sınır)
        y_pred_np[:, i] = np.clip(y_pred_np[:, i], a_min=0.0, a_max=None)

# ============================================================
# BÖLÜM 6 — Performans Metrikleri Hesapla
# ============================================================
# Her hedef için ayrı ayrı üç metrik hesaplanır:
#
#   MAE  = (1/N) Σ |y_gerçek - y_tahmin|
#          → Tahminlerin ortalama mutlak sapması; aynı birimde yorumlanır.
#
#   RMSE = √[(1/N) Σ (y_gerçek - y_tahmin)²]
#          → Büyük hatalara daha duyarlı; MAE'den genellikle büyüktür.
#
#   R²   = 1 - SS_res / SS_tot
#          → 1.0 = mükemmel fit, 0.0 = sadece ortalama tahmin ediyor,
#            < 0 = ortalamadan daha kötü.

# Hedef etiketleri (Türkçe grafik isimleri için)
TARGET_LABELS = {
    "formation_energy_per_atom": "Formasyon Enerjisi (eV/atom)",
    "band_gap":                  "Bant Aralığı (eV)",
    "cbm":                       "Bant Enerjisi — CBM (eV)",
    "energy_above_hull":         "Hull Üstü Enerji (eV/atom)",
}

print("\n[5/6] Performans metrikleri hesaplanıyor...")
print("\n" + "=" * 65)
print(f"  {'Hedef':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("=" * 65)

metrics = {}
for i, name in enumerate(target_names):
    true_i = y_true_np[:, i]
    pred_i = y_pred_np[:, i]

    mae  = np.mean(np.abs(true_i - pred_i))
    rmse = np.sqrt(np.mean((true_i - pred_i) ** 2))

    # R² hesabı: kalıntı kareler toplamı / toplam kareler toplamı
    ss_res = np.sum((true_i - pred_i) ** 2)
    ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    label = TARGET_LABELS.get(name, name)
    print(f"  {label:<35} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

print("=" * 65)

# ============================================================
# BÖLÜM 7 — Gerçek vs Tahmin Grafikleri Oluştur
# ============================================================
# Her hedef için ayrı bir alt grafik oluşturulur.
# Her grafik şunları içerir:
#   - Dağılım noktaları (tahmin vs gerçek)
#   - İdeal uyum çizgisi (y = x) — sarı kesikli
#   - Regresyon uyum çizgisi — mavi
#   - Metrik bilgi kutusu (MAE, RMSE, R²)
print("\n[6/6] Grafikler oluşturuluyor...")

# Izgara düzeni: 2 sütun, gerektiği kadar satır
n_cols = min(2, n_targets)
n_rows = (n_targets + n_cols - 1) // n_cols

fig = plt.figure(figsize=(8 * n_cols, 7 * n_rows))
gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

# Her hedef için farklı renk
COLORS = ["#800020", "#1a6e8e", "#2e7d32", "#7b1fa2"]

for i, name in enumerate(target_names):
    ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

    true_i = y_true_np[:, i]
    pred_i = y_pred_np[:, i]
    m      = metrics[name]

    # Dağılım noktaları
    ax.scatter(true_i, pred_i,
               color=COLORS[i % len(COLORS)],
               alpha=0.5, s=18, label="Tahminler")

    # İdeal uyum çizgisi (y = x)
    lims = [min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())]
    ax.plot(lims, lims,
            color="#FFD700", linestyle='--', linewidth=2, label="İdeal Uyum (y=x)")

    # Polinom regresyon uyum çizgisi (doğrusal)
    z = np.polyfit(true_i, pred_i, 1)
    p = np.poly1d(z)
    ax.plot(true_i, p(true_i),
            color="steelblue", linewidth=1.5,
            label=f"Regresyon (eğim={z[0]:.2f})")

    # Başlık ve eksen etiketleri
    label = TARGET_LABELS.get(name, name)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel("Gerçek Değer", fontsize=11)
    ax.set_ylabel("Tahmin Edilen Değer", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Metrik bilgi kutusu — sol üst köşeye yerleştir
    metrics_text = (f"MAE  = {m['MAE']:.4f}\n"
                    f"RMSE = {m['RMSE']:.4f}\n"
                    f"R²   = {m['R2']:.4f}")
    ax.text(0.04, 0.96, metrics_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.85,
                      edgecolor='gray', boxstyle='round,pad=0.4'))

    ax.legend(fontsize=9, loc="lower right")

fig.suptitle(
    "Gerçek Değer vs Tahmin Edilen Değer — Tüm Hedefler",
    fontsize=16, fontweight='bold', y=1.01
)

# ============================================================
# BÖLÜM 8 — Grafikleri Kaydet
# ============================================================
# EPS: vektör format — LaTeX/dergi makale gönderimi için
# SVG: vektör format — web ve sunum için
eps_path = os.path.join("figures", "true_vs_predicted_plot.eps")
svg_path = os.path.join("figures", "true_vs_predicted_plot.svg")

plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight')

print(f"  Grafik kaydedildi:")
print(f"    {eps_path}  (EPS — yayın kalitesi)")
print(f"    {svg_path}  (SVG — web/sunum)")

plt.show()

print("\n" + "=" * 65)
print("  DEĞERLENDİRME TAMAMLANDI")
print("=" * 65)
