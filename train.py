"""
train.py — Derin Öğrenme Modeli Eğitim Modülü
===============================================
Bu dosya, preparation.py tarafından oluşturulan ön işlenmiş veri üzerinde
çok çıktılı (multi-output) bir derin sinir ağı eğitir.

Mimari:
  Giriş → 512 → 512 → 256 → 128 → 64 → 32 → 4 Çıkış

4 çıkış şunlardır:
  [0] formation_energy_per_atom  — Formasyon enerjisi (eV/atom)
  [1] band_gap                   — Bant aralığı (eV)
  [2] cbm                        — İletim bandı minimum enerjisi (eV)
  [3] energy_above_hull          — Hull üstü enerji / kararlılık göstergesi (eV/atom)

Eğitim stratejisi:
  - 5 katlı çapraz doğrulama (K-Fold Cross Validation)
  - Kayıp fonksiyonu: L1Loss (MAE — mutlak ortalama hata)
  - Optimizer: Adam (lr=0.001, weight_decay=1e-5)
  - Öğrenme hızı düzenleyici: ReduceLROnPlateau (plateau'da lr/2)
  - Erken durdurma: 15 epoch sabır (patience)
  - Her hedef için ayrı z-score normalizasyonu
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# BÖLÜM 1 — Ortam Kurulumu
# ============================================================

# GPU varsa CUDA, yoksa CPU kullan.
# Kod hiçbir değişiklik gerektirmeden her iki ortamda da çalışır.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")
if device.type == "cuda":
    # GPU adını ve mevcut belleği göster
    print(f"GPU adı         : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Toplam VRAM     : {vram:.1f} GB")

# Model kayıt dizinini oluştur (zaten varsa hata vermez)
os.makedirs("models", exist_ok=True)

# Sıfıra bölmeden koruma sabiti (normalizasyonda std ≈ 0 durumuna karşı)
eps = 1e-8

# ============================================================
# BÖLÜM 2 — Veri Yükleme
# ============================================================

# preparation.py tarafından üretilen ön işlenmiş veriler yüklenir.
# X_preprocessed.csv : özellik matrisi (her satır bir kristal)
# y_preprocessed.csv : hedef matrisi (4 sütun — 4 tahmin hedefi)
print("\n[1/7] Veri yükleniyor...")
X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_preprocessed.csv")

print(f"  Özellik matrisi (X) : {X.shape[0]:,} örnek × {X.shape[1]} özellik")
print(f"  Hedef matrisi   (y) : {y.shape[0]:,} örnek × {y.shape[1]} hedef")
print(f"  Hedef sütunlar     : {y.columns.tolist()}")

# Kaç hedef tahmin edileceği buradan otomatik belirlenir.
# preparation.py çıktısına göre değişebilir — sabit kodlamaktan kaçınıyoruz.
n_targets = y.shape[1]
target_names = y.columns.tolist()

# ============================================================
# BÖLÜM 3 — PyTorch Tensörlerine Dönüştürme
# ============================================================

# pandas DataFrame → PyTorch float32 tensörü
# y_tensor boyutu: (N_örnekler, 4) — her satır 4 hedef değer içerir
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# ============================================================
# BÖLÜM 4 — Z-Score Normalizasyonu
# ============================================================
# Her özellik ve her hedef için ayrı ayrı standartlaştırma yapılır:
#   z = (x - ortalama) / standart_sapma
#
# Neden normalizasyon?
#   - Formation enerjisi genellikle −3 ile +1 eV/atom aralığında
#   - Band gap 0 ile 10+ eV arasında değişebilir
#   - Normalizasyon, farklı ölçekteki bu değerlerin birbirini
#     baskılamasını önler ve eğitim kararlılığını artırır.
#
# dim=0 → her sütun (özellik/hedef) için bağımsız istatistik hesaplar
print("\n[2/7] Normalizasyon hesaplanıyor...")
X_mean = X_tensor.mean(dim=0)   # Şekil: (n_özellik,)
X_std  = X_tensor.std(dim=0)    # Şekil: (n_özellik,)
y_mean = y_tensor.mean(dim=0)   # Şekil: (n_hedef,)  — her hedef için ayrı
y_std  = y_tensor.std(dim=0)    # Şekil: (n_hedef,)

# Normalizasyonu uygula; eps sıfıra bölmeyi önler
X_norm = (X_tensor - X_mean) / (X_std + eps)
y_norm = (y_tensor - y_mean) / (y_std + eps)

# --- Normalizasyon istatistiklerini kaydet ---
# Bu değerler, predict.py ve predict_crystal.py'de yeni veriler
# geldiğinde tam aynı normalizasyon uygulanması için gereklidir.
torch.save({
    "X_mean":       X_mean.cpu(),       # Özellik ortalamaları
    "X_std":        X_std.cpu(),        # Özellik standart sapmaları
    "y_mean":       y_mean.cpu(),       # Hedef ortalamaları (4 değer)
    "y_std":        y_std.cpu(),        # Hedef standart sapmaları (4 değer)
    "target_names": target_names,       # Hedef sütun adları
    "n_targets":    n_targets,          # Hedef sayısı (4)
}, "models/normalization_stats.pth")

print(f"  Normalizasyon istatistikleri kaydedildi.")
print(f"  Model çıktı sayısı : {n_targets} → {target_names}")

# ============================================================
# BÖLÜM 5 — Sinir Ağı Mimarisi Tanımı
# ============================================================
class NeuralNetwork(nn.Module):
    """
    Tam bağlı (fully-connected) çok çıktılı derin sinir ağı.

    Katmanlar:
      Giriş → Linear(512) → ReLU → Dropout(0.1)
             → Linear(512) → ReLU → Dropout(0.1)
             → Linear(256) → ReLU
             → Linear(128) → ReLU
             → Linear(64)  → ReLU
             → Linear(32)  → ReLU
             → Linear(output_size)  ← lineer çıkış (regresyon)

    Neden bu mimari?
      - İlk iki katmandaki Dropout(0.1), aşırı öğrenmeyi (overfitting)
        azaltır; %10 nöron rastgele kapatılır.
      - Son katmanda aktivasyon yoktur çünkü çıkış sürekli bir değerdir
        (sınıflandırma değil regresyon).
      - ReLU, gradyan kaybolması (vanishing gradient) sorununu hafifletir.

    Parametreler:
        input_size  (int): Özellik sayısı (X.shape[1])
        output_size (int): Hedef sayısı (4)
    """
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),          # Aşırı öğrenmeye karşı düzenlileştirme
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),          # İkinci düzenlileştirme katmanı
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),  # Lineer çıkış — regresyon için aktivasyon yok
        )

    def forward(self, x):
        """İleri geçiş: girişi tüm katmanlardan geçirerek tahmini üretir."""
        return self.layers(x)

# ============================================================
# BÖLÜM 6 — K-Katlı Çapraz Doğrulama Kurulumu
# ============================================================
# K-Fold Cross Validation nedir?
#   Veri seti 5 eşit parçaya bölünür. Her turda 4 parça eğitim,
#   1 parça doğrulama için kullanılır. Bu işlem 5 kez tekrarlanır.
#   Her tur farklı doğrulama kümesi kullanır → model daha güvenilir değerlenir.
print("\n[3/7] K-Katlı çapraz doğrulama başlatılıyor...")
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# TensorDataset, X ve y tensörlerini birleştirerek DataLoader'a hazırlar.
# DataLoader mini-batch'leri otomatik oluşturur.
dataset = TensorDataset(X_norm, y_norm)

# Tüm katlardaki en iyi modeli takip eden değişkenler
best_overall_loss = float('inf')
best_overall_model_state = None

# Her kattaki son eğitim ve doğrulama kayıpları (grafik için)
fold_train_losses, fold_test_losses = [], []

# ============================================================
# BÖLÜM 7 — Eğitim Döngüsü
# ============================================================
print(f"\n[4/7] Eğitim başlıyor ({k_folds} kat × maks. 500 epoch)...")
print("-" * 60)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_norm)):
    print(f"\n===== Kat {fold + 1} / {k_folds} =====")

    # --- Veri yükleyicileri oluştur ---
    # Subset: tüm veri setinden yalnızca ilgili indeksleri seçer
    # shuffle=True: her epoch'ta veriyi karıştırarak eğitim kümesini çeşitlendirir
    # batch_size=64: her adımda 64 örnek işlenir — GPU belleği ve hız dengesi
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=64)

    # --- Bu kat için yeni model oluştur ---
    # Her katta sıfırdan başlanır → katlar birbirinden bağımsızdır
    model = NeuralNetwork(X_tensor.shape[1], n_targets).to(device)

    # L1Loss (MAE): |tahmin - gerçek| ortalaması.
    # MSE'ye göre aykırı değerlere daha dayanıklıdır.
    criterion = nn.L1Loss()

    # Adam optimizer: adaptif öğrenme hızı — genellikle SGD'den hızlı yakınsar.
    # weight_decay=1e-5: L2 düzenlileştirme → ağırlıkların çok büyümesini önler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # ReduceLROnPlateau: doğrulama kaybı 5 epoch iyileşmezse lr'yi yarıya indirir.
    # Bu sayede eğitim sonunda ince ayar (fine-tuning) yapılabilir.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Erken durdurma değişkenleri
    best_test_loss   = float('inf')  # Bu katta görülen en iyi doğrulama kaybı
    patience_counter = 0             # Kaç epoch boyunca iyileşme olmadı
    patience         = 15            # Bu kadar epoch sabırsız kalırsa durdur
    num_epochs       = 500           # Maksimum epoch sayısı

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):

        # ── Eğitim aşaması ──────────────────────────────────────
        # model.train(): Dropout ve BatchNorm katmanları aktif olur
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()          # Önceki gradyanları sıfırla
            outputs = model(inputs)        # İleri geçiş: tahmin üret
            loss = criterion(outputs, targets)  # Kayıp hesapla
            loss.backward()                # Geri yayılım: gradyanları hesapla
            optimizer.step()               # Ağırlıkları güncelle
            # Toplam kaybı örnek sayısıyla ağırlıklandır (batch boyutları farklı olabilir)
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)  # Örnek başına ortalama kayıp

        # ── Doğrulama aşaması ────────────────────────────────────
        # model.eval(): Dropout kapatılır, deterministik tahmin yapılır
        model.eval()
        test_loss = 0.0
        with torch.no_grad():  # Gradyan hesaplama devre dışı → hız ve bellek tasarrufu
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)

        # Öğrenme hızı zamanlayıcısını güncelle
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # ── Erken durdurma kontrolü ──────────────────────────────
        # Doğrulama kaybı iyileştiyse sayacı sıfırla, değilse artır.
        # Sabır aşılırsa eğitimi durdur — aşırı öğrenmeyi önler.
        if test_loss < best_test_loss:
            best_test_loss   = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Erken durdurma] {epoch + 1}. epoch'ta duruldu "
                      f"(en iyi kayıp: {best_test_loss:.4f})")
                break

        # Her 50 epoch'ta ilerlemeyi göster
        if (epoch + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Eğitim: {train_loss:.4f} | "
                  f"Doğrulama: {test_loss:.4f} | "
                  f"LR: {lr_now:.6f}")

    # Bu katın son kayıplarını sakla
    fold_train_losses.append(train_losses[-1])
    fold_test_losses.append(best_test_loss)

    # --- Tüm katlar arasında en iyisini kaydet ---
    if best_test_loss < best_overall_loss:
        best_overall_loss       = best_test_loss
        best_overall_model_state = model.state_dict()
        torch.save(best_overall_model_state, "models/best_model_overall.pth")
        print(f"  → Yeni en iyi model! Kat {fold+1}, Kayıp = {best_test_loss:.4f}")

# ============================================================
# BÖLÜM 8 — K-Fold Kayıp Özeti Grafiği
# ============================================================
print("\n[5/7] Kayıp özet grafiği oluşturuluyor...")
plt.figure(figsize=(8, 6))
plt.plot(range(1, k_folds + 1), fold_train_losses, 'o--', label='Eğitim Kaybı')
plt.plot(range(1, k_folds + 1), fold_test_losses,  'o--', label='Doğrulama Kaybı')
plt.xlabel("Kat (Fold)", fontsize=14)
plt.ylabel("Kayıp (MAE, normalize)", fontsize=14)
plt.title(f"K-Fold Kayıp Özeti ({n_targets} hedef)", fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("models/kfold_loss_summary.svg", format='svg', dpi=300)
plt.show()

print("\n[6/7] K-Fold Özet İstatistikleri:")
print(f"  Ortalama Eğitim Kaybı    : {np.mean(fold_train_losses):.4f}")
print(f"  Ortalama Doğrulama Kaybı : {np.mean(fold_test_losses):.4f}")
for i, (tr, te) in enumerate(zip(fold_train_losses, fold_test_losses)):
    print(f"    Kat {i+1}: Eğitim={tr:.4f}  Doğrulama={te:.4f}")

# ============================================================
# BÖLÜM 9 — En İyi Modeli Kaydet
# ============================================================
# Tüm katlarda en düşük doğrulama kaybını veren modelin ağırlıkları
# best_model_overall.pth'e zaten kaydedildi.
# Burada bu ağırlıklar yüklenerek tam model (.pt) olarak da saklanır.
# .pt dosyası mimari + ağırlıkları birlikte içerir → tek dosyadan yüklenebilir.
print("\n[7/7] En iyi model kaydediliyor...")
best_model = NeuralNetwork(X_tensor.shape[1], n_targets).to(device)
best_model.load_state_dict(
    torch.load("models/best_model_overall.pth", map_location=device)
)
best_model.eval()
torch.save(best_model, "models/best_model_full.pt")

print("  models/best_model_overall.pth  → sadece ağırlıklar  ✓")
print("  models/best_model_full.pt      → mimari + ağırlıklar ✓")
print(f"\n  Toplam en iyi doğrulama kaybı : {best_overall_loss:.4f}")

print("\n" + "=" * 60)
print("  EĞİTİM TAMAMLANDI")
print(f"  Model şu {n_targets} hedefi tahmin ediyor:")
for i, name in enumerate(target_names):
    print(f"    [{i+1}] {name}")
print("\n  Sıradaki adım: python evaluate.py  veya  python predict_crystal.py")
print("=" * 60)
