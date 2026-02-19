# Derin Öğrenme ile Kristal Malzeme Özelliklerinin Tahmini

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2053--1591%2Fae22cb-blue)](https://doi.org/10.1088/2053-1591/ae22cb)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.7-red)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Lisans](https://img.shields.io/badge/Lisans-MIT-lightgrey)

---

## 📌 Proje Hakkında

Bu proje, inorganik kristal malzemelerin kimyasal formülü ve kristalografik simetrisi kullanılarak derin öğrenme ile birden fazla fiziksel özelliğinin aynı anda tahmin edilmesini sağlayan uçtan uca bir makine öğrenmesi sistemidir.

Sistem; **hiç sentezlenmemiş hipotetik kristaller** de dahil olmak üzere herhangi bir kimyasal formül için anlık tahmin yapabilmektedir.

> **Yayın:** Torlao V.C. et al., *"Formation energy prediction of material crystal structures using deep learning"*, Materials Research Express (2025).  
> **DOI:** [10.1088/2053-1591/ae22cb](https://doi.org/10.1088/2053-1591/ae22cb)

---

## 🎯 Tahmin Edilen Özellikler

Model, girilen kristal için aşağıdaki 4 özelliği **aynı anda** tahmin eder:

| # | Özellik | Birim | Açıklama |
|---|---|---|---|
| 1 | **Formasyon Enerjisi** | eV/atom | Kristal oluşumunun serbest elementlere göre enerji farkı |
| 2 | **Bant Aralığı** | eV | Valans bandı ile iletim bandı arasındaki enerji boşluğu |
| 3 | **Bant Enerjisi (CBM)** | eV | İletim Bandı Minimum enerji seviyesi |
| 4 | **Hull Üstü Enerji** | eV/atom | Konveks hull'a olan mesafe — kararlılık göstergesi |

Bu tahminlerden türetilen ek sınıflandırmalar:

**Termodinamik Kararlılık** (hull üstü enerjiden):

| Hull Üstü Enerji (eV/atom) | Sınıf |
|---|---|
| ≤ 0.025 | ✅ Kararlı (Stable) |
| 0.025 – 0.100 | ⚠️ Meta-Kararlı (Metastable) |
| > 0.100 | ❌ Kararsız (Unstable) |

**Elektronik Tip** (bant aralığından):

| Bant Aralığı (eV) | Tip |
|---|---|
| < 0.01 | Metal |
| 0.01 – 1.5 | Yarıiletken (Semiconductor) |
| > 1.5 | Yalıtkan (Insulator) |

---

## 📊 Model Performansı

Modelin 136.569 malzeme üzerindeki değerlendirme sonuçları:

| Hedef | MAE | RMSE | R² | Yorum |
|---|---|---|---|---|
| Formasyon Enerjisi | **0.0554 eV/atom** | 0.1452 | **0.9854** | 🟢 Mükemmel |
| Bant Aralığı | **0.1545 eV** | 0.3798 | **0.9377** | 🟢 Çok İyi |
| Hull Üstü Enerji | **0.0316 eV/atom** | 0.1397 | **0.8960** | 🟢 İyi |
| Bant Enerjisi (CBM) | 0.5061 eV | 1.3733 | 0.6951 | 🟡 Orta |

> Formasyon enerjisinde R²=0.985 ve MAE=0.055 eV/atom, literatürdeki kompozisyon tabanlı DNN modelleriyle rekabet eden bir performans seviyesidir.

---

## 🧠 Model Mimarisi

Tam bağlı (fully-connected) çok çıktılı derin sinir ağı:

```
Giriş (346 özellik)
    ↓
Linear(512) → ReLU → Dropout(0.1)
    ↓
Linear(512) → ReLU → Dropout(0.1)
    ↓
Linear(256) → ReLU
    ↓
Linear(128) → ReLU
    ↓
Linear(64)  → ReLU
    ↓
Linear(32)  → ReLU
    ↓
Linear(4)   ← 4 çıkış: [formasyon_enerjisi, bant_aralığı, CBM, hull_enerjisi]
```

**Eğitim Stratejisi:**
- 5 katlı çapraz doğrulama (K-Fold Cross Validation)
- Kayıp fonksiyonu: L1Loss (Ortalama Mutlak Hata)
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Öğrenme hızı düzenleyici: ReduceLROnPlateau (sabır=5, faktör=0.5)
- Erken durdurma: 15 epoch sabır
- Maksimum epoch: 500

---

## 🔬 Özellik Mühendisliği

Model toplam **346 özellik** kullanmaktadır:

### 1. Element Fraksiyonları (103 sütun)
Her elementin (H'dan Lr'ye) kristaldeki mol kesri.  
Örnek: Fe₂O₃ → Fe=0.40, O=0.60, diğer tüm elementler=0.0

### 2. Fiziksel Tanımlayıcılar (12 sütun)
| Tanımlayıcı | Açıklama |
|---|---|
| n_atoms | Birim hücresindeki toplam atom sayısı |
| n_elements | Farklı element sayısı |
| avg_atomic_mass | Ağırlıklı ortalama atom kütlesi (g/mol) |
| en_mean/max/min/range | Pauling elektronegatiflik istatistikleri |
| avg_covalent_radius | Ortalama kovalent yarıçap (Å) |
| ea_mean/max/min/range | Elektron ilgisi istatistikleri (eV) |

### 3. Uzay Grubu One-Hot Kodlaması (221 sütun)
Uzay grubu numarası (1–230) → sg_1, sg_2, ..., sg_230 sütunlarına dönüştürülür.

### 4. Kararlılık Etiketi One-Hot Kodlaması (3 sütun)
stab_Stable, stab_Metastable, stab_Unstable

---

## 📂 Veri Seti

Proje, **Materials Project** veritabanından alınan 136.569 inorganik kristal malzemeyi içermektedir.

| Detay | Değer |
|---|---|
| Toplam malzeme sayısı | 136.569 |
| Kararlı (Stable) | ~58.177 (%42.6) |
| Meta-Kararlı (Metastable) | ~35.665 (%26.1) |
| Kararsız (Unstable) | ~42.727 (%31.3) |

Veri seti Zenodo'da barındırılmaktadır:

👉 **[Zenodo Veri Seti (DOI)](https://zenodo.org/records/17504632)**

İndirilen CSV dosyasını `data/` klasörüne koyun.

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.10 veya üstü
- NVIDIA GPU (isteğe bağlı, CPU ile de çalışır)
- NVIDIA Driver 525+ (CUDA kullanılacaksa)

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
```

### 2. PyTorch Kurulumu

**GPU ile (CUDA 12.8 — önerilen):**
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

**CPU ile:**
```bash
pip install torch torchvision torchaudio
```

### 3. Diğer Bağımlılıkları Kurun
```bash
pip install -r requirements.txt
```

> CUDA 13.1 sürücüsü olan sistemler cu128 wheel ile sorunsuz çalışır (geriye dönük uyumluluk).

---

## 🚀 Kullanım

### Adım 1 — Veriyi Hazırla
```bash
python preparation.py
```
Ham CSV'yi işler, 4 hedefli `y_preprocessed.csv` ve `feature_columns.json` üretir.

Beklenen çıktı:
```
[1/4] Ham veri yükleniyor ve temizleniyor...
      Temizlenmiş veri seti boyutu : 136,569 satır × 126 sütun
      Kararlılık dağılımı:
        Stable     :   58,177 (42.6%)
        Metastable :   35,665 (26.1%)
        Unstable   :   42,727 (31.3%)
[2/4] Özellik mühendisliği ve kodlama yapılıyor...
      Özellik matrisi (X) boyutu : 136,569 örnek × 346 özellik
      Hedef matrisi   (y) boyutu : 136,569 örnek × 4 hedef
HAZIRLIK TAMAMLANDI
```

---

### Adım 2 — Modeli Eğit
```bash
python train.py
```
GPU varsa otomatik kullanır. 5-Fold çapraz doğrulama ile eğitim yapar.

Beklenen çıktı:
```
Kullanılan cihaz: cuda
GPU adı         : NVIDIA GeForce RTX 3050 6GB Laptop GPU

===== Kat 1 / 5 =====
  Epoch [ 50/500] | Eğitim: 0.1652 | Doğrulama: 0.1946
  [Erken durdurma] 207. epoch'ta duruldu

Ortalama Doğrulama Kaybı : 0.1786
EĞİTİM TAMAMLANDI
```

---

### Adım 3 — Kristal Tahmini ⭐

**Etkileşimli mod:**
```bash
python predict_crystal.py
```

```
╔══════════════════════════════════════════════════════════════╗
║       KRİSTAL ÖZELLİK TAHMİN SİSTEMİ                      ║
╚══════════════════════════════════════════════════════════════╝

Formül → Fe2O3
Uzay grubu [1-230]: 167

  Formasyon Enerjisi (eV/atom)  :  -2.1500
  Bant Aralığı (eV)             :   2.1020  → Yarıiletken
  Bant Enerjisi / CBM (eV)      :  +1.3200
  Hull Üstü Enerji (eV/atom)    :   0.0000  → ✅ KARARLI
```

**Doğrudan komut satırı:**
```bash
python predict_crystal.py Fe2O3 167
python predict_crystal.py GaAs 216
python predict_crystal.py NaCl 225
python predict_crystal.py BaTiO3 123
```

**Desteklenen formül formatları:**
```
Fe2O3          → Basit bileşik
Ca3(PO4)2      → Parantezli grup
(NH4)2SO4      → Karmaşık formül
GaAs           → Stokiyometrisiz bileşik
```

---

### Adım 4 — Toplu Tahmin (İsteğe Bağlı)
```bash
python predict.py
```
Tüm veri seti üzerinde tahmin yapar ve `models/predictions.csv` üretir.

---

### Adım 5 — Model Değerlendirme
```bash
python evaluate.py
```
Her hedef için MAE, RMSE, R² hesaplar ve grafikleri `figures/` klasörüne kaydeder.

---

## 📁 Proje Yapısı

```
formation-energy-prediction/
│
├── data/
│   ├── MP_queried_data_featurized_w_additional_acr_ae_en.csv  ← ham veri (Zenodo'dan)
│   ├── X_preprocessed.csv        ← özellik matrisi (preparation.py üretir)
│   ├── y_preprocessed.csv        ← 4 hedefli çıktı matrisi
│   ├── feature_columns.json      ← özellik sütun listesi (çıkarım için)
│   └── target_columns.json       ← hedef sütun listesi
│
├── models/
│   ├── best_model_full.pt        ← eğitilmiş model (mimari + ağırlıklar)
│   ├── best_model_overall.pth    ← yalnızca ağırlıklar
│   ├── normalization_stats.pth   ← z-score normalizasyon istatistikleri
│   └── kfold_loss_summary.svg    ← K-Fold kayıp grafiği
│
├── figures/
│   ├── true_vs_predicted_plot.svg   ← Gerçek vs Tahmin grafiği (4 hedef)
│   └── true_vs_predicted_plot.eps   ← Yayın kalitesi EPS
│
├── preparation.py      ← veri ön işleme ve özellik mühendisliği
├── train.py            ← model eğitimi (K-Fold, GPU destekli)
├── predict.py          ← toplu tahmin (veri seti üzerinde)
├── predict_crystal.py  ← tek kristal etkileşimli tahmin aracı ⭐
├── evaluate.py         ← model değerlendirme ve görselleştirme
├── requirements.txt    ← bağımlılık listesi
│
├── distribution.ipynb  ← veri dağılımı analizi
├── dnn_new.ipynb       ← model geliştirme deneyleri
├── exploration.ipynb   ← veri keşfi
├── shap.ipynb          ← SHAP özellik önem analizi
└── README.md
```

---

## 🔁 Çalışma Akışı

```
Ham CSV (Materials Project)
        ↓
  preparation.py
  ├── Aykırı değer filtresi (±5σ)
  ├── Eksik değer doldurma
  ├── Kararlılık etiketi oluşturma
  ├── One-hot kodlama (uzay grubu + kararlılık)
  └── X_preprocessed.csv + y_preprocessed.csv (4 hedef)
        ↓
    train.py
  ├── Z-score normalizasyon (hedef başına ayrı)
  ├── 5-Fold çapraz doğrulama
  ├── Erken durdurma + LR zamanlayıcı
  └── best_model_full.pt + normalization_stats.pth
        ↓
  predict_crystal.py          evaluate.py
  (tek kristal tahmini)       (model metrikleri + grafikler)
```

---

## 📈 SHAP Özellik Önem Analizi

`shap.ipynb` notebook'u ile hangi özelliklerin modelin tahminlerini en çok etkilediği görselleştirilebilir:

- **Element katkıları:** hangi elementlerin formasyon enerjisini en çok düşürdüğü/yükselttiği
- **Fiziksel tanımlayıcılar:** elektronegatiflik, kovalent yarıçap, elektron ilgisinin etkisi
- **Uzay grubu etkisi:** kristal simetrisi ile özellikler arasındaki ilişki

---

## ⚠️ Önemli Notlar ve Sınırlılıklar

1. **Yapısal bilgi eksikliği:** Model yalnızca kimyasal kompozisyon ve uzay grubu kullanır. Bağ açıları, koordinasyon sayısı, atomlar arası mesafe gibi yapısal bilgiler dahil değildir.

2. **DFT/GGA sınırlılığı:** Eğitim verisi saf GGA hesaplamalarından gelmektedir. Bazı geçiş metali oksitlerinde (Fe₂O₃ gibi) GGA, bant aralığını hatalı hesaplayarak 0 (metalik) verebilir. Bu veri seti kaynağının bilinen bir kısıtlamasıdır.

3. **Tahmin güvensizliği:** Hipotetik (hiç sentezlenmemiş) kristaller için tahminler, veri setindeki benzeri malzemelere olan uzaklığa göre değişir. Benzer malzeme yoksa doğruluk düşer.

4. **evaluate.py metrikleri:** Gösterilen metrikler (R²=0.985) tam eğitim verisi üzerinde hesaplanmıştır. Gerçek genelleme performansı için K-Fold doğrulama kaybına bakın: ortalama ~0.18 (normalize edilmiş MAE).

---

## 📜 Kurulum Sorunları

**"No module named mendeleev" hatası:**
```bash
pip install mendeleev
```

**"feature_columns.json not found" hatası:**
```bash
python preparation.py
```

**"best_model_full.pt not found" hatası:**
```bash
python train.py
```

**CUDA kullanılmıyor (CPU çalışıyor):**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 🧾 Atıf

Bu repo, model veya veri seti kullanılıyorsa lütfen alttaki çalışmayı atıf gösterin:

```bibtex
@article{torlao2025formation,
  title   = {Formation energy prediction of material crystal structures using deep learning},
  author  = {Torlao, V.C. et al.},
  journal = {Materials Research Express},
  year    = {2025},
  doi     = {10.1088/2053-1591/ae22cb}
}
```

---

## 📜 Lisans

Bu proje MIT Lisansı ile yayınlanmıştır. Ayrıntılar için `LICENSE` dosyasına bakın.
