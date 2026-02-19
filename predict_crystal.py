"""
predict_crystal.py — Tek Kristal Etkileşimli Tahmin Aracı
==========================================================
Bu dosya, kullanıcının bir kristal kimyasal formülü girmesine ve
derin öğrenme modelinin o kristal için 4 özelliği tahmin etmesine
olanak sağlar:

  [1] Formasyon Enerjisi  (eV/atom) — kristal oluşumunun enerjisi
  [2] Bant Aralığı        (eV)      — elektronik bant boşluğu
  [3] Bant Enerjisi / CBM (eV)      — iletim bandı minimum enerji seviyesi
  [4] Hull Üstü Enerji    (eV/atom) → kararlılık tahmini

Özellikler:
  - Bilinen kristaller (Fe2O3, SiO2, GaAs...) için çalışır
  - Hiç sentezlenmemiş, hipotetik kristaller için de çalışır
  - Parantezli formüller desteklenir: Ca3(PO4)2, (NH4)2SO4
  - Komut satırından doğrudan kullanılabilir: python predict_crystal.py Fe2O3 167
  - Etkileşimli mod: python predict_crystal.py

Gerekli ön koşullar:
  preparation.py çalıştırılmış olmalı → data/feature_columns.json
  train.py çalıştırılmış olmalı       → models/best_model_full.pt
                                         models/normalization_stats.pth
"""

import json
import re
import sys
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# BÖLÜM 1 — Element Özellikleri Veri Tabanı
# ============================================================
# Tüm 103 element için 4 temel fiziksel özellik:
#   (atom_kütlesi_g/mol, pauling_elektronegatiflik, kovalent_yarıçap_Å, elektron_ilgisi_eV)
#
# Bu veriler ne için kullanılır?
#   Kullanıcı "Fe2O3" girdiğinde model; element kompozisyon fraksiyonlarına
#   ek olarak avg_atomic_mass, en_mean, avg_covalent_radius, ea_mean gibi
#   toplu tanımlayıcıları (descriptor) hesaplaması gerekir.
#   Bu hesaplama için her elementin bireysel özelliklerine ihtiyaç vardır.
#
# Veri kaynakları:
#   - Atom kütlesi       : IUPAC 2021
#   - Elektronegatiflik  : Pauling ölçeği
#   - Kovalent yarıçap   : Cordero 2008 (pm → Å dönüşümü yapılmış)
#   - Elektron ilgisi    : NIST referans verileri (eV)
#
# Değeri bilinmeyen elementler (bazı aktinidler/lantanidler) için
# yaklaşık değerler kullanılmıştır; bu kristaller genellikle veri
# setinde çok az temsil edildiğinden tahmin doğruluğu düşük olabilir.
ELEMENT_DATA = {
    'H':  (1.008,   2.20, 0.31, 0.754),
    'He': (4.003,   0.00, 0.28, 0.000),
    'Li': (6.941,   0.98, 1.28, 0.618),
    'Be': (9.012,   1.57, 0.96, 0.000),
    'B':  (10.81,   2.04, 0.84, 0.279),
    'C':  (12.01,   2.55, 0.77, 1.262),
    'N':  (14.01,   3.04, 0.71, 0.000),
    'O':  (16.00,   3.44, 0.66, 1.461),
    'F':  (19.00,   3.98, 0.64, 3.401),
    'Ne': (20.18,   0.00, 0.58, 0.000),
    'Na': (22.99,   0.93, 1.66, 0.548),
    'Mg': (24.31,   1.31, 1.41, 0.000),
    'Al': (26.98,   1.61, 1.21, 0.433),
    'Si': (28.09,   1.90, 1.11, 1.385),
    'P':  (30.97,   2.19, 1.07, 0.747),
    'S':  (32.07,   2.58, 1.05, 2.077),
    'Cl': (35.45,   3.16, 1.02, 3.613),
    'Ar': (39.95,   0.00, 1.06, 0.000),
    'K':  (39.10,   0.82, 2.03, 0.501),
    'Ca': (40.08,   1.00, 1.76, 0.018),
    'Sc': (44.96,   1.36, 1.70, 0.188),
    'Ti': (47.87,   1.54, 1.60, 0.079),
    'V':  (50.94,   1.63, 1.53, 0.526),
    'Cr': (52.00,   1.66, 1.39, 0.666),
    'Mn': (54.94,   1.55, 1.50, 0.000),
    'Fe': (55.85,   1.83, 1.42, 0.151),
    'Co': (58.93,   1.88, 1.38, 0.662),
    'Ni': (58.69,   1.91, 1.24, 1.156),
    'Cu': (63.55,   1.90, 1.32, 1.228),
    'Zn': (65.38,   1.65, 1.22, 0.000),
    'Ga': (69.72,   1.81, 1.22, 0.430),
    'Ge': (72.63,   2.01, 1.20, 1.233),
    'As': (74.92,   2.18, 1.19, 0.804),
    'Se': (78.97,   2.55, 1.20, 2.021),
    'Br': (79.90,   2.96, 1.20, 3.365),
    'Kr': (83.80,   0.00, 1.16, 0.000),
    'Rb': (85.47,   0.82, 2.20, 0.486),
    'Sr': (87.62,   0.95, 1.95, 0.052),
    'Y':  (88.91,   1.22, 1.90, 0.307),
    'Zr': (91.22,   1.33, 1.75, 0.426),
    'Nb': (92.91,   1.60, 1.64, 0.916),
    'Mo': (95.96,   2.16, 1.54, 0.748),
    'Tc': (98.00,   1.90, 1.47, 0.550),
    'Ru': (101.07,  2.20, 1.46, 1.046),
    'Rh': (102.91,  2.28, 1.42, 1.137),
    'Pd': (106.42,  2.20, 1.39, 0.562),
    'Ag': (107.87,  1.93, 1.45, 1.302),
    'Cd': (112.41,  1.69, 1.44, 0.000),
    'In': (114.82,  1.78, 1.42, 0.300),
    'Sn': (118.71,  1.96, 1.39, 1.112),
    'Sb': (121.76,  2.05, 1.39, 1.047),
    'Te': (127.60,  2.10, 1.38, 1.971),
    'I':  (126.90,  2.66, 1.39, 3.059),
    'Xe': (131.29,  0.00, 1.40, 0.000),
    'Cs': (132.91,  0.79, 2.44, 0.472),
    'Ba': (137.33,  0.89, 2.15, 0.145),
    'La': (138.91,  1.10, 2.07, 0.470),
    'Ce': (140.12,  1.12, 2.04, 0.500),
    'Pr': (140.91,  1.13, 2.03, 0.500),
    'Nd': (144.24,  1.14, 2.01, 0.500),
    'Pm': (145.00,  1.13, 1.99, 0.500),
    'Sm': (150.36,  1.17, 1.98, 0.500),
    'Eu': (151.96,  1.20, 1.98, 0.500),
    'Gd': (157.25,  1.20, 1.96, 0.500),
    'Tb': (158.93,  1.10, 1.94, 0.500),
    'Dy': (162.50,  1.22, 1.92, 0.500),
    'Ho': (164.93,  1.23, 1.92, 0.500),
    'Er': (167.26,  1.24, 1.89, 0.500),
    'Tm': (168.93,  1.25, 1.90, 0.500),
    'Yb': (173.05,  1.10, 1.87, 0.500),
    'Lu': (174.97,  1.27, 1.87, 0.500),
    'Hf': (178.49,  1.30, 1.75, 0.000),
    'Ta': (180.95,  1.50, 1.70, 0.322),
    'W':  (183.84,  2.36, 1.62, 0.815),
    'Re': (186.21,  1.90, 1.51, 0.150),
    'Os': (190.23,  2.20, 1.44, 1.078),
    'Ir': (192.22,  2.20, 1.41, 1.565),
    'Pt': (195.08,  2.28, 1.36, 2.128),
    'Au': (196.97,  2.54, 1.36, 2.309),
    'Hg': (200.59,  2.00, 1.32, 0.000),
    'Tl': (204.38,  1.62, 1.45, 0.200),
    'Pb': (207.20,  2.33, 1.46, 0.364),
    'Bi': (208.98,  2.02, 1.48, 0.946),
    'Po': (209.00,  2.00, 1.40, 1.900),
    'At': (210.00,  2.20, 1.50, 2.800),
    'Rn': (222.00,  0.00, 1.50, 0.000),
    'Fr': (223.00,  0.70, 2.60, 0.486),
    'Ra': (226.00,  0.90, 2.21, 0.100),
    'Ac': (227.00,  1.10, 2.15, 0.350),
    'Th': (232.04,  1.30, 2.06, 0.608),
    'Pa': (231.04,  1.50, 2.00, 0.550),
    'U':  (238.03,  1.38, 1.96, 0.530),
    'Np': (237.00,  1.36, 1.90, 0.480),
    'Pu': (244.00,  1.28, 1.87, 0.000),
    'Am': (243.00,  1.30, 1.80, 0.000),
    'Cm': (247.00,  1.30, 1.69, 0.000),
    'Bk': (247.00,  1.30, 1.68, 0.000),
    'Cf': (251.00,  1.30, 1.68, 0.000),
    'Es': (252.00,  1.30, 1.65, 0.000),
    'Fm': (257.00,  1.30, 1.67, 0.000),
    'Md': (258.00,  1.30, 1.73, 0.000),
    'No': (259.00,  1.30, 1.76, 0.000),
    'Lr': (262.00,  1.30, 1.61, 0.000),
}

# Periyodik tablodaki tüm elementlerin sembol listesi (H'dan Lr'ye).
# Özellik vektörü oluşturulurken bu sıra kullanılır.
ALL_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr',
]


# ============================================================
# BÖLÜM 2 — Sinir Ağı Mimarisi (train.py ile aynı)
# ============================================================
class NeuralNetwork(nn.Module):
    """
    Tam bağlı çok çıktılı sinir ağı (train.py ile özdeş mimari).
    Model .pt dosyasından yüklenirken bu sınıf tanımı gereklidir.
    Giriş → 512 → 512 → 256 → 128 → 64 → 32 → Çıkış(4)
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
# BÖLÜM 3 — Sınıflandırıcı Fonksiyonlar
# ============================================================

def classify_stability(e_hull: float):
    """
    Hull üstü enerji değerinden kararlılık etiketi ve simge döndürür.

    Eşik değerleri:
      ≤ 0.025 eV/atom → Kararlı    (sentezlenebilir, stabil)
      ≤ 0.100 eV/atom → Meta-Kararlı (koşullu stabil)
      > 0.100 eV/atom → Kararsız   (bozunma eğilimli)

    Parametreler:
        e_hull (float): Hull üstü enerji (eV/atom)

    Döndürür:
        tuple[str, str]: (etiket, durum_simgesi)
    """
    if e_hull <= 0.025:
        return "KARARLI (Stable)", "✅"
    elif e_hull <= 0.100:
        return "META-KARARLI (Metastable)", "⚠️"
    else:
        return "KARARLI DEĞİL (Unstable)", "❌"


def classify_electronic(band_gap: float):
    """
    Bant aralığı değerinden elektronik tip etiketi döndürür.

    Eşik değerleri:
      < 0.01 eV  → Metal       (valans ve iletim bandı üst üste)
      0.01–1.5 eV → Yarıiletken (ısıyla aktive edilebilir)
      > 1.5 eV   → Yalıtkan    (geniş bant aralığı, iletmiyor)

    Parametreler:
        band_gap (float): Bant aralığı değeri (eV)

    Döndürür:
        str: Elektronik tip adı
    """
    if band_gap < 0.01:
        return "Metal"
    elif band_gap < 1.5:
        return "Yarıiletken (Semiconductor)"
    else:
        return "Yalıtkan (Insulator)"


# ============================================================
# BÖLÜM 4 — Kimyasal Formül Ayrıştırıcı
# ============================================================

def parse_formula(formula: str) -> dict:
    """
    Kimyasal formül dizesini element:adet sözlüğüne dönüştürür.

    Desteklenen formatlar:
      - Basit   : Fe2O3, SiO2, NaCl, GaAs
      - Parantezli: Ca3(PO4)2, (NH4)2SO4
      - İç içe parantez: (Ca(PO4))2 (özyinelemeli çözülür)
      - Ondalıklı katsayı: Fe1.5O2.5

    Algoritma:
      1. Formül içindeki parantezler bulunarak katsayıyla çarpılır
         ve düzleştirilir (özyinelemeli).
      2. Düzleştirilmiş dizeden büyük harf + küçük harf + sayı
         örüntüsü ile element-adet çiftleri regex ile çekilir.
      3. Aynı element birden fazla kez geçiyorsa (örn. H2O + H2)
         adetler toplanır.

    Parametreler:
        formula (str): Kimyasal formül dizesi

    Döndürür:
        dict: {element_sembolü: adet} — örn. {'Fe': 2.0, 'O': 3.0}

    Örnek:
        parse_formula("Ca3(PO4)2")
        → {'Ca': 3.0, 'P': 2.0, 'O': 8.0}
    """
    def expand_parentheses(s: str) -> str:
        """Parantez içindeki grupları katsayıyla çarpar ve açar."""
        # Örüntü: (içerik)katsayı
        pattern = r'\(([^()]+)\)(\d*\.?\d*)'
        while '(' in s:
            def replacer(m):
                inner = m.group(1)          # Parantez içindeki içerik
                mult  = float(m.group(2)) if m.group(2) else 1.0  # Dış katsayı
                # İç içeriği tekrar ayrıştır ve katsayıyla çarp
                tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', inner)
                result = ''
                for sym, cnt in tokens:
                    new_cnt = (float(cnt) if cnt else 1.0) * mult
                    # Tam sayıysa int olarak yaz, değilse ondalıklı
                    result += sym + (str(int(new_cnt)) if new_cnt == int(new_cnt)
                                     else str(new_cnt))
                return result
            s = re.sub(pattern, replacer, s)
        return s

    # Adım 1: Parantezleri aç ve düzleştir
    expanded = expand_parentheses(formula.strip())

    # Adım 2: Element sembolü + adet çiftlerini bul
    # Örüntü: Büyük harf + isteğe bağlı küçük harf + isteğe bağlı sayı
    # Örnek: "Fe2O3" → [("Fe","2"), ("O","3")]
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', expanded)

    # Adım 3: Adetleri topla
    composition: dict = {}
    for symbol, count in tokens:
        cnt = float(count) if count else 1.0
        composition[symbol] = composition.get(symbol, 0.0) + cnt

    return composition


# ============================================================
# BÖLÜM 5 — Özellik Vektörü Oluşturucu
# ============================================================

def build_feature_vector(formula: str, space_group: int,
                          feature_columns: list) -> np.ndarray:
    """
    Bir kristal formülü ve uzay grubundan, modelin beklediği tam
    özellik vektörünü oluşturur.

    Vektör bileşenleri (eğitim sırası korunarak):
      1. Element fraksiyonları (103 sütun)
           Her element için: adet / toplam_atom
           Formülde olmayan elementler için: 0.0
      2. Fiziksel tanımlayıcılar (12 sütun)
           n_atoms, n_elements, avg_atomic_mass,
           en_mean/max/min/range, avg_covalent_radius,
           ea_mean/max/min/range
      3. Uzay grubu one-hot (sg_1 … sg_230)
           Girilen uzay grubu için 1, diğerleri 0
      4. Kararlılık etiketi one-hot (stab_Stable, stab_Metastable, stab_Unstable)
           Yeni/bilinmeyen kristaller için tümü 0

    Neden uzay grubu ve kararlılık etiketi gerekli?
      Bunlar eğitim sırasında özellik matrisine eklendi.
      Tahmin için de aynı sütun yapısı gerekir.
      Bilinmeyen kristaller için tümü 0 bırakılır.

    Parametreler:
        formula        (str)  : Kimyasal formül (örn. "Fe2O3")
        space_group    (int)  : Uzay grubu numarası 1–230
        feature_columns(list) : Eğitimde kullanılan sütun adları listesi

    Döndürür:
        np.ndarray: float32 tipinde özellik vektörü
    """
    # Formülü ayrıştır: {sembol: adet}
    composition = parse_formula(formula)
    if not composition:
        raise ValueError(f"Formül ayrıştırılamadı: '{formula}'")

    # Bilinmeyen elementler için uyarı ver
    unknown = [s for s in composition if s not in ELEMENT_DATA]
    if unknown:
        print(f"  [!] Bilinmeyen element(ler) — varsayılan değer kullanılıyor: {unknown}")

    # Toplam atom sayısı ve mol kesirleri
    total_atoms = sum(composition.values())
    fractions   = {sym: cnt / total_atoms for sym, cnt in composition.items()}

    # Her element için özellikleri ağırlıklı listele
    # (kütle, elektronegatiflik, kovalent yarıçap, elektron ilgisi)
    masses, ens, cov_rs, eas = [], [], [], []
    for sym, frac in fractions.items():
        # Bilinmeyen element için genel yaklaşım değerleri
        props = ELEMENT_DATA.get(sym, (100.0, 1.5, 1.5, 0.5))
        mass, en, cov_r, ea = props
        masses.append((mass, frac))
        if en > 0:                  # Elektronegatifliği olmayan elementleri atla (soy gazlar)
            ens.append((en, frac))
        cov_rs.append((cov_r, frac))
        eas.append((ea, frac))

    # --- Toplu tanımlayıcıları hesapla ---

    # Ağırlıklı ortalama atom kütlesi
    avg_mass = sum(m * f for m, f in masses)

    # Elektronegatiflik istatistikleri (yalnızca EN > 0 olan elementler dahil)
    if ens:
        en_total = sum(f for _, f in ens)
        en_mean  = sum(e * f for e, f in ens) / en_total  # Ağırlıklı ortalama
        en_max   = max(e for e, _ in ens)
        en_min   = min(e for e, _ in ens)
        en_range = en_max - en_min                         # Aralık = elektron çekme farkı
    else:
        en_mean = en_max = en_min = en_range = 0.0

    # Ağırlıklı ortalama kovalent yarıçap
    avg_cov_r = sum(r * f for r, f in cov_rs)

    # Elektron ilgisi istatistikleri
    ea_vals  = [e for e, _ in eas]
    ea_fracs = [f for _, f in eas]
    ea_total = sum(ea_fracs)
    ea_mean  = (sum(e * f for e, f in zip(ea_vals, ea_fracs)) / ea_total
                if ea_total else 0.0)
    ea_max   = max(ea_vals) if ea_vals else 0.0
    ea_min   = min(ea_vals) if ea_vals else 0.0
    ea_range = ea_max - ea_min

    # --- Özellik sözlüğünü oluştur ---
    feat: dict = {}

    # 1) Element fraksiyonları — 103 sütun
    for el in ALL_ELEMENTS:
        feat[el] = fractions.get(el, 0.0)  # Formülde yoksa 0.0

    # 2) Fiziksel tanımlayıcılar — 12 sütun
    feat['n_atoms']             = total_atoms
    feat['n_elements']          = float(len(composition))
    feat['avg_atomic_mass']     = avg_mass
    feat['en_mean']             = en_mean
    feat['en_max']              = en_max
    feat['en_min']              = en_min
    feat['en_range']            = en_range
    feat['avg_covalent_radius'] = avg_cov_r
    feat['ea_mean']             = ea_mean
    feat['ea_max']              = ea_max
    feat['ea_min']              = ea_min
    feat['ea_range']            = ea_range

    # 3) Uzay grubu one-hot — eğitimde görülen sg_* sütunları
    # 4) Kararlılık etiketi one-hot — stab_* sütunları (yeni kristal için 0)
    sg_col = f"sg_{space_group}"
    for col in feature_columns:
        if col.startswith("sg_"):
            feat[col] = 1.0 if col == sg_col else 0.0
        elif col.startswith("stab_"):
            feat[col] = 0.0   # Yeni kristal için kararlılık bilinmiyor → 0

    # Eğitim sırası korunarak vektörü oluştur
    vector = np.array(
        [feat.get(col, 0.0) for col in feature_columns],
        dtype=np.float32
    )
    return vector


# ============================================================
# BÖLÜM 6 — Ana Tahmin Fonksiyonu
# ============================================================

def predict(formula: str, space_group: int = 1,
            device: torch.device = None) -> dict:
    """
    Bir kristal formülü için 4 özelliği tahmin eder.

    İşlem adımları:
      1. data/feature_columns.json yüklenerek özellik sırası alınır.
      2. models/normalization_stats.pth'den normalizasyon istatistikleri yüklenir.
      3. models/best_model_full.pt'den eğitilmiş model yüklenir.
      4. Formül ayrıştırılarak özellik vektörü oluşturulur.
      5. Özellikler eğitimdekiyle aynı normalizasyona tabi tutulur.
      6. Model normalize uzayda tahmin üretir.
      7. Ters normalizasyon ile orijinal ölçeğe döndürülür.
      8. Fiziksel kısıtlamalar uygulanır (band_gap ≥ 0, hull_energy ≥ 0).

    Parametreler:
        formula     (str)            : Kimyasal formül (örn. "Fe2O3")
        space_group (int)            : Uzay grubu numarası (varsayılan: 1)
        device      (torch.device)   : CPU veya CUDA cihazı

    Döndürür:
        dict: {hedef_adı: tahmin_değeri}
              Örn. {'formation_energy_per_atom': -2.18, 'band_gap': 1.95, ...}

    Hata fırlatır:
        FileNotFoundError: feature_columns.json veya model dosyaları bulunamazsa
        ValueError: Formül ayrıştırılamazsa
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- feature_columns.json yükle ---
    # Bu dosya preparation.py tarafından üretilir ve modelin beklediği
    # tam sütun listesini içerir. Olmadan doğru özellik sırası kurulamaz.
    try:
        with open("data/feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "data/feature_columns.json bulunamadı.\n"
            "Çözüm: önce 'python preparation.py' çalıştırın."
        )

    # --- Normalizasyon istatistiklerini yükle ---
    try:
        norm_stats = torch.load(
            "models/normalization_stats.pth", map_location=device
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "models/normalization_stats.pth bulunamadı.\n"
            "Çözüm: önce 'python train.py' çalıştırın."
        )

    X_mean       = norm_stats["X_mean"].to(device)
    X_std        = norm_stats["X_std"].to(device)
    y_mean       = norm_stats["y_mean"].to(device)
    y_std        = norm_stats["y_std"].to(device)
    target_names = norm_stats.get("target_names", ["formation_energy_per_atom"])

    # --- Eğitilmiş modeli yükle ---
    try:
        safe_classes = [NeuralNetwork, torch.nn.modules.container.Sequential]
        with torch.serialization.safe_globals(safe_classes):
            model = torch.load(
                "models/best_model_full.pt",
                map_location=device,
                weights_only=False
            )
        model.eval()  # Dropout kapatılır; tahmin deterministik olur
    except FileNotFoundError:
        raise FileNotFoundError(
            "models/best_model_full.pt bulunamadı.\n"
            "Çözüm: önce 'python train.py' çalıştırın."
        )

    # --- Özellik vektörü oluştur ve normalize et ---
    eps = 1e-8
    x = build_feature_vector(formula, space_group, feature_columns)

    # (1, n_özellik) boyutuna getir — model tek örnek bekliyor
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # Eğitimde kullanılan normalizasyonu uygula
    x_tensor = (x_tensor - X_mean) / (X_std + eps)

    # --- Modeli çalıştır ve ters normalizasyon uygula ---
    with torch.no_grad():   # Eğitim yapmıyoruz, gradyan hesabına gerek yok
        y_norm = model(x_tensor)                      # Normalize uzayda tahmin
        y_pred = y_norm * (y_std + eps) + y_mean      # Orijinal ölçeğe dönüştür

    # Sonuçları sözlüğe al; fiziksel kısıtları uygula
    results: dict = {}
    for i, name in enumerate(target_names):
        val = y_pred[0, i].item()
        if name in ("band_gap", "energy_above_hull"):
            val = max(0.0, val)    # Bu değerler negatif olamaz (fiziksel sınır)
        results[name] = val

    return results


# ============================================================
# BÖLÜM 7 — Sonuç Görüntüleyici
# ============================================================

def display_results(formula: str, space_group: int, results: dict):
    """
    Tahmin sonuçlarını okunaklı, açıklamalı biçimde ekrana yazdırır.

    Her hedef için:
      - Sayısal değer gösterilir
      - Kısa yorum eklenir (enerjetik uygunluk, elektron tipi vb.)

    Son olarak özet bir kutucuk sunulur.

    Parametreler:
        formula    (str) : Tahmin yapılan formül
        space_group(int) : Kullanılan uzay grubu numarası
        results    (dict): predict() fonksiyonunun çıktısı
    """
    line = "=" * 64

    print(f"\n{line}")
    print(f"  KRİSTAL TAHMİN SONUÇLARI")
    print(f"  Formül      : {formula}")
    print(f"  Uzay Grubu  : {space_group}")
    print(f"{line}")

    # --- Formasyon Enerjisi ---
    ef = results.get("formation_energy_per_atom")
    if ef is not None:
        print(f"\n  Formasyon Enerjisi (eV/atom)  :  {ef:+.4f}")
        if ef < -2.0:
            yorum = "Çok güçlü bağ oluşumu — termodinamik olarak çok kararlı"
        elif ef < 0:
            yorum = "Negatif → bileşim oluşumu enerjetik açıdan uygundur"
        else:
            yorum = "Pozitif → serbest elementlere göre kararsız"
        print(f"           → {yorum}")

    # --- Bant Aralığı ---
    bg = results.get("band_gap")
    if bg is not None:
        etype = classify_electronic(bg)
        print(f"\n  Bant Aralığı (eV)             :  {bg:.4f}")
        print(f"           → Elektronik Tip: {etype}")

    # --- Bant Enerjisi (CBM) ---
    cbm = results.get("cbm")
    if cbm is not None:
        print(f"\n  Bant Enerjisi / CBM (eV)      :  {cbm:+.4f}")
        print(f"           → İletim Bandı Minimum Enerji Seviyesi")

    # --- Hull Üstü Enerji / Kararlılık ---
    eh = results.get("energy_above_hull")
    if eh is not None:
        label, icon = classify_stability(eh)
        print(f"\n  Hull Üstü Enerji (eV/atom)    :  {eh:.4f}")
        print(f"           → Kararlılık: {icon} {label}")
        if eh <= 0.025:
            detay = "Convex hull üzerinde veya çok yakın → sentezlenme olasılığı yüksek"
        elif eh <= 0.100:
            detay = "Hafif meta-kararlı → sentezlenebilir ancak daha stabil fazlara ayrışabilir"
        else:
            detay = "Hull'dan uzak → termodinamik olarak kararsız, ayrışma beklenir"
        print(f"              {detay}")

    # --- Özet Kutusu ---
    print(f"\n{'-' * 64}")
    print(f"  GENEL ÖZET:")
    if eh is not None:
        label, icon = classify_stability(eh)
        print(f"  {icon}  Termodinamik Kararlılık : {label}")
    if bg is not None:
        print(f"  ⚡  Elektronik Özellik     : {classify_electronic(bg)}")
    if ef is not None:
        durum = "kararlı" if ef < 0 else "kararsız"
        print(f"  🔋  Formasyon Durumu      : Enerjetik açıdan {durum} ({ef:+.4f} eV/atom)")
    print(f"{line}\n")


# ============================================================
# BÖLÜM 8 — Etkileşimli Komut Satırı Arayüzü
# ============================================================

# Başlangıçta gösterilecek karşılama mesajı
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       KRİSTAL ÖZELLİK TAHMİN SİSTEMİ                      ║
║       Derin Öğrenme Tabanlı Malzeme Analizi                 ║
╠══════════════════════════════════════════════════════════════╣
║  Tahmin edilen özellikler:                                   ║
║    • Formasyon Enerjisi  (eV/atom)                          ║
║    • Bant Aralığı        (eV)                               ║
║    • Bant Enerjisi / CBM (eV)                               ║
║    • Kararlılık (Hull Üstü Enerji)                          ║
║                                                              ║
║  Komutlar: ornek | uzay | q (çıkış)                         ║
╚══════════════════════════════════════════════════════════════╝
"""

# Kullanıcı "ornek" yazdığında gösterilecek örnekler
EXAMPLES = [
    ("Fe2O3",   167, "Hematit"),
    ("SiO2",    154, "Kuvars"),
    ("GaAs",    216, "Galyum Arsenid"),
    ("NaCl",    225, "Halit (Tuz)"),
    ("TiO2",    136, "Rutil"),
    ("BaTiO3",  123, "Baryum Titanat"),
    ("Al2O3",   167, "Korund"),
    ("ZnO",     186, "Çinko Oksit"),
    ("Ca3(PO4)2", 11, "Hidroksiapatit (yaklaşık)"),
]

# Kullanıcı "uzay" yazdığında gösterilecek uzay grubu ipuçları
SPACE_GROUP_HINTS = """
Yaygın Uzay Grupları (uzay grubu bilinmiyorsa 1 girin):
  Kübik    : 225 (Fm-3m/FCC), 229 (Im-3m/BCC), 216 (F-43m/Zincblende)
  Tetragonal: 139 (I4/mmm),   136 (P4₂/mnm)
  Hegzagonal: 194 (P6₃/mmc), 186 (P6₃mc)
  Trigonal  : 167 (R-3c),     166 (R-3m)
  Ortorombik: 62  (Pnma),      63 (Cmcm)
  Monoklinik: 14  (P2₁/c)
  Triklinik : 1   (P1) ← varsayılan (en genel)
"""


def run_interactive():
    """
    Etkileşimli tahmin arayüzünü başlatır.

    Kullanıcıdan döngü halinde:
      1. Kristal formülü alır (örn. Fe2O3)
      2. Uzay grubu numarası alır (isteğe bağlı, varsayılan: 1)
      3. predict() ile tahmin yapar
      4. display_results() ile sonuçları gösterir

    Özel komutlar:
      'q' veya 'cikis' → programdan çık
      'ornek'          → örnek kristal listesini göster
      'uzay'           → uzay grubu ipuçlarını göster
    """
    print(BANNER)

    # GPU/CPU bilgisini göster
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_bilgi = (f"GPU: {torch.cuda.get_device_name(0)}"
                 if dev.type == "cuda" else "CPU modu")
    print(f"  Cihaz: {dev} ({gpu_bilgi})\n")
    print("  Hazır! Kristal formülünü girin.")
    print("  Yardım için 'ornek' veya 'uzay' yazın.\n")

    while True:
        # Formül girişi al
        try:
            formula_raw = input("Formül → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Çıkılıyor...")
            break

        # Boş giriş → atla
        if not formula_raw:
            continue

        # Çıkış komutu
        if formula_raw.lower() in ('q', 'quit', 'exit', 'cikis', 'çıkış'):
            print("  İyi çalışmalar!")
            break

        # Örnek listesini göster
        if formula_raw.lower() in ('ornek', 'örnek', 'example', 'help', 'yardim'):
            print("\n  Örnek Kristaller:")
            print(f"  {'Formül':<15} {'Uzay Gr.':>8}   {'Malzeme Adı'}")
            print("  " + "-" * 45)
            for fm, sg, name in EXAMPLES:
                print(f"  {fm:<15} {sg:>8}   {name}")
            print()
            continue

        # Uzay grubu ipuçlarını göster
        if formula_raw.lower() in ('uzay', 'sg', 'spacegroup'):
            print(SPACE_GROUP_HINTS)
            continue

        # Uzay grubu girişi al (isteğe bağlı)
        print("  Uzay grubu [1-230, bilinmiyorsa Enter]: ", end="")
        try:
            sg_raw = input().strip()
        except (KeyboardInterrupt, EOFError):
            sg_raw = ""

        if sg_raw == "":
            space_group = 1
            print("  → Varsayılan uzay grubu: 1 (P1 - en genel)")
        elif sg_raw.lstrip('-').isdigit() and 1 <= int(sg_raw) <= 230:
            space_group = int(sg_raw)
        else:
            print("  → Geçersiz değer; varsayılan 1 kullanılıyor.")
            space_group = 1

        # Tahmin yap ve sonucu göster
        print(f"\n  '{formula_raw}' (Uzay Grubu: {space_group}) için tahmin yapılıyor...")
        try:
            results = predict(formula_raw, space_group, dev)
            display_results(formula_raw, space_group, results)
        except ValueError as e:
            print(f"\n  [Hata] {e}\n")
        except FileNotFoundError as e:
            print(f"\n  [Dosya Hatası] {e}\n")
            break   # Gerekli dosyalar yoksa döngüden çık
        except Exception as e:
            print(f"\n  [Beklenmedik Hata] {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# BÖLÜM 9 — Giriş Noktası
# ============================================================
# Kullanım şekilleri:
#
#   Etkileşimli mod (menü):
#     python predict_crystal.py
#
#   Doğrudan komut satırı:
#     python predict_crystal.py <formül> [uzay_grubu]
#     python predict_crystal.py Fe2O3 167
#     python predict_crystal.py GaAs 216
#     python predict_crystal.py NaCl
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Komut satırı argümanlarıyla çalıştırıldıysa tek tahmin yap
        fm  = sys.argv[1]
        sg  = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            res = predict(fm, sg, dev)
            display_results(fm, sg, res)
        except Exception as e:
            print(f"Hata: {e}")
            sys.exit(1)
    else:
        # Argüman verilmediyse etkileşimli modu başlat
        run_interactive()
