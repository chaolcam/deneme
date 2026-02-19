"""
preparation.py — Veri Hazırlama ve Ön İşleme Modülü
=====================================================
Bu dosya, Materials Project veritabanından alınan ham kristal malzeme
verilerini derin öğrenme modeline hazır hale getirir.

Yapılan işlemler sırasıyla:
  1. Ham CSV okunur, aykırı değerler ±5σ kriteri ile çıkarılır.
  2. Bant aralığı (band_gap) ve bant enerjisi (cbm) için eksik değerler
     doldurulur (metaller için band_gap = 0 kabul edilir).
  3. Energy above hull değerinden kararlılık etiketi üretilir.
  4. Uzay grubu numarası (space group) ve kararlılık etiketi one-hot
     kodlaması ile sayısal sütunlara dönüştürülür.
  5. Özellik matrisi (X) ve 4-sütunlu hedef matrisi (y) CSV olarak kaydedilir.
  6. Çıkarım (inference) sırasında kullanılmak üzere sütun adları JSON'a yazılır.
"""

import json
import pandas as pd

# ============================================================
# SABITLER — Hedef ve özellik sütun tanımları
# ============================================================

# Modelin tahmin edeceği 4 hedef değişken:
#   - formation_energy_per_atom : Formasyon enerjisi (eV/atom)
#   - band_gap                  : Bant aralığı (eV)
#   - cbm                       : İletim bandı minimum enerjisi (eV)
#   - energy_above_hull         : Hull üstü enerji — kararlılık göstergesi (eV/atom)
TARGET_COLS = ['formation_energy_per_atom', 'band_gap', 'cbm', 'energy_above_hull']

# Periyodik tablodaki 103 elementin sembolü (H'dan Lr'ye kadar).
# Bu sütunlar, bir kristaldeki her elementin mol kesrini (fraksiyon) tutar.
# Örneğin Fe2O3 için Fe=0.40, O=0.60 olur.
ELEMENT_COLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr'
]

# Bileşimden türetilen fiziksel tanımlayıcılar (descriptor):
#   - n_atoms            : Birim hücresindeki toplam atom sayısı
#   - n_elements         : Farklı element sayısı
#   - avg_atomic_mass    : Ağırlıklı ortalama atom kütlesi (g/mol)
#   - en_mean/max/min/range : Elektronegativite istatistikleri (Pauling)
#   - avg_covalent_radius : Ortalama kovalent yarıçap (Å)
#   - ea_mean/max/min/range : Elektron ilgisi istatistikleri (eV)
DESCRIPTOR_COLS = [
    'n_atoms', 'n_elements', 'avg_atomic_mass', 'en_mean',
    'en_max', 'en_min', 'en_range', 'avg_covalent_radius',
    'ea_mean', 'ea_max', 'ea_min', 'ea_range'
]


# ============================================================
# FONKSİYON 1 — Kararlılık Sınıflandırma
# ============================================================
def classify_stability(e_hull):
    """
    Energy above hull (hull üstü enerji) değerine göre
    malzemenin kararlılık kategorisini belirler.

    Eşik değerleri Materials Project literatüründen alınmıştır:
      ≤ 0.025 eV/atom → Stable    (kararlı, sentezlenebilir)
      ≤ 0.100 eV/atom → Metastable (meta-kararlı, koşullu kararlı)
      > 0.100 eV/atom → Unstable  (kararsız)

    Parametreler:
        e_hull (float): Hull üstü enerji değeri (eV/atom)

    Döndürür:
        str: "Stable", "Metastable" veya "Unstable"
    """
    if e_hull <= 0.025:
        return "Stable"
    elif e_hull <= 0.100:
        return "Metastable"
    else:
        return "Unstable"


# ============================================================
# FONKSİYON 2 — Ham Veriyi Yükle ve Filtrele
# ============================================================
def load_and_filter_data(csv_path):
    """
    Ham Materials Project CSV dosyasını okur, aykırı değerleri temizler
    ve kararlılık etiketlerini ekler.

    İşlem adımları:
      1. CSV pandas DataFrame'e yüklenir.
      2. formation_energy_per_atom için ortalama (μ) ve standart sapma (σ)
         hesaplanır; μ ± 5σ aralığı dışındaki satırlar çıkarılır.
         Bu adım, DFT hesaplama hatalarından kaynaklanan uç değerleri eler.
      3. Metalik malzemelerde band_gap = 0 ve cbm = Fermi seviyesi olduğu
         için bu sütunlardaki NaN değerler 0 ile doldurulur.
      4. energy_above_hull sütunundan classify_stability() ile etiket üretilir.
      5. ML için gereken sütunlar seçilir; CSV'de bulunmayan sütunlar
         (varsa) sessizce atlanır.
      6. Aynı (formül + uzay grubu + kararlılık) üçlüsü için yalnızca
         en düşük formasyon enerjili satır saklanır — duplikasyon önlenir.

    Parametreler:
        csv_path (str): Ham veri CSV dosyasının yolu

    Döndürür:
        pd.DataFrame: Temizlenmiş ve etiketlenmiş veri çerçevesi
    """
    # Ham veriyi oku
    df = pd.read_csv(csv_path)

    # --- Aykırı değer filtresi (±5σ) ---
    # Formation enerjisinin ortalaması ve standart sapması hesaplanır.
    # 5 sigma dışındaki noktalar büyük ihtimalle DFT hesaplama hatasıdır.
    mean = df['formation_energy_per_atom'].mean()
    std = df['formation_energy_per_atom'].std()
    lower, upper = mean - 5 * std, mean + 5 * std

    df_filtered = df[
        (df['formation_energy_per_atom'] >= lower) &
        (df['formation_energy_per_atom'] <= upper)
    ].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    # --- Eksik bant özellikleri doldur ---
    # Metal malzemelerde band_gap = 0 ve cbm Fermi düzeyinde olduğundan
    # bu değerler 0 ile doldurulur. Bu, Materials Project'in davranışıyla tutarlıdır.
    for col in ['band_gap', 'cbm']:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].fillna(0.0)

    # --- Kararlılık etiketi üret ---
    # Her satır için energy_above_hull değerine bakılarak etiket atanır.
    df_filtered["stability_label"] = df_filtered["energy_above_hull"].apply(classify_stability)

    # --- İlgili sütunları seç ---
    # Yalnızca ML pipeline için gerekli olan sütunlar saklanır.
    # CSV'de bulunmayan sütunlar liste dışı bırakılır (hata önleme).
    columns_to_keep = [
        'material_id', 'formula_pretty',
        'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'cbm',
        'crystal_system', 'number', 'symbol', 'point_group',
    ] + ELEMENT_COLS + DESCRIPTOR_COLS + ['stability_label']

    columns_to_keep = [c for c in columns_to_keep if c in df_filtered.columns]
    df_filtered = df_filtered[columns_to_keep]

    # --- Duplikasyon önleme ---
    # Aynı kimyasal formül + uzay grubu + kararlılık etiketi için
    # birden fazla satır varsa en düşük formasyon enerjili olanı tut.
    # Bu, eğitim verisindeki bilgi sızıntısını (data leakage) azaltır.
    df_lowest_energy = (
        df_filtered.sort_values('formation_energy_per_atom')
        .drop_duplicates(
            subset=['formula_pretty', 'number', 'stability_label'],
            keep='first'
        )
        .reset_index(drop=True)
    )

    return df_lowest_energy


# ============================================================
# FONKSİYON 3 — ML İçin Ön İşleme
# ============================================================
def preprocess_material_data(df, fill_strategy="zero"):
    """
    Temizlenmiş DataFrame'i makine öğrenmesine hazır X (özellik matrisi)
    ve y (çok sütunlu hedef matrisi) çiftine dönüştürür.

    İşlem adımları:
      1. formation_energy_per_atom eksik olan satırlar düşürülür.
      2. Diğer hedef sütunlardaki NaN değerler 0 ile doldurulur.
      3. Uzay grubu ('number') ve kararlılık etiketi ('stability_label')
         one-hot kodlaması ile sayısal sütunlara çevrilir:
           Örnek: number=225 → sg_225=1, diğer sg_* = 0
      4. bool tipindeki one-hot sütunlar int'e dönüştürülür
         (PyTorch float32 tensörleriyle uyum için).
      5. Özellik sütunları (element + descriptor + one-hot) birleştirilir.
      6. Eksik değerler fill_strategy'e göre doldurulur.
      7. Hedef değişkenler 4-sütunlu y DataFrame'i olarak döndürülür.

    Parametreler:
        df           (pd.DataFrame): load_and_filter_data() çıktısı
        fill_strategy (str)        : Eksik değer stratejisi — 'zero' veya 'mean'

    Döndürür:
        X (pd.DataFrame): Özellik matrisi
        y (pd.DataFrame): Hedef matrisi (4 sütun)
    """
    # Birincil hedef eksikse satırı düşür
    df = df.dropna(subset=['formation_energy_per_atom']).copy()

    # İkincil hedeflerdeki NaN'leri doldur
    for col in ['band_gap', 'cbm', 'energy_above_hull']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # --- One-hot kodlama ---
    # 'number' (uzay grubu, 1–230) → sg_1, sg_2, ..., sg_230
    # 'stability_label' (Stable/Metastable/Unstable) → stab_Stable, stab_Metastable, stab_Unstable
    # pd.get_dummies(), her benzersiz değer için yeni bir 0/1 sütunu oluşturur.
    df_encoded = pd.get_dummies(
        df,
        columns=["number", "stability_label"],
        prefix=["sg", "stab"]
    )

    # One-hot sütunlar bazen bool olarak gelir; PyTorch uyumluluğu için int'e çevir
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)

    # One-hot kodlanmış sütun isimlerini topla
    dummy_cols = [
        col for col in df_encoded.columns
        if col.startswith("sg_") or col.startswith("stab_")
    ]

    # Yalnızca veri setinde mevcut olan özellik sütunlarını kullan
    feature_cols = [c for c in ELEMENT_COLS + DESCRIPTOR_COLS if c in df_encoded.columns]

    # Tüm özellik sütunlarını birleştir: element fraksiyonları + tanımlayıcılar + one-hot
    all_features = feature_cols + dummy_cols

    # --- Eksik değer doldurma ---
    # 'zero' : Olmayan element için fraksiyon = 0 (en yaygın ve mantıklı seçim)
    # 'mean' : Eksik değeri sütun ortalamasıyla doldur
    if fill_strategy == "zero":
        X = df_encoded[all_features].fillna(0)
    elif fill_strategy == "mean":
        X = df_encoded[all_features].fillna(df_encoded[all_features].mean())
    else:
        raise ValueError("fill_strategy 'zero' veya 'mean' olmalıdır.")

    # --- Çok hedefli çıktı ---
    # Yalnızca veri setinde mevcut olan hedef sütunları al
    available_targets = [c for c in TARGET_COLS if c in df_encoded.columns]
    y = df_encoded[available_targets]

    return X, y


# ============================================================
# ANA ÇALIŞMA BLOĞU — Bu dosya doğrudan çalıştırıldığında
# ============================================================
if __name__ == "__main__":
    # Ham veri dosyasının yolu
    data_path = "data/MP_queried_data_featurized_w_additional_acr_ae_en.csv"

    print("=" * 60)
    print("  VERİ HAZIRLAMA AŞAMASI")
    print("=" * 60)

    # Adım 1: Ham veriyi yükle, temizle, etiketle
    print("\n[1/4] Ham veri yükleniyor ve temizleniyor...")
    df_clean = load_and_filter_data(data_path)
    print(f"      Temizlenmiş veri seti boyutu : {df_clean.shape[0]:,} satır × {df_clean.shape[1]} sütun")

    # Kararlılık dağılımını göster
    if 'stability_label' in df_clean.columns:
        dist = df_clean['stability_label'].value_counts()
        print(f"      Kararlılık dağılımı:")
        for label, count in dist.items():
            print(f"        {label:<15}: {count:>8,} ({count/len(df_clean)*100:.1f}%)")

    # Adım 2: Özellik ve hedef matrislerini oluştur
    print("\n[2/4] Özellik mühendisliği ve kodlama yapılıyor...")
    X, y = preprocess_material_data(df_clean, fill_strategy="zero")
    print(f"      Özellik matrisi (X) boyutu : {X.shape[0]:,} örnek × {X.shape[1]} özellik")
    print(f"      Hedef matrisi   (y) boyutu : {y.shape[0]:,} örnek × {y.shape[1]} hedef")
    print(f"      Tahmin edilecek hedefler   : {y.columns.tolist()}")

    # Adım 3: Ön işlenmiş veriyi kaydet
    print("\n[3/4] Ön işlenmiş veriler kaydediliyor...")
    X.to_csv("data/X_preprocessed.csv", index=False)
    y.to_csv("data/y_preprocessed.csv", index=False)
    print("      data/X_preprocessed.csv  ✓")
    print("      data/y_preprocessed.csv  ✓")

    # Adım 4: Çıkarım (inference) için sütun meta verisini kaydet
    # predict_crystal.py, yeni bir kristal girildiğinde tam olarak
    # hangi sütunların ve hangi sırayla bekleneceğini buradan öğrenir.
    print("\n[4/4] Sütun meta verisi kaydediliyor...")
    with open("data/feature_columns.json", "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    with open("data/target_columns.json", "w") as f:
        json.dump(y.columns.tolist(), f, indent=2)

    print(f"      data/feature_columns.json ({len(X.columns)} özellik sütunu)  ✓")
    print(f"      data/target_columns.json  ({len(y.columns)} hedef sütunu)    ✓")

    print("\n" + "=" * 60)
    print("  HAZIRLIK TAMAMLANDI — Sıradaki adım: python train.py")
    print("=" * 60)
