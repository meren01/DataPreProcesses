from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================
# AYARLAR
# =========================
TARIH = "DateTime"
HEDEF = "Speed"
FREQ = "H"
HORIZON = 1  # t+1

# Feature set (klasik ML için güçlü + güvenli)
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]
ADD_TIME_CYCLES = True  # hour_sin, hour_cos

# Imputation
INTERP_LIMIT_HOURS = 6
FILL_LONG_GAPS_WITH = None  # None: uzun boşluklar NaN kalır -> sample drop (en güvenlisi)

# =========================
# Yardımcılar
# =========================
def masaustu_bul() -> Path:
    home = Path.home()
    for c in [home / "Desktop", home / "Masaüstü", home / "Masaustu"]:
        if c.exists() and c.is_dir():
            return c
    return home

def otomatik_fold_listesi_bul() -> Path:
    base = masaustu_bul() / "bitirme-2"
    candidates = [
        base / "03_CV_Rolling" / "fold_listesi.csv",
        base / "04_CV_Expanding" / "fold_listesi.csv",
        base / "3_CaprazDogrulama_Yurumeli" / "3A_Kaydirarak_Rolling_Sezon" / "fold_listesi.csv",
        base / "3_CaprazDogrulama_Yurumeli" / "3B_Buyuterek_Expanding_Sezon" / "fold_listesi.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("fold_listesi.csv bulunamadı. Klasör yolunu kontrol et.")

def ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARIH] = pd.to_datetime(df[TARIH], errors="coerce")
    df = df.dropna(subset=[TARIH]).sort_values(TARIH).reset_index(drop=True)
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")
    # Duplicate timestamp varsa ortalama ile tekilleştir (saatlik standard)
    dup = df.duplicated(subset=[TARIH]).sum()
    if dup > 0:
        df = df.groupby(TARIH, as_index=False).mean(numeric_only=True)
        df = df.sort_values(TARIH).reset_index(drop=True)
    return df

def hourly_reindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index(TARIH)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_idx)  # eksik saatler satır olarak gelir
    df.index.name = TARIH
    df = df.reset_index()
    return df

def limited_time_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARIH] = pd.to_datetime(df[TARIH])
    s = df.set_index(TARIH)[HEDEF]

    # Sadece kısa boşlukları doldur
    s2 = s.interpolate(method="time",
                       limit=INTERP_LIMIT_HOURS,
                       limit_direction="both")

    df[HEDEF] = s2.values

    if FILL_LONG_GAPS_WITH == "ffill":
        df[HEDEF] = df[HEDEF].ffill()
    elif isinstance(FILL_LONG_GAPS_WITH, (int, float)):
        df[HEDEF] = df[HEDEF].fillna(float(FILL_LONG_GAPS_WITH))
    # None ise uzun boşluklar NaN kalır
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[TARIH])
    hour = dt.dt.hour
    if ADD_TIME_CYCLES:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    else:
        df["hour"] = hour
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage’siz feature üretimi:
    - Lag: Speed.shift(k) -> sadece geçmiş
    - Rolling: Speed.shift(1).rolling(w) -> t anı bile dahil olmaz
    - Hedef: y = Speed(t+1) => shift(-1)
    """
    df = df.copy()
    s = df[HEDEF]

    for k in LAGS:
        df[f"lag_{k}"] = s.shift(k)

    s_shift = s.shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = s_shift.rolling(window=w, min_periods=w).mean()
        df[f"roll_std_{w}"] = s_shift.rolling(window=w, min_periods=w).std(ddof=1)

    df["y_t_plus_1"] = s.shift(-HORIZON)

    df = add_time_features(df)
    return df

def split_Xy(df_feat: pd.DataFrame):
    df_feat = df_feat.copy()

    y_col = "y_t_plus_1"
    drop_cols = {TARIH, HEDEF, y_col}
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]

    # Geçerli satırlar: tüm feature'lar + y dolu
    valid = df_feat[feature_cols + [y_col]].notna().all(axis=1)
    clean = df_feat.loc[valid, [TARIH] + feature_cols + [y_col]].reset_index(drop=True)

    X = clean[feature_cols]
    y = clean[y_col]
    times = clean[TARIH]
    return X, y, times, feature_cols

def scale_fit_train_transform_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # ✅ Leakage önleme: scaler sadece train’e fit
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def write_outputs(out_dir: Path, X_train, y_train, X_test, y_test, feature_cols, summary: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Anlaşılır isimler
    X_train.to_csv(out_dir / "egitim_featurelari.csv", index=False, encoding="utf-8-sig")
    y_train.to_csv(out_dir / "egitim_hedefi_t_plus_1.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(out_dir / "test_featurelari.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(out_dir / "test_hedefi_t_plus_1.csv", index=False, encoding="utf-8-sig")

    (out_dir / "feature_listesi.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    with open(out_dir / "preprocess_ozet.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    fold_path = otomatik_fold_listesi_bul()
    fold_df = pd.read_csv(fold_path)

    # Kolon isim uyumu (iki varyasyonu da destekle)
    train_col = "EgitimDosyasi" if "EgitimDosyasi" in fold_df.columns else "egitim_dosyasi"
    test_col  = "TestDosyasi" if "TestDosyasi" in fold_df.columns else "test_dosyasi"

    if train_col not in fold_df.columns or test_col not in fold_df.columns:
        raise ValueError(f"fold_listesi.csv içinde eğitim/test path kolonları yok. Kolonlar: {list(fold_df.columns)}")

    print(f"✅ fold_listesi: {fold_path}")
    print(f"   Deneme sayısı: {len(fold_df)}")

    for i, row in fold_df.iterrows():
        train_path = row[train_col]
        test_path  = row[test_col]

        # Deneme klasörü -> train dosyasının bulunduğu klasör
        deneme_dir = Path(train_path).parent
        out_dir = deneme_dir / "hazir_veri"

        # 1) Oku + sırala
        train_raw = ensure_datetime_sorted(pd.read_csv(train_path))
        test_raw  = ensure_datetime_sorted(pd.read_csv(test_path))

        # 2) Saatliğe oturt (eksik timestamp satır olarak oluşur)
        train_hr = hourly_reindex(train_raw)
        test_hr  = hourly_reindex(test_raw)

        # 3) İmpute (train/test ayrı ayrı; asla birleştirme yok)
        train_imp = limited_time_interpolate(train_hr)
        test_imp  = limited_time_interpolate(test_hr)

        # 4) Feature üret
        train_feat = build_features(train_imp)
        test_feat  = build_features(test_imp)

        # 5) X/y ayır (NaN'lı satırlar düşer)
        X_train, y_train, t_train, feat_cols = split_Xy(train_feat)
        X_test,  y_test,  t_test,  _ = split_Xy(test_feat)

        # 6) Normalize (fit sadece train)
        X_train_s, X_test_s = scale_fit_train_transform_test(X_train, X_test)

        # Özet
        summary = {
            "deneme_no": int(i + 1),
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "tahmin_ufku": "t_plus_1",
            "imputation": {
                "method": "time_interpolate_limited",
                "limit_hours": INTERP_LIMIT_HOURS,
                "fill_long_gaps_with": FILL_LONG_GAPS_WITH
            },
            "features": {
                "lags": LAGS,
                "rolling_windows": ROLL_WINDOWS,
                "time_cycles": ADD_TIME_CYCLES
            },
            "counts": {
                "train_rows_raw": int(len(train_raw)),
                "test_rows_raw": int(len(test_raw)),
                "train_rows_hourly": int(len(train_hr)),
                "test_rows_hourly": int(len(test_hr)),
                "train_samples_final": int(len(X_train_s)),
                "test_samples_final": int(len(X_test_s)),
                "feature_count": int(len(feat_cols))
            },
            "ranges": {
                "train": f"{train_raw[TARIH].min()} -> {train_raw[TARIH].max()}",
                "test":  f"{test_raw[TARIH].min()} -> {test_raw[TARIH].max()}"
            },
            "leakage_guards": [
                "train/test asla concat edilmedi",
                "scaler fit sadece train",
                "rolling hesaplarda shift(1) kullanıldı",
                "test imputasyonu sadece test içi bilgiyle yapıldı"
            ]
        }

        write_outputs(out_dir, X_train_s, y_train, X_test_s, y_test, feat_cols, summary)

        print(f"✔ Deneme {i+1}/{len(fold_df)} hazır -> {out_dir}")

    print("\n✅ Bitti. Artık model scriptlerin şunları okuyacak:")
    print("   - hazir_veri/egitim_featurelari.csv")
    print("   - hazir_veri/egitim_hedefi_t_plus_1.csv")
    print("   - hazir_veri/test_featurelari.csv")
    print("   - hazir_veri/test_hedefi_t_plus_1.csv")

if __name__ == "__main__":
    main()
