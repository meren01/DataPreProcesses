# preprocces.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# PREPROCESS (PIPELINE UYUMLU, ROLLING + EXPANDING)
# - 3. scriptin ürettiği fold'lardaki egitim/test csv'leri okur
# - DateTime_LST + WS50M ile çalışır (yoksa otomatik adaylardan seçer)
# - Saatliğe oturtur, kısa boşlukları interpolate eder (limitli)
# - Lag + rolling + hour_sin/cos feature üretir (leakage'siz)
# - Hedef: t+1
# - MinMaxScaler SADECE train’de fit (leakage yok)
# - Çıktı: Deneme_XX/hazir_veri/...
# ============================================================

# =========================
# AYARLAR
# =========================
PREFERRED_TIME_COL = "DateTime_LST"
PREFERRED_TARGET_COL = "WS50M"

FREQ = "H"        # saatlik
HORIZON = 1       # t+1

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]
ADD_TIME_CYCLES = True  # hour_sin/cos

INTERP_LIMIT_HOURS = 6
FILL_LONG_GAPS_WITH = None  # None: uzun boşluklar NaN kalır (en güvenlisi)

# fold_listesi kolonları (senin 3. scriptin üretiyor)
FOLD_TRAIN_COL = "EgitimDosyasi"
FOLD_TEST_COL  = "TestDosyasi"


# =========================
# YARDIMCILAR
# =========================
def find_project_root() -> Path:
    """
    En güvenlisi: scriptin olduğu klasörü proje kökü kabul et.
    (Senin durumda: C:\\Users\\erenf\\bitirme-2)
    """
    return Path(__file__).resolve().parent


def find_fold_lists(project_root: Path) -> list[Path]:
    """
    Hem Rolling hem Expanding fold_listesi.csv bulur.
    """
    candidates = [
        project_root / "03_CV_Rolling" / "fold_listesi.csv",
        project_root / "04_CV_Expanding" / "fold_listesi.csv",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        raise FileNotFoundError(
            "fold_listesi.csv bulunamadı. Beklenen yollar:\n"
            f"- {candidates[0]}\n- {candidates[1]}"
        )
    return found


def pick_column(df: pd.DataFrame, preferred: str, candidates: list[str], purpose: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"{purpose} kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")


def ensure_datetime_sorted(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """
    - time parse
    - target numeric
    - duplicate timestamp -> mean ile tekilleştir
    - time'a göre sırala
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    dup = int(df.duplicated(subset=[time_col]).sum())
    if dup > 0:
        df = df.groupby(time_col, as_index=False).mean(numeric_only=True)

    return df.sort_values(time_col).reset_index(drop=True)


def hourly_reindex(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Eksik saatleri satır olarak ekler (target NaN).
    """
    df = df.copy().set_index(time_col)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_idx)
    df.index.name = time_col
    return df.reset_index()


def limited_time_interpolate(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """
    Sadece kısa boşlukları doldurur (limit=INTERP_LIMIT_HOURS).
    Uzun boşluklar NaN kalır (None seçeneği en güvenlisi).
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    s = df.set_index(time_col)[target_col]

    s2 = s.interpolate(
        method="time",
        limit=INTERP_LIMIT_HOURS,
        limit_direction="both",
    )

    df[target_col] = s2.values

    if FILL_LONG_GAPS_WITH == "ffill":
        df[target_col] = df[target_col].ffill()
    elif isinstance(FILL_LONG_GAPS_WITH, (int, float)):
        df[target_col] = df[target_col].fillna(float(FILL_LONG_GAPS_WITH))
    # None: uzun boşluklar NaN kalır

    return df


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    hour = dt.dt.hour
    if ADD_TIME_CYCLES:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    else:
        df["hour"] = hour
    return df


def build_features(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """
    Leakage'siz:
    - Lag: s.shift(k)
    - Rolling: s.shift(1).rolling(...)
    - Target: y = s.shift(-HORIZON)  (t+1)
    """
    df = df.copy()
    s = df[target_col]

    for k in LAGS:
        df[f"lag_{k}"] = s.shift(k)

    s_shift = s.shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = s_shift.rolling(window=w, min_periods=w).mean()
        df[f"roll_std_{w}"] = s_shift.rolling(window=w, min_periods=w).std(ddof=1)

    df["y_t_plus_1"] = s.shift(-HORIZON)
    df = add_time_features(df, time_col)
    return df


def split_Xy(df_feat: pd.DataFrame, time_col: str):
    """
    Sadece engineered feature'ları alır:
    - lag_*
    - roll_mean_*, roll_std_*
    - hour_sin/hour_cos (veya hour)
    """
    y_col = "y_t_plus_1"

    engineered = []
    engineered += [f"lag_{k}" for k in LAGS]
    for w in ROLL_WINDOWS:
        engineered += [f"roll_mean_{w}", f"roll_std_{w}"]
    engineered += ["hour_sin", "hour_cos"] if ADD_TIME_CYCLES else ["hour"]

    feature_cols = [c for c in engineered if c in df_feat.columns]

    # NaN drop: feature veya y NaN ise çıkar
    valid = df_feat[feature_cols + [y_col]].notna().all(axis=1)
    clean = df_feat.loc[valid, [time_col] + feature_cols + [y_col]].reset_index(drop=True)

    X = clean[feature_cols]
    y = clean[y_col]
    times = clean[time_col]
    return X, y, times, feature_cols


def scale_fit_train_transform_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Leakage yok: scaler sadece train'e fit.
    """
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_s, X_test_s


def write_outputs(out_dir: Path, X_train, y_train, X_test, y_test, feature_cols, summary: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "egitim_featurelari.csv", index=False, encoding="utf-8-sig")
    y_train.to_csv(out_dir / "egitim_hedefi_t_plus_1.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(out_dir / "test_featurelari.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(out_dir / "test_hedefi_t_plus_1.csv", index=False, encoding="utf-8-sig")

    (out_dir / "feature_listesi.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    with open(out_dir / "preprocess_ozet.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# MAIN
# =========================
def main():
    project_root = find_project_root()
    fold_paths = find_fold_lists(project_root)

    print(f"📌 Proje kökü: {project_root}")
    print("📌 Bulunan fold listeleri:")
    for p in fold_paths:
        print("   -", p)

    for fold_path in fold_paths:
        fold_df = pd.read_csv(fold_path, encoding="utf-8-sig")

        if FOLD_TRAIN_COL not in fold_df.columns or FOLD_TEST_COL not in fold_df.columns:
            raise ValueError(
                f"{fold_path} içinde '{FOLD_TRAIN_COL}' / '{FOLD_TEST_COL}' yok.\n"
                f"Kolonlar: {list(fold_df.columns)}"
            )

        print(f"\n✅ İşleniyor: {fold_path}")
        print(f"   Deneme sayısı: {len(fold_df)}")

        for i, row in fold_df.iterrows():
            train_path = Path(row[FOLD_TRAIN_COL])
            test_path  = Path(row[FOLD_TEST_COL])

            deneme_dir = train_path.parent
            out_dir = deneme_dir / "hazir_veri"

            # 1) Okuma
            train_df0 = pd.read_csv(train_path, encoding="utf-8-sig")
            test_df0  = pd.read_csv(test_path,  encoding="utf-8-sig")

            # 2) Kolon seçimi
            time_col = pick_column(
                train_df0,
                preferred=PREFERRED_TIME_COL,
                candidates=["DateTime_LST", "DateTime", "datetime", "timestamp", "DateTime_LST"],
                purpose="Tarih/Time"
            )
            target_col = pick_column(
                train_df0,
                preferred=PREFERRED_TARGET_COL,
                candidates=["WS50M", "Speed", "ws50m", "speed"],
                purpose="Hedef/Target"
            )

            # 3) Sırala + duplicate düzelt (train/test ayrı)
            train_raw = ensure_datetime_sorted(train_df0, time_col, target_col)
            test_raw  = ensure_datetime_sorted(test_df0,  time_col, target_col)

            # 4) Saatliğe oturt
            train_hr = hourly_reindex(train_raw, time_col)
            test_hr  = hourly_reindex(test_raw,  time_col)

            # 5) İmpute (train/test ayrı)
            train_imp = limited_time_interpolate(train_hr, time_col, target_col)
            test_imp  = limited_time_interpolate(test_hr,  time_col, target_col)

            # 6) Feature üret
            train_feat = build_features(train_imp, time_col, target_col)
            test_feat  = build_features(test_imp,  time_col, target_col)

            # 7) X/y ayır
            X_train, y_train, _, feat_cols = split_Xy(train_feat, time_col)
            X_test,  y_test,  _, _        = split_Xy(test_feat,  time_col)

            # 8) Normalize (fit only train)
            X_train_s, X_test_s = scale_fit_train_transform_test(X_train, X_test)

            summary = {
                "deneme_no": int(i + 1),
                "fold_listesi": str(fold_path),
                "train_csv": str(train_path),
                "test_csv": str(test_path),
                "time_col": time_col,
                "target_col": target_col,
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
                    "train": f"{train_raw[time_col].min()} -> {train_raw[time_col].max()}",
                    "test":  f"{test_raw[time_col].min()} -> {test_raw[time_col].max()}"
                },
                "leakage_guards": [
                    "train/test asla concat edilmedi",
                    "scaler fit sadece train",
                    "rolling hesaplarda shift(1) kullanıldı",
                    "test imputasyonu sadece test içi bilgiyle yapıldı"
                ]
            }

            write_outputs(out_dir, X_train_s, y_train, X_test_s, y_test, feat_cols, summary)
            print(f"✔ {fold_path.parent.name} | Deneme {i+1}/{len(fold_df)} hazır -> {out_dir}")

    print("\n✅ Bitti. Model scriptlerin şunları okuyacak:")
    print("   - hazir_veri/egitim_featurelari.csv")
    print("   - hazir_veri/egitim_hedefi_t_plus_1.csv")
    print("   - hazir_veri/test_featurelari.csv")
    print("   - hazir_veri/test_hedefi_t_plus_1.csv")


if __name__ == "__main__":
    main()