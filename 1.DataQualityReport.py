from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================
# AYARLAR
# =========================
GIRDI_KLASOR = Path(r"C:\Users\erenf")
GIRDI_DOSYA = "seferihisar.csv"

HEDEF = "WS50M"   # 50 metre rüzgar hızı
FREQ = "h"        # saatlik


def masaustu_bul() -> Path:
    home = Path.home()
    for c in [home / "Desktop", home / "Masaüstü", home / "Masaustu"]:
        if c.exists() and c.is_dir():
            return c
    return home


def ardisik_eksik_bloklari(missing_ts: pd.DatetimeIndex):
    if len(missing_ts) == 0:
        return []

    miss = missing_ts.sort_values()
    blocks = []
    start = miss[0]
    prev = miss[0]
    length = 1

    for t in miss[1:]:
        if t - prev == pd.Timedelta(hours=1):
            length += 1
        else:
            blocks.append((start, prev, length))
            start = t
            length = 1
        prev = t

    blocks.append((start, prev, length))
    return blocks


def mad(x: pd.Series) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def safe_pct(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return float((part / whole) * 100.0)


def main():
    input_path = GIRDI_KLASOR / GIRDI_DOSYA
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi yok: {input_path}")

    # NASA POWER header satırlarını atla
    df = pd.read_csv(input_path, skiprows=12, encoding="utf-8-sig")

    gerekli_kolonlar = ["YEAR", "MO", "DY", "HR", HEDEF]
    eksik = [c for c in gerekli_kolonlar if c not in df.columns]
    if eksik:
        raise ValueError(f"Eksik kolon(lar): {eksik}. Mevcut kolonlar: {list(df.columns)}")

    # LST zaman damgası oluştur
    df["DateTime_LST"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["YEAR"], errors="coerce"),
            month=pd.to_numeric(df["MO"], errors="coerce"),
            day=pd.to_numeric(df["DY"], errors="coerce"),
            hour=pd.to_numeric(df["HR"], errors="coerce"),
        ),
        errors="coerce"
    )

    df = df.dropna(subset=["DateTime_LST"]).copy()

    # Sayısal dönüşüm
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")

    # NASA missing code
    df.loc[df[HEDEF] == -999, HEDEF] = np.nan

    # Yardımcı kolonlar
    df["Month"] = df["DateTime_LST"].dt.month
    df["Year"] = df["DateTime_LST"].dt.year

    if df.empty:
        raise ValueError("Veri seti boş.")

    # Duplicate timestamp kontrolü
    dup_count = int(df.duplicated(subset=["DateTime_LST"]).sum())
    if dup_count > 0:
        df = df.groupby("DateTime_LST", as_index=False).mean(numeric_only=True)
        df["Month"] = df["DateTime_LST"].dt.month
        df["Year"] = df["DateTime_LST"].dt.year

    df = df.sort_values("DateTime_LST").reset_index(drop=True)

    # Kolon bazlı eksik sayısı
    missing_values = df.isna().sum().to_dict()

    # Beklenen saat ekseni (tüm veri için doğru)
    start = df["DateTime_LST"].min()
    end = df["DateTime_LST"].max()
    full_idx = pd.date_range(start, end, freq=FREQ)
    obs_idx = pd.DatetimeIndex(df["DateTime_LST"])

    missing_ts = full_idx.difference(obs_idx)
    blocks = ardisik_eksik_bloklari(missing_ts)
    max_block = max([b[2] for b in blocks], default=0)

    expected_hours = int(len(full_idx))
    observed_hours = int(len(obs_idx))
    row_count = int(len(df))
    missing_timestamps_count = int(len(missing_ts))
    missing_rate = safe_pct(missing_timestamps_count, expected_hours)

    # Hedef seri
    s = df[HEDEF].dropna()
    if len(s) == 0:
        raise ValueError(f"{HEDEF} sütununda sayısal veri yok.")

    # Temel istatistikler
    count_val = int(s.count())
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std(ddof=1))
    var_val = float(s.var(ddof=1))
    min_val = float(s.min())
    max_val = float(s.max())

    # Çeyrekler
    q1 = float(s.quantile(0.25))
    q2 = float(s.quantile(0.50))
    q3 = float(s.quantile(0.75))
    iqr = float(q3 - q1)

    # Ek metrikler
    range_val = float(max_val - min_val)
    mad_val = mad(s)
    skew_val = float(s.skew())
    kurt_val = float(s.kurt())

    zero_count = int((s == 0).sum())
    zero_rate = safe_pct(zero_count, count_val)

    negative_count = int((s < 0).sum())
    negative_rate = safe_pct(negative_count, count_val)

    # IQR outlier
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)

    outlier_mask = (s < lower_bound) | (s > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_rate = safe_pct(outlier_count, count_val)

    cv = float(std_val / mean_val) if mean_val != 0 else None

    target_stats = {
        "count": count_val,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "var": var_val,
        "min": min_val,
        "max": max_val,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "IQR": iqr,
        "range": range_val,
        "mad": mad_val,
        "skew": skew_val,
        "kurtosis_excess": kurt_val,
        "missing_rate_percent": missing_rate,
        "zero_count": zero_count,
        "zero_rate_percent": zero_rate,
        "negative_count": negative_count,
        "negative_rate_percent": negative_rate,
        "outlier_lower_bound": lower_bound,
        "outlier_upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_rate_percent": outlier_rate,
        "cv": cv
    }

    # Yıllık istatistikler
    yearly = df.groupby("Year")[HEDEF].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()

    # Kış sezonu bazlı istatistikler (tüm veriden üretilebilir)
    winter = df[df["Month"].isin([12, 1, 2])].copy()
    winter["WinterSeasonYear"] = np.where(
        winter["Month"].isin([1, 2]),
        winter["Year"],
        winter["Year"] + 1
    )

    def is_full_season(g):
        months = set(g["Month"].unique())
        return {12, 1, 2}.issubset(months)

    winter_full = winter.groupby("WinterSeasonYear").apply(is_full_season).reset_index()
    winter_full.columns = ["WinterSeasonYear", "FullSeason"]

    winter_stats = winter.groupby("WinterSeasonYear")[HEDEF].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()

    winter_stats = winter_stats.merge(winter_full, on="WinterSeasonYear", how="left")

    # Aykırı değerler listesi
    temp_s = df[["DateTime_LST", HEDEF]].dropna().copy()
    temp_s["is_outlier"] = (temp_s[HEDEF] < lower_bound) | (temp_s[HEDEF] > upper_bound)
    outliers_df = temp_s[temp_s["is_outlier"]][["DateTime_LST", HEDEF]].copy()

    # Çıktı klasörü
    out_dir = masaustu_bul() / "bitirme-2" / "10_VeriKalitesiRaporu_WS50M"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dosyaları kaydet
    pd.DataFrame([target_stats]).to_csv(
        out_dir / "WS50M_Istatistik.csv",
        index=False,
        encoding="utf-8-sig"
    )

    yearly.to_csv(
        out_dir / "Yillik_Istatistik.csv",
        index=False,
        encoding="utf-8-sig"
    )

    winter_stats.to_csv(
        out_dir / "KisSezon_Istatistik.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"MissingTimestamp": missing_ts.astype(str)}).to_csv(
        out_dir / "EksikTimestamp_Listesi.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(blocks, columns=["Baslangic", "Bitis", "SaatSayisi"]).to_csv(
        out_dir / "EksikBloklar.csv",
        index=False,
        encoding="utf-8-sig"
    )

    outliers_df.to_csv(
        out_dir / "AykiriDegerler_IQR.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "input_path": str(input_path),
        "time_reference": "LST",
        "target_column": HEDEF,
        "date_range_start": str(start),
        "date_range_end": str(end),
        "rows": row_count,
        "columns": list(df.columns),
        "missing_values_by_column": missing_values,
        "duplicate_datetime_count_before_merge": dup_count,
        "expected_hours": expected_hours,
        "observed_hours": observed_hours,
        "missing_timestamps_count": missing_timestamps_count,
        "missing_rate_percent": missing_rate,
        "missing_blocks_count": int(len(blocks)),
        "max_consecutive_missing_hours": int(max_block),
        "target_stats": target_stats
    }

    with open(out_dir / "VeriKalitesi_Ozet.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme = (
        "10_VeriKalitesiRaporu_WS50M klasörü çıktıları:\n"
        "- VeriKalitesi_Ozet.json : genel özet + kalite metrikleri\n"
        "- EksikTimestamp_Listesi.csv : eksik saatlerin listesi\n"
        "- EksikBloklar.csv : ardışık eksik saat blokları\n"
        "- WS50M_Istatistik.csv : WS50M için temel istatistikler + Q1/Q2/Q3 + IQR + outlier\n"
        "- Yillik_Istatistik.csv : yıl bazlı temel istatistik\n"
        "- KisSezon_Istatistik.csv : kış sezonu bazlı istatistik + FullSeason\n"
        "- AykiriDegerler_IQR.csv : IQR yöntemine göre aykırı değerler\n"
        "\nNot:\n"
        "- Zaman referansı LST (Local Solar Time) olarak korunmuştur.\n"
        "- DateTime_LST, YEAR+MO+DY+HR sütunlarından oluşturulmuştur.\n"
        "- Bu script tüm veri seti için kalite raporu üretir; kış filtresi uygulanmaz.\n"
        "- Bu script eksik doldurma/temizleme yapmaz.\n"
    )

    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("✅ Veri Kalitesi Raporu üretildi:", out_dir)


if __name__ == "__main__":
    main()