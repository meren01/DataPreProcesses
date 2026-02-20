# veri_kalitesi_raporu.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================
# AYARLAR
# =========================
GIRDI_KLASOR = Path(r"C:\Users\erenf")
GIRDI_DOSYA = "FEKE.csv"

TARIH = "DateTime"
HEDEF = "Speed"

FREQ = "h"  # hourly

def masaustu_bul() -> Path:
    home = Path.home()
    for c in [home / "Desktop", home / "Masaüstü", home / "Masaustu"]:
        if c.exists() and c.is_dir():
            return c
    return home

def ardışık_eksik_bloklari(missing_ts: pd.DatetimeIndex):
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

def main():
    input_path = GIRDI_KLASOR / GIRDI_DOSYA
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi yok: {input_path}")

    df = pd.read_csv(input_path)

    if TARIH not in df.columns:
        raise ValueError(f"{TARIH} yok. Mevcut: {list(df.columns)}")
    if HEDEF not in df.columns:
        raise ValueError(f"{HEDEF} yok. Mevcut: {list(df.columns)}")

    # Date parse
    df[TARIH] = pd.to_datetime(df[TARIH], errors="coerce")
    df = df.dropna(subset=[TARIH]).copy()

    # Speed numeric
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")

    # Duplicate timestamp kontrolü
    dup_count = int(df.duplicated(subset=[TARIH]).sum())
    if dup_count > 0:
        # Aynı saat birden fazlaysa ortalama al
        df = df.groupby(TARIH, as_index=False).mean(numeric_only=True)

    df = df.sort_values(TARIH).reset_index(drop=True)

    # Eksik değer sayıları
    missing_values = df.isna().sum().to_dict()

    # Hourly timeline ve eksik timestamp
    start, end = df[TARIH].min(), df[TARIH].max()
    full_idx = pd.date_range(start, end, freq=FREQ)
    obs_idx = pd.DatetimeIndex(df[TARIH])
    missing_ts = full_idx.difference(obs_idx)

    blocks = ardışık_eksik_bloklari(missing_ts)
    max_block = max([b[2] for b in blocks], default=0)

    # Speed istatistikleri
    s = df[HEDEF].dropna()
    speed_stats = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)),
        "var": float(s.var(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "skew": float(s.skew()),
        "kurtosis_excess": float(s.kurt()),
        "mad": mad(s)
    }

    # Yıllık istatistikler
    df["Year"] = df[TARIH].dt.year
    yearly = df.groupby("Year")[HEDEF].agg(["count","mean","median","std","min","max"]).reset_index()

    # Kış sezonu istatistikler (Aralık + Ocak/Şubat)
    df["Month"] = df[TARIH].dt.month
    winter = df[df["Month"].isin([12,1,2])].copy()
    winter["WinterSeasonYear"] = np.where(winter["Month"].isin([1,2]), winter["Year"], winter["Year"]+1)

    # Tam sezon kontrolü: 12,1,2 var mı?
    def is_full_season(g):
        months = set(g["Month"].unique())
        return {12,1,2}.issubset(months)

    winter_full = winter.groupby("WinterSeasonYear").apply(is_full_season).reset_index()
    winter_full.columns = ["WinterSeasonYear","FullSeason"]

    winter_stats = winter.groupby("WinterSeasonYear")[HEDEF].agg(["count","mean","median","std","min","max"]).reset_index()
    winter_stats = winter_stats.merge(winter_full, on="WinterSeasonYear", how="left")

    # Çıktı klasörü
    out_dir = masaustu_bul() / "bitirme-2" / "10_VeriKalitesiRaporu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kaydetmeler
    pd.DataFrame([speed_stats]).to_csv(out_dir / "Speed_Istatistik.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "Yillik_Istatistik.csv", index=False, encoding="utf-8-sig")
    winter_stats.to_csv(out_dir / "KisSezon_Istatistik.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"MissingTimestamp": missing_ts.astype(str)}).to_csv(
        out_dir / "EksikTimestamp_Listesi.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(blocks, columns=["Baslangic","Bitis","SaatSayisi"]).to_csv(
        out_dir / "EksikBloklar.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "input_path": str(input_path),
        "date_range_start": str(start),
        "date_range_end": str(end),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values_by_column": missing_values,
        "duplicate_datetime_count_before_merge": dup_count,
        "expected_hours": int(len(full_idx)),
        "observed_hours": int(len(obs_idx)),
        "missing_timestamps_count": int(len(missing_ts)),
        "missing_blocks_count": int(len(blocks)),
        "max_consecutive_missing_hours": int(max_block),
        "speed_stats": speed_stats
    }

    with open(out_dir / "VeriKalitesi_Ozet.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme = (
        "10_VeriKalitesiRaporu klasörü çıktıları:\n"
        "- VeriKalitesi_Ozet.json : genel özet + eksik timestamp sayısı + Speed istatistikleri\n"
        "- EksikTimestamp_Listesi.csv : eksik saatlerin listesi\n"
        "- EksikBloklar.csv : ardışık eksik saat blokları (başlangıç-bitiş-uzunluk)\n"
        "- Speed_Istatistik.csv : mean/median/std/MAD/quantile/skew/kurtosis\n"
        "- Yillik_Istatistik.csv : yıl bazlı temel istatistik\n"
        "- KisSezon_Istatistik.csv : kış sezonu bazlı istatistik + FullSeason etiketi\n"
        "\nNot:\n"
        "Bu script sadece KALİTE RAPORU üretir. Eksik doldurma/temizleme yapmaz.\n"
    )
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("✅ Veri Kalitesi Raporu üretildi:", out_dir)

if __name__ == "__main__":
    main()
