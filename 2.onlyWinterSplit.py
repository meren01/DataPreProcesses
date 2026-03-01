from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# AYARLAR (NASA POWER / YENİ VERİ)
# =========================
BASE_DIR = Path(r"C:\Users\erenf")
INPUT_FILE = "seferihisar.csv"   # <-- yeni dosyan
SKIPROWS = 12                    # NASA POWER header satırları

HEDEF = "WS50M"                  # hedef kolon
FREQ = "h"                       # saatlik

WINTER_MONTHS = {12, 1, 2}
OUT_FOLDER_NAME = "bitirme-2"


def find_desktop() -> Path:
    home = Path.home()
    for c in [home / "Desktop", home / "Masaüstü", home / "Masaustu"]:
        if c.exists() and c.is_dir():
            return c
    return home


def main():
    # 1) Girdi
    input_path = BASE_DIR / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    # 2) Çıktı klasörleri (SEASON bazlı)
    desktop = find_desktop()
    root_out = desktop / OUT_FOLDER_NAME
    seasons_out = root_out / "sezonlar"   # <-- önceki "yıllar" yerine "sezonlar"
    root_out.mkdir(parents=True, exist_ok=True)
    seasons_out.mkdir(parents=True, exist_ok=True)

    # 3) Oku
    df = pd.read_csv(input_path, skiprows=SKIPROWS, encoding="utf-8-sig")

    # 4) Kolon kontrol
    gerekli_kolonlar = ["YEAR", "MO", "DY", "HR", HEDEF]
    eksik = [c for c in gerekli_kolonlar if c not in df.columns]
    if eksik:
        raise ValueError(f"Eksik kolon(lar): {eksik}. Mevcut kolonlar: {list(df.columns)}")

    # 5) DateTime_LST oluştur
    df["DateTime_LST"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["YEAR"], errors="coerce"),
            month=pd.to_numeric(df["MO"], errors="coerce"),
            day=pd.to_numeric(df["DY"], errors="coerce"),
            hour=pd.to_numeric(df["HR"], errors="coerce"),
        ),
        errors="coerce",
    )
    df = df.dropna(subset=["DateTime_LST"]).copy()

    # 6) Hedef sayısal + -999 => NaN
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")
    df.loc[df[HEDEF] == -999, HEDEF] = np.nan

    # 7) Yardımcı kolonlar
    df["Year"] = df["DateTime_LST"].dt.year
    df["Month"] = df["DateTime_LST"].dt.month

    # 8) Kış aylarını filtrele
    winter_df = df[df["Month"].isin(WINTER_MONTHS)].copy()

    # 9) Kış SEZONU yılı (Aralık +1)
    # 2020-12 + 2021-01 + 2021-02 => WinterSeasonYear = 2021
    winter_df["WinterSeasonYear"] = np.where(
        winter_df["Month"].isin([1, 2]),
        winter_df["Year"],
        winter_df["Year"] + 1
    )

    # 10) Birleşik kış verisi
    combined_path = root_out / "kis_aylari_butun.csv"
    winter_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    # 11) SEZON SEZON kaydet
    seasons = sorted(winter_df["WinterSeasonYear"].unique().tolist())
    for season_year in seasons:
        out_path = seasons_out / f"kis_{season_year}.csv"
        winter_df[winter_df["WinterSeasonYear"] == season_year].to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )

    # 12) Log
    log_path = root_out / "LOG.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Kış Ayları Çıktı Logu (SEZON bazlı - WinterSeasonYear)\n")
        f.write(f"Girdi: {input_path}\n")
        f.write(f"Toplam satır (temiz datetime): {len(df)}\n")
        f.write(f"Kış satır: {len(winter_df)}\n")
        f.write(f"Sezonlar (WinterSeasonYear): {seasons}\n")
        f.write(f"Birleşik dosya: {combined_path}\n")
        f.write(f"Sezon klasörü: {seasons_out}\n")

    print("✅ Tamamlandı! (SEZON bazlı)")
    print(f"📌 Okunan dosya: {input_path}")
    print(f"📁 Çıktı klasörü: {root_out}")
    print(f"📄 Birleşik kış dosyası: {combined_path}")
    print(f"📂 Sezon dosyaları: {seasons_out}")


if __name__ == "__main__":
    main()