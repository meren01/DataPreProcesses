from pathlib import Path
import pandas as pd

WINTER_MONTHS = {12, 1, 2}


BASE_DIR = Path(r"C:\Users\erenf")
INPUT_FILE = "FEKE.csv"

DATETIME_COL = "DateTime"
SPEED_COL = "Speed"

OUT_FOLDER_NAME = "bitirme-2"

def find_desktop() -> Path:
    
    home = Path.home()
    candidates = [home / "Desktop", home / "Masaüstü", home / "Masaustu"]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return home  

def main():
    # 1) Girdi yolu
    input_path = BASE_DIR / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    # 2) Çıktı klasörleri
    desktop = find_desktop()
    root_out = desktop / OUT_FOLDER_NAME
    years_out = root_out / "yıllar"
    root_out.mkdir(parents=True, exist_ok=True)
    years_out.mkdir(parents=True, exist_ok=True)

    # 3) Oku (CSV)
    df = pd.read_csv(input_path)

    # 4) Kolon kontrol
    if DATETIME_COL not in df.columns:
        raise ValueError(f"Datetime sütunu yok: {DATETIME_COL}. Mevcut sütunlar: {list(df.columns)}")
    if SPEED_COL not in df.columns:
        raise ValueError(f"Speed sütunu yok: {SPEED_COL}. Mevcut sütunlar: {list(df.columns)}")

    # 5) DateTime parse
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL]).copy()

    # 6) Kış aylarını filtrele
    df["Year"] = df[DATETIME_COL].dt.year
    df["Month"] = df[DATETIME_COL].dt.month
    winter_df = df[df["Month"].isin(WINTER_MONTHS)].copy()

    # 7) Birleşik kış verisi kaydet
    combined_path = root_out / "kis_aylari_butun.csv"
    winter_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    # 8) Yıl yıl kaydet (timestamp yılına göre)
    years = sorted(winter_df["Year"].unique().tolist())
    for y in years:
        out_path = years_out / f"kis_{y}.csv"
        winter_df[winter_df["Year"] == y].to_csv(out_path, index=False, encoding="utf-8-sig")

    # 9) Log
    log_path = root_out / "LOG.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Kış Ayları Çıktı Logu\n")
        f.write(f"Girdi: {input_path}\n")
        f.write(f"Toplam satır: {len(df)}\n")
        f.write(f"Kış satır: {len(winter_df)}\n")
        f.write(f"Yıllar: {years}\n")
        f.write(f"Birleşik dosya: {combined_path}\n")
        f.write(f"Yıllık klasör: {years_out}\n")

    print("✅ Tamamlandı!")
    print(f"📌 Okunan dosya: {input_path}")
    print(f"📁 Çıktı klasörü: {root_out}")
    print(f"📄 Birleşik kış dosyası: {combined_path}")
    print(f"📂 Yıllık kış dosyaları: {years_out}")

if __name__ == "__main__":
    main()
