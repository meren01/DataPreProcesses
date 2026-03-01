from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def grafik_kaydet(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    # DateTime_LST oluştur
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

    # Hedef sayısal dönüşüm
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")

    # NASA missing code
    df.loc[df[HEDEF] == -999, HEDEF] = np.nan

    # Duplicate timestamp kontrolü
    dup_count = int(df.duplicated(subset=["DateTime_LST"]).sum())
    if dup_count > 0:
        df = df.groupby("DateTime_LST", as_index=False).mean(numeric_only=True)

    df = df.sort_values("DateTime_LST").reset_index(drop=True)

    if df.empty:
        raise ValueError("Veri seti boş.")

    # Yardımcı kolonlar
    df["Year"] = df["DateTime_LST"].dt.year
    df["Month"] = df["DateTime_LST"].dt.month
    df["YearMonth"] = df["DateTime_LST"].dt.to_period("M").astype(str)

    # Kış verisi
    winter_df = df[df["Month"].isin([12, 1, 2])].copy()
    winter_df["WinterSeasonYear"] = np.where(
        winter_df["Month"].isin([1, 2]),
        winter_df["Year"],
        winter_df["Year"] + 1
    )

    # Çıktı klasörü
    out_dir = masaustu_bul() / "bitirme-2" / "11_VeriKalitesiGorselleri_WS50M"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1) Tüm veri - Zaman serisi
    # =========================
    fig = plt.figure(figsize=(14, 5))
    plt.plot(df["DateTime_LST"], df[HEDEF])
    plt.title("WS50M Zaman Serisi")
    plt.xlabel("DateTime_LST")
    plt.ylabel("WS50M (m/s)")
    plt.grid(True, alpha=0.3)
    grafik_kaydet(fig, out_dir / "01_WS50M_ZamanSerisi.png")

    # =========================
    # 2) Tüm veri - Histogram
    # =========================
    fig = plt.figure(figsize=(10, 5))
    plt.hist(df[HEDEF].dropna(), bins=30)
    plt.title("WS50M Histogram")
    plt.xlabel("WS50M (m/s)")
    plt.ylabel("Frekans")
    plt.grid(True, alpha=0.3)
    grafik_kaydet(fig, out_dir / "02_WS50M_Histogram.png")

    # =========================
    # 3) Tüm veri - Boxplot
    # =========================
    fig = plt.figure(figsize=(8, 5))
    plt.boxplot(df[HEDEF].dropna(), vert=True)
    plt.title("WS50M Boxplot")
    plt.ylabel("WS50M (m/s)")
    plt.grid(True, alpha=0.3)
    grafik_kaydet(fig, out_dir / "03_WS50M_Boxplot.png")

    # =========================
    # 4) Tüm veri - Yıllık ortalama
    # =========================
    yearly_mean = df.groupby("Year")[HEDEF].mean().reset_index()

    fig = plt.figure(figsize=(10, 5))
    plt.bar(yearly_mean["Year"].astype(str), yearly_mean[HEDEF])
    plt.title("Yıllık Ortalama WS50M")
    plt.xlabel("Yıl")
    plt.ylabel("Ortalama WS50M (m/s)")
    plt.grid(True, axis="y", alpha=0.3)
    grafik_kaydet(fig, out_dir / "04_Yillik_Ortalama_WS50M.png")

    # =========================
    # 5) Tüm veri - Aylık ortalama
    # =========================
    monthly_mean = df.groupby("Month")[HEDEF].mean().reindex(range(1, 13)).reset_index()

    fig = plt.figure(figsize=(10, 5))
    plt.bar(monthly_mean["Month"].astype(str), monthly_mean[HEDEF])
    plt.title("Aylık Ortalama WS50M")
    plt.xlabel("Ay")
    plt.ylabel("Ortalama WS50M (m/s)")
    plt.grid(True, axis="y", alpha=0.3)
    grafik_kaydet(fig, out_dir / "05_Aylik_Ortalama_WS50M.png")

    # =========================
    # 6) Tüm veri - Aylık eksik değer sayısı
    # =========================
    monthly_missing = (
        df.assign(YearMonth=df["DateTime_LST"].dt.to_period("M").astype(str))
          .groupby("YearMonth")[HEDEF]
          .apply(lambda x: x.isna().sum())
          .reset_index(name="MissingCount")
    )

    fig = plt.figure(figsize=(14, 5))
    plt.bar(monthly_missing["YearMonth"], monthly_missing["MissingCount"])
    plt.title("Aylık Eksik WS50M Değer Sayısı")
    plt.xlabel("Yıl-Ay")
    plt.ylabel("Eksik Değer Sayısı")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", alpha=0.3)
    grafik_kaydet(fig, out_dir / "06_Aylik_EksikDeger_Sayisi.png")

    # =========================
    # 7) Tüm veri - Yıllara göre boxplot
    # =========================
    years_sorted = sorted(df["Year"].dropna().unique())
    box_data = [df.loc[df["Year"] == y, HEDEF].dropna().values for y in years_sorted]

    fig = plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, tick_labels=[str(y) for y in years_sorted], vert=True)
    plt.title("Yıllara Göre WS50M Dağılımı")
    plt.xlabel("Yıl")
    plt.ylabel("WS50M (m/s)")
    plt.grid(True, axis="y", alpha=0.3)
    grafik_kaydet(fig, out_dir / "07_Yillara_Gore_Boxplot_WS50M.png")

    # =========================
    # 8) Tüm veri - Aylık ortalama zaman serisi
    # =========================
    monthly_ts = (
        df.set_index("DateTime_LST")[HEDEF]
          .resample("ME")
          .mean()
          .reset_index()
    )

    fig = plt.figure(figsize=(14, 5))
    plt.plot(monthly_ts["DateTime_LST"], monthly_ts[HEDEF])
    plt.title("Aylık Ortalama WS50M Zaman Serisi")
    plt.xlabel("DateTime_LST")
    plt.ylabel("Aylık Ortalama WS50M (m/s)")
    plt.grid(True, alpha=0.3)
    grafik_kaydet(fig, out_dir / "08_Aylik_Ortalama_ZamanSerisi_WS50M.png")

    # =========================
    # KIŞ GRAFİKLERİ
    # =========================
    if not winter_df.empty:
        # 9) Kış - Zaman serisi
        fig = plt.figure(figsize=(14, 5))
        plt.plot(winter_df["DateTime_LST"], winter_df[HEDEF])
        plt.title("Kış Ayları WS50M Zaman Serisi (Aralık-Ocak-Şubat)")
        plt.xlabel("DateTime_LST")
        plt.ylabel("WS50M (m/s)")
        plt.grid(True, alpha=0.3)
        grafik_kaydet(fig, out_dir / "09_Kis_WS50M_ZamanSerisi.png")

        # 10) Kış - Histogram
        fig = plt.figure(figsize=(10, 5))
        plt.hist(winter_df[HEDEF].dropna(), bins=30)
        plt.title("Kış Ayları WS50M Histogram")
        plt.xlabel("WS50M (m/s)")
        plt.ylabel("Frekans")
        plt.grid(True, alpha=0.3)
        grafik_kaydet(fig, out_dir / "10_Kis_WS50M_Histogram.png")

        # 11) Kış - Boxplot
        fig = plt.figure(figsize=(8, 5))
        plt.boxplot(winter_df[HEDEF].dropna(), vert=True)
        plt.title("Kış Ayları WS50M Boxplot")
        plt.ylabel("WS50M (m/s)")
        plt.grid(True, alpha=0.3)
        grafik_kaydet(fig, out_dir / "11_Kis_WS50M_Boxplot.png")

        # 12) Kış - Ay bazlı ortalama (12,1,2)
        winter_monthly_mean = winter_df.groupby("Month")[HEDEF].mean().reindex([12, 1, 2]).reset_index()

        fig = plt.figure(figsize=(8, 5))
        plt.bar(winter_monthly_mean["Month"].astype(str), winter_monthly_mean[HEDEF])
        plt.title("Kış Ayları Ortalama WS50M")
        plt.xlabel("Ay (12=Aralık, 1=Ocak, 2=Şubat)")
        plt.ylabel("Ortalama WS50M (m/s)")
        plt.grid(True, axis="y", alpha=0.3)
        grafik_kaydet(fig, out_dir / "12_Kis_Aylik_Ortalama_WS50M.png")

        # 13) Kış sezonlarına göre boxplot
        winter_seasons = sorted(winter_df["WinterSeasonYear"].dropna().unique())
        winter_box_data = [
            winter_df.loc[winter_df["WinterSeasonYear"] == y, HEDEF].dropna().values
            for y in winter_seasons
        ]

        fig = plt.figure(figsize=(12, 6))
        plt.boxplot(winter_box_data, tick_labels=[str(y) for y in winter_seasons], vert=True)
        plt.title("Kış Sezonlarına Göre WS50M Dağılımı")
        plt.xlabel("Kış Sezonu Yılı")
        plt.ylabel("WS50M (m/s)")
        plt.grid(True, axis="y", alpha=0.3)
        grafik_kaydet(fig, out_dir / "13_KisSezon_Yillarina_Gore_Boxplot_WS50M.png")

    # README
    readme = (
        "11_VeriKalitesiGorselleri_WS50M klasörü çıktıları:\n"
        "- 01_WS50M_ZamanSerisi.png : tüm veri saatlik zaman serisi\n"
        "- 02_WS50M_Histogram.png : tüm veri dağılım histogramı\n"
        "- 03_WS50M_Boxplot.png : tüm veri genel boxplot\n"
        "- 04_Yillik_Ortalama_WS50M.png : yıllık ortalama bar grafiği\n"
        "- 05_Aylik_Ortalama_WS50M.png : ay bazlı ortalama bar grafiği\n"
        "- 06_Aylik_EksikDeger_Sayisi.png : ay bazlı eksik değer sayısı\n"
        "- 07_Yillara_Gore_Boxplot_WS50M.png : yıl bazlı boxplot\n"
        "- 08_Aylik_Ortalama_ZamanSerisi_WS50M.png : aylık ortalama zaman serisi\n"
        "- 09_Kis_WS50M_ZamanSerisi.png : kış ayları zaman serisi\n"
        "- 10_Kis_WS50M_Histogram.png : kış ayları histogramı\n"
        "- 11_Kis_WS50M_Boxplot.png : kış ayları boxplot\n"
        "- 12_Kis_Aylik_Ortalama_WS50M.png : Aralık-Ocak-Şubat ortalamaları\n"
        "- 13_KisSezon_Yillarina_Gore_Boxplot_WS50M.png : kış sezonlarına göre boxplot\n"
        "\nNot:\n"
        "- Zaman referansı LST (Local Solar Time) olarak korunmuştur.\n"
        "- DateTime_LST, YEAR+MO+DY+HR sütunlarından oluşturulmuştur.\n"
        "- Hedef değişken olarak WS50M kullanılmıştır.\n"
        "- Kış ayları Aralık, Ocak ve Şubat olarak alınmıştır.\n"
    )
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("✅ Tüm veri ve kış görselleri üretildi:", out_dir)


if __name__ == "__main__":
    main()