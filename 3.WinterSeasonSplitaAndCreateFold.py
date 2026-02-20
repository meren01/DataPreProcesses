from pathlib import Path
import shutil
import pandas as pd

# =========================
# AYARLAR (Dokunma)
# =========================
KOK_KLASOR = "bitirme-2"

# Girdi dosyası alternatifleri (otomatik bulur)
# 1) Desktop/bitirme-2/1_KisVerisi/kis_verisi_tumu.csv
# 2) Desktop/bitirme-2/kis_aylari_butun.csv
ADAY_GIRDI_YOLLARI = [
    ("1_KisVerisi", "kis_verisi_tumu.csv"),
    ("", "kis_aylari_butun.csv"),
    ("", "kis_aylari_butun.csv".replace("butun", "bütün")),
]

TARIH = "DateTime"
HEDEF = "Speed"

# Walk-forward sezon bazlı ayarlar
ROLLING_KAC_SEZON_EGITIM = 2   # Rolling: son 2 kışla eğit, sonraki kışı test et
EXPANDING_MIN_SEZON = 2        # Expanding: en az 2 sezonla başla, büyüterek devam et

# =========================
# Yardımcılar
# =========================
def masaustu_bul() -> Path:
    home = Path.home()
    for c in [home / "Desktop", home / "Masaüstü", home / "Masaustu"]:
        if c.exists() and c.is_dir():
            return c
    return home

def giris_dosyasi_bul(root: Path) -> Path:
    denenler = []
    for folder, fname in ADAY_GIRDI_YOLLARI:
        p = root / folder / fname if folder else root / fname
        denenler.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        "Girdi dosyası bulunamadı.\nDenenen yollar:\n- " + "\n- ".join(denenler)
    )

def yaz_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def kaydet_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def kis_sezon_yili(dt: pd.Series) -> pd.Series:
    """
    Kış sezon yılı:
      - Ocak/Şubat -> aynı yıl
      - Aralık -> bir sonraki yıl
    Örn: 2023-12 -> Kış 2024
    """
    y = dt.dt.year
    m = dt.dt.month
    return y.where(m.isin([1, 2]), y + 1)

def sezon_dosya_adi(sezon_yili: int) -> str:
    return f"Kis_{sezon_yili}.csv"

# =========================
# Ana İş
# =========================
def main():
    desktop = masaustu_bul()
    root = desktop / KOK_KLASOR
    root.mkdir(parents=True, exist_ok=True)

    # --- Standart klasörler ---
    dir_girdi = root / "00_Girdi"
    dir_sezon = root / "01_KisSezonlari"
    dir_final = root / "02_FinalTest"
    dir_roll  = root / "03_CV_Rolling"
    dir_exp   = root / "04_CV_Expanding"
    dir_log   = root / "99_LOG"

    for d in [dir_girdi, dir_sezon, dir_final, dir_roll, dir_exp, dir_log]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Girdiyi bul ---
    giris = giris_dosyasi_bul(root)

    # --- Girdiyi 00_Girdi içine kopyala ---
    giris_kopya = dir_girdi / Path(giris).name
    try:
        shutil.copy2(giris, giris_kopya)
    except Exception:
        # kopyalama izin vermezse, yine de devam edelim
        giris_kopya = Path(giris)

    # --- Oku ---
    df = pd.read_csv(giris_kopya)

    if TARIH not in df.columns:
        raise ValueError(f"'{TARIH}' kolonu yok. Mevcut kolonlar: {list(df.columns)}")
    if HEDEF not in df.columns:
        raise ValueError(f"'{HEDEF}' kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    # --- DateTime parse + sıralama ---
    df[TARIH] = pd.to_datetime(df[TARIH], errors="coerce")
    df = df.dropna(subset=[TARIH]).copy()
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors="coerce")

    # --- Duplicate timestamp kontrolü (varsa ortalama al) ---
    before_rows = len(df)
    dup_count = df.duplicated(subset=[TARIH]).sum()
    if dup_count > 0:
        df = df.groupby(TARIH, as_index=False).mean(numeric_only=True)

    df = df.sort_values(TARIH).reset_index(drop=True)
    after_rows = len(df)

    # --- Kış ayı kontrolü (emin olalım, değilse sadece 12/1/2 al) ---
    months_all = set(df[TARIH].dt.month.unique().tolist())
    if not months_all.issubset({12, 1, 2}):
        df = df[df[TARIH].dt.month.isin([12, 1, 2])].copy()
        df = df.sort_values(TARIH).reset_index(drop=True)

    # --- Sezon yılı ---
    df["KisSezonYili"] = kis_sezon_yili(df[TARIH])

    # --- Sezonlara ayır ve yaz ---
    sezon_ozet = []
    sezonlar = sorted(df["KisSezonYili"].dropna().unique().astype(int).tolist())

    for s in sezonlar:
        s_df = df[df["KisSezonYili"] == s].copy()
        s_df = s_df.sort_values(TARIH).reset_index(drop=True)

        aylar = sorted(set(s_df[TARIH].dt.month.unique().tolist()))
        tam_sezon = set([12, 1, 2]).issubset(set(aylar))

        out_path = dir_sezon / sezon_dosya_adi(s)
        kaydet_csv(s_df, out_path)

        sezon_ozet.append({
            "KisSezonYili": s,
            "Baslangic": str(s_df[TARIH].min()),
            "Bitis": str(s_df[TARIH].max()),
            "SatirSayisi": len(s_df),
            "Aylar": ",".join(map(str, aylar)),
            "TamSezon": tam_sezon,
            "Dosya": str(out_path),
        })

    ozet_df = pd.DataFrame(sezon_ozet).sort_values("KisSezonYili")
    kaydet_csv(ozet_df, dir_sezon / "KisSezon_Ozet.csv")

    # --- Final test: en son TAM sezon ---
    tamlar = ozet_df[ozet_df["TamSezon"] == True].sort_values("KisSezonYili")
    if len(tamlar) < 3:
        raise ValueError(
            "Walk-forward için yeterli TAM sezon yok (en az 3 önerilir).\n"
            "01_KisSezonlari/KisSezon_Ozet.csv dosyasından kontrol et."
        )

    final_sezon = int(tamlar["KisSezonYili"].iloc[-1])
    final_test_path = tamlar["Dosya"].iloc[-1]

    # eğitim sezonları = final hariç tam sezonlar
    egitim_tamlar = tamlar.iloc[:-1].copy()
    egitim_df = pd.concat([pd.read_csv(p) for p in egitim_tamlar["Dosya"]], ignore_index=True)
    test_df = pd.read_csv(final_test_path)

    kaydet_csv(egitim_df, dir_final / "egitim_tam_sezonlar.csv")
    kaydet_csv(test_df, dir_final / f"final_test_Kis_{final_sezon}.csv")

    # --- CV sadece eğitim sezonları üzerinde ---
    cv_sezonlar = egitim_tamlar["KisSezonYili"].astype(int).tolist()
    sezon_dosya_map = {int(r["KisSezonYili"]): r["Dosya"] for _, r in tamlar.iterrows()}

    # ========== ROLLING ==========
    rolling_list = []
    deneme = 1
    for i in range(ROLLING_KAC_SEZON_EGITIM, len(cv_sezonlar)):
        train_sezonlar = cv_sezonlar[i-ROLLING_KAC_SEZON_EGITIM:i]
        test_sezon = cv_sezonlar[i]

        train_fold = pd.concat([pd.read_csv(sezon_dosya_map[s]) for s in train_sezonlar], ignore_index=True)
        test_fold = pd.read_csv(sezon_dosya_map[test_sezon])

        out_dir = dir_roll / f"Deneme_{deneme:02d}"
        kaydet_csv(train_fold, out_dir / "egitim.csv")
        kaydet_csv(test_fold, out_dir / "test.csv")

        rolling_list.append({
            "Yontem": "Rolling_Sezon",
            "DenemeNo": deneme,
            "EgitimSezonlari": ",".join(map(str, train_sezonlar)),
            "TestSezonu": test_sezon,
            "EgitimDosyasi": str(out_dir / "egitim.csv"),
            "TestDosyasi": str(out_dir / "test.csv"),
        })
        deneme += 1

    kaydet_csv(pd.DataFrame(rolling_list), dir_roll / "fold_listesi.csv")

    # ========== EXPANDING ==========
    expanding_list = []
    deneme = 1
    for i in range(EXPANDING_MIN_SEZON, len(cv_sezonlar)):
        train_sezonlar = cv_sezonlar[:i]
        test_sezon = cv_sezonlar[i]

        train_fold = pd.concat([pd.read_csv(sezon_dosya_map[s]) for s in train_sezonlar], ignore_index=True)
        test_fold = pd.read_csv(sezon_dosya_map[test_sezon])

        out_dir = dir_exp / f"Deneme_{deneme:02d}"
        kaydet_csv(train_fold, out_dir / "egitim.csv")
        kaydet_csv(test_fold, out_dir / "test.csv")

        expanding_list.append({
            "Yontem": "Expanding_Sezon",
            "DenemeNo": deneme,
            "EgitimSezonlari": ",".join(map(str, train_sezonlar)),
            "TestSezonu": test_sezon,
            "EgitimDosyasi": str(out_dir / "egitim.csv"),
            "TestDosyasi": str(out_dir / "test.csv"),
        })
        deneme += 1

    kaydet_csv(pd.DataFrame(expanding_list), dir_exp / "fold_listesi.csv")

    # --- README’ler (ne nerededir) ---
    yaz_text(dir_girdi / "README.txt",
             "00_Girdi:\n"
             "- Kullanılan girdi dosyasının kopyası burada durur.\n"
             "- Orijinal dosyan bozulmaz.\n")

    yaz_text(dir_sezon / "README.txt",
             "01_KisSezonlari:\n"
             "- Kis_YYYY.csv dosyaları: kış sezonlarına ayrılmış veriler.\n"
             "- KisSezon_Ozet.csv: her sezonun başlangıç/bitiş, satır sayısı ve tam sezon kontrolü.\n")

    yaz_text(dir_final / "README.txt",
             "02_FinalTest:\n"
             "- egitim_tam_sezonlar.csv: final test hariç tüm TAM sezonların birleşimi.\n"
             "- final_test_Kis_YYYY.csv: en son TAM sezon (asla eğitimde kullanılmaz).\n")

    yaz_text(dir_roll / "README.txt",
             "03_CV_Rolling:\n"
             "- Deneme_XX/egitim.csv ve test.csv: rolling walk-forward denemeleri.\n"
             "- fold_listesi.csv: model scriptinin okuyacağı dosya listesi.\n")

    yaz_text(dir_exp / "README.txt",
             "04_CV_Expanding:\n"
             "- Deneme_XX/egitim.csv ve test.csv: expanding walk-forward denemeleri.\n"
             "- fold_listesi.csv: model scriptinin okuyacağı dosya listesi.\n")

    # --- LOG ---
    log_text = (
        "VERI PARCALAMA LOG\n"
        f"Girdi dosyasi: {giris}\n"
        f"Kopya (kullanilan): {giris_kopya}\n"
        f"Duplicate DateTime sayisi: {int(dup_count)}\n"
        f"Satir sayisi (once): {before_rows}\n"
        f"Satir sayisi (sonra): {after_rows}\n"
        f"Uretilen sezon sayisi: {len(sezonlar)}\n"
        f"TAM sezon sayisi: {len(tamlar)}\n"
        f"Final test sezonu: Kis_{final_sezon}\n"
        f"Rolling deneme sayisi: {len(rolling_list)}\n"
        f"Expanding deneme sayisi: {len(expanding_list)}\n"
        "\nKritik dosyalar:\n"
        f"- Sezon ozet: {dir_sezon / 'KisSezon_Ozet.csv'}\n"
        f"- Final test: {dir_final / f'final_test_Kis_{final_sezon}.csv'}\n"
    )
    yaz_text(dir_log / "LOG.txt", log_text)

    print("✅ Her şey düzenli şekilde üretildi!")
    print(f"📁 Kök klasör: {root}")
    print(f"📌 Sezon özet: {dir_sezon / 'KisSezon_Ozet.csv'}")
    print(f"📌 Final test: {dir_final / f'final_test_Kis_{final_sezon}.csv'}")
    print("🔎 Bir şey karışıksa: 99_LOG/LOG.txt bak.")

if __name__ == "__main__":
    main()
