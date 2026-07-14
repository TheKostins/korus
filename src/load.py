"""
load.py — читает нужные колонки из PISA 2022 Student .sav, фильтрует страны,
сохраняет в parquet. Также печатает покрытие целевых переменных по странам.

Запуск:
    .venv/bin/python src/load.py            # обзор покрытия по ВСЕМ странам
    .venv/bin/python src/load.py CNT1 CNT2  # + сохранить parquet по выбранным странам
"""
import sys
import zipfile
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
ZIP = RAW / "STU_QQQ_SPSS.zip"

# --- колонки, которые нам нужны (проверенные имена PISA 2022) ---
# PARED (годы образования родителей) в PISA 2022 отсутствует.
# Замены: HISCED (наивысший ISCED родителей), PAREDINT (интервальная шкала лет).
TARGET = ["OCOD3", "BSMJ"]
PREDICTORS = ["CNT", "ST004D01T", "HISEI", "BMMJ1", "BFMJ2",
              "HISCED", "PAREDINT", "ESCS", "W_FSTUWT"]
PV_MATH = [f"PV{i}MATH" for i in range(1, 11)]
PV_OTHER = ["PV1READ", "PV1SCIE"]
BRR = [f"W_FSTURWT{i}" for i in range(1, 81)]

# --- кандидаты «других факторов» (чистые WLE-индексы + пункты) ---
CAREER = ["SISCO", "INFOSEEK", "EXPECEDU", "MATHMOT"]
PSYCH = ["MATHEFF", "MATHEF21", "ANXMAT", "CREATEFF", "GROSAGR",
         "PERSEVAGR", "CURIOAGR", "BELONG", "SDLEFF"]
CONTEXT = ["FAMSUP", "CURSUPP", "TEACHSUP", "IMMIG", "HOMEPOS", "ICTRES"]
REPEAT = ["ST127Q01TA", "ST127Q02TA", "ST127Q03TA"]
EXTRA = CAREER + PSYCH + CONTEXT + REPEAT + PV_OTHER

WANT = PREDICTORS + TARGET + PV_MATH + EXTRA + BRR


def find_sav() -> Path:
    """Распаковать .sav из zip, если ещё не распакован; вернуть путь."""
    savs = [p for p in RAW.glob("**/*") if p.suffix.lower() == ".sav"]
    if savs:
        return max(savs, key=lambda p: p.stat().st_size)
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".sav")]
        if not names:
            raise SystemExit(f"В {ZIP.name} нет .sav: {z.namelist()}")
        target = max(names, key=lambda n: z.getinfo(n).file_size)
        print(f"Распаковываю {target} ({z.getinfo(target).file_size/1e9:.2f} ГБ)...")
        z.extract(target, RAW)
    savs = [p for p in RAW.glob("**/*") if p.suffix.lower() == ".sav"]
    return max(savs, key=lambda p: p.stat().st_size)


def resolve_columns(sav: Path) -> list[str]:
    """Прочитать только метаданные, сверить имена колонок с тем, что реально есть."""
    _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
    have = set(meta.column_names)
    present = [c for c in WANT if c in have]
    missing = [c for c in WANT if c not in have]
    if missing:
        print(f"[!] Нет в файле ({len(missing)}): {missing[:15]}...")
        # попробуем найти целевые по подстроке в лейблах
        for key in ("occupation", "ISEI", "expected"):
            hits = [k for k, v in meta.column_names_to_labels.items()
                    if v and key.lower() in v.lower()]
            print(f"    похожие на '{key}': {hits[:10]}")
    print(f"[ok] Читаю {len(present)} колонок.")
    return present


def main():
    keep_countries = [a.upper() for a in sys.argv[1:]]
    sav = find_sav()
    print(f"SAV: {sav} ({sav.stat().st_size/1e9:.2f} ГБ)")
    cols = resolve_columns(sav)

    df, meta = pyreadstat.read_sav(str(sav), usecols=cols)
    print(f"Прочитано: {df.shape[0]:,} строк × {df.shape[1]} колонок")

    # --- покрытие целевых переменных по странам ---
    if "BSMJ" in df and "CNT" in df:
        cov = (df.assign(has_bsmj=df["BSMJ"].notna())
                 .groupby("CNT")["has_bsmj"]
                 .agg(n="size", covered="mean")
                 .sort_values("covered", ascending=False))
        cov["covered"] = (cov["covered"] * 100).round(1)
        print("\n=== Покрытие BSMJ по странам (топ и хвост) ===")
        print(cov.head(20).to_string())
        print("...")
        print(cov.tail(10).to_string())
        cov.to_csv(INTERIM / "bsmj_coverage_by_country.csv")
        print(f"\n[saved] {INTERIM/'bsmj_coverage_by_country.csv'}")

    # всегда кэшируем полный тонкий parquet для быстрой последующей EDA
    full = INTERIM / "pisa_full.parquet"
    df.to_parquet(full, index=False)
    print(f"\n[saved] {full}: {df.shape[0]:,} строк × {df.shape[1]} колонок")

    if keep_countries:
        sub = df[df["CNT"].isin(keep_countries)].copy()
        out = INTERIM / "pisa.parquet"
        sub.to_parquet(out, index=False)
        print(f"[saved] {out}: {sub.shape[0]:,} строк для {keep_countries}")


if __name__ == "__main__":
    main()
