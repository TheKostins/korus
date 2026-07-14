"""
explore.py — обзорная EDA по кэшированному PISA-parquet.
Печатает: кодировки, распределения, пропуски, структуру OCOD3/STEM,
топ-профессии по полу, первый взвешённый взгляд на BSMJ~HISEI.
    .venv/bin/python src/explore.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "interim" / "pisa_full.parquet")

def hr(t): print(f"\n{'='*60}\n{t}\n{'='*60}")

hr("РАЗМЕР И КОЛОНКИ")
print(f"{df.shape[0]:,} строк × {df.shape[1]} колонок, стран: {df['CNT'].nunique()}")
core = ["BSMJ","OCOD3","HISEI","BMMJ1","BFMJ2","ESCS","HISCED","PAREDINT","ST004D01T","W_FSTUWT"]
print(df[core].dtypes.to_string())

hr("ПОЛ (ST004D01T)")
print(df["ST004D01T"].value_counts(dropna=False).to_string())

hr("ПРОПУСКИ по ключевым (доля NA, %)")
na = (df[core].isna().mean()*100).round(1).sort_values(ascending=False)
print(na.to_string())

hr("РАСПРЕДЕЛЕНИЕ BSMJ (престиж ожидаемой профессии, ISEI)")
b = df["BSMJ"].dropna()
print(f"n={len(b):,}  min={b.min():.0f}  max={b.max():.0f}  mean={b.mean():.1f}  median={b.median():.0f}")
print("квантили 10/25/50/75/90:", [round(b.quantile(q),0) for q in (.1,.25,.5,.75,.9)])

hr("HISEI (проф. статус родителей)")
h = df["HISEI"].dropna()
print(f"n={len(h):,}  min={h.min():.0f}  max={h.max():.0f}  mean={h.mean():.1f}")

hr("OCOD3 — структура кодов (ISCO-08)")
oc = df["OCOD3"].dropna().astype(str)
print("примеры значений:", oc.value_counts().head(12).to_dict())
# первая цифра = major group; спец-коды бывают нечисловыми
first = oc.str.extract(r"^(\d)")[0]
print("\nраспределение по major group (1-я цифра ISCO):")
print(first.value_counts(dropna=False).sort_index().to_string())

hr("ПЕРВЫЙ ВЗГЛЯД: взвеш. BSMJ по полу (весь пул, без учёта дизайна SE)")
d = df.dropna(subset=["BSMJ","ST004D01T","W_FSTUWT"])
for g,sub in d.groupby("ST004D01T"):
    m = np.average(sub["BSMJ"], weights=sub["W_FSTUWT"])
    print(f"  пол={g}: взвеш. BSMJ = {m:.1f}  (n={len(sub):,})")

hr("ПЕРВЫЙ ВЗГЛЯД: взвеш. корреляция BSMJ~HISEI (весь пул)")
d2 = df.dropna(subset=["BSMJ","HISEI","W_FSTUWT"])
w = d2["W_FSTUWT"].values
x, y = d2["HISEI"].values, d2["BSMJ"].values
def wcorr(x,y,w):
    mx,my = np.average(x,weights=w), np.average(y,weights=w)
    cov = np.average((x-mx)*(y-my),weights=w)
    return cov/np.sqrt(np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w))
print(f"  r = {wcorr(x,y,w):.3f}  (n={len(d2):,})  — только ориентир, не финальная оценка")
