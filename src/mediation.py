"""
mediation.py — проверка гипотезы: часть «факторов» (мат. самоэффективность и др.)
не независимы, а являются СЛЕДСТВИЯМИ успеваемости/психологии.

Три взгляда:
  1) взвеш. корреляции между самими факторами (насколько переплетены);
  2) иерархическая регрессия блоками в причинном порядке (усыхание коэффициентов);
  3) явная декомпозиция медиации для двух цепочек:
       math   -> MATHEFF -> BSMJ  (самоэффективность как следствие успеваемости)
       HISEI  -> math    -> BSMJ  (H1: статус родителей через успеваемость)

ВНИМАНИЕ: PISA кросс-секционна → «медиация» здесь СТАТИСТИЧЕСКАЯ декомпозиция при
ПРЕДПОЛОЖЕННОМ порядке, а не доказанная причинность (связь может быть взаимной).
"""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

df = pd.read_parquet(Path(__file__).resolve().parent.parent/"data"/"interim"/"pisa_full.parquet")
W = "W_FSTUWT"
df["math"] = df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)
df["female"] = df["ST004D01T"].map({1: 1.0, 2: 0.0})

def z(s): return (s - s.mean())/s.std()
def wls(y, Xcols, d):
    X = sm.add_constant(d[Xcols].apply(z))
    return sm.WLS(d[y], X, weights=d[W]).fit()

# --- 1. корреляции между факторами ---
fac = ["HISEI","math","MATHEFF","MATHEF21","ANXMAT","CURIOAGR","GROSAGR","CREATEFF","BELONG"]
d0 = df.dropna(subset=fac+[W])
def wcorrmat(d, cols):
    w = d[W].values; M = {}
    for a in cols:
        row = {}
        xa = d[a].values; ma = np.average(xa, weights=w)
        for b in cols:
            xb = d[b].values; mb = np.average(xb, weights=w)
            cov = np.average((xa-ma)*(xb-mb), weights=w)
            row[b] = cov/np.sqrt(np.average((xa-ma)**2,weights=w)*np.average((xb-mb)**2,weights=w))
        M[a] = row
    return pd.DataFrame(M).T[cols]
print("="*72); print("1) Взвеш. корреляции между факторами (n=%d)"%len(d0)); print("="*72)
print(wcorrmat(d0, fac).round(2).to_string())

# --- 2. иерархические блоки ---
print("\n"+"="*72); print("2) Иерархическая регрессия BSMJ: коэф. по блокам (стандартиз.)"); print("="*72)
B1 = ["female","IMMIG","HISEI"]                 # фон (задан рано)
B2 = B1 + ["math"]                              # + успеваемость
B3 = B2 + ["MATHEFF","CURIOAGR","GROSAGR","ANXMAT","CREATEFF","BELONG"]  # + психология (проксимальная)
d = df.dropna(subset=B3+["BSMJ",W]).copy()
track = ["HISEI","math","MATHEFF","CURIOAGR"]
res = {}
for name, cols in [("M1 фон",B1),("M2 +успев.",B2),("M3 +психол.",B3)]:
    m = wls("BSMJ", cols, d)
    res[name] = {**{k: m.params.get(k, np.nan) for k in track}, "R2": m.rsquared}
print(pd.DataFrame(res).T.round(3).to_string())
print("→ смотрим, как β(HISEI) и β(math) усыхают при добавлении блока-следствия")

# --- 3. явная медиация ---
def mediate(x, med, y, d):
    dd = d.dropna(subset=[x, med, y, W]).copy()
    for c in (x, med): dd[c] = z(dd[c])
    c_tot = sm.WLS(dd[y], sm.add_constant(dd[[x]]), weights=dd[W]).fit().params[x]      # total
    a     = sm.WLS(dd[med], sm.add_constant(dd[[x]]), weights=dd[W]).fit().params[x]    # x->med
    mfull = sm.WLS(dd[y], sm.add_constant(dd[[x, med]]), weights=dd[W]).fit()
    c_dir, b = mfull.params[x], mfull.params[med]                                       # direct, med->y
    ind = a*b
    return c_tot, c_dir, ind, ind/c_tot*100, len(dd)

print("\n"+"="*72); print("3) Декомпозиция медиации (BSMJ; при предположенном порядке)"); print("="*72)
for x, med, lbl in [("math","MATHEFF","math → MATHEFF → BSMJ"),
                    ("HISEI","math","HISEI → math → BSMJ (H1)")]:
    tot, dir_, ind, pct, n = mediate(x, med, "BSMJ", df)
    print(f"\n  {lbl}   (n={n:,})")
    print(f"    полный эффект {x}:  {tot:+.3f}")
    print(f"    прямой (при контроле {med}): {dir_:+.3f}")
    print(f"    косвенный через {med}: {ind:+.3f}  = {pct:.0f}% полного")
