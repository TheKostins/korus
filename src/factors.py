"""
factors.py — обзорный скрининг «других факторов» выбора профессии.
Оценивает связь набора предикторов с (1) престижем ожидаемой профессии BSMJ
и (2) вероятностью STEM-ожидания — бивариатно И в совместной модели.

ВНИМАНИЕ: взвешивание по W_FSTUWT; PV/BRR НЕ применены (это скрининг, не финал).
Пул по всем странам (риск Симпсона) — далее считать по выбранной панели.
    .venv/bin/python src/factors.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT/"data"/"interim"/"pisa_full.parquet")
W = "W_FSTUWT"

# --- деривации ---
oc = df["OCOD3"].astype(str)
SPECIAL = {"9701","9702","9703","9704","9705","9997","9998","9999","0000"}
df["valid_occ"] = ~oc.isin(SPECIAL) & oc.str.match(r"^\d{3,4}$")
df["is_stem"] = (df["valid_occ"] & oc.str[:2].isin(["21","25"])).astype(float)
df["female"] = df["ST004D01T"].map({1: 1.0, 2: 0.0})
df["math"] = df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)  # скрининг: среднее PV
# второгодничество: любой из уровней = «да» (коды >1 при валидном ответе)
rep = df[["ST127Q01TA","ST127Q02TA","ST127Q03TA"]]
df["repeated"] = ((rep > 1) & (rep < 90)).any(axis=1).astype(float)
df.loc[rep.isna().all(axis=1), "repeated"] = np.nan
# EXPECEDU — порядковая со спец-кодами 95..99 → в NaN
df["EXPECEDU"] = df["EXPECEDU"].where(df["EXPECEDU"] < 90)

FACTORS = ["HISEI","ESCS","math","SISCO","INFOSEEK","EXPECEDU","MATHMOT",
           "MATHEFF","MATHEF21","ANXMAT","CREATEFF","GROSAGR","PERSEVAGR",
           "CURIOAGR","BELONG","SDLEFF","FAMSUP","CURSUPP","TEACHSUP",
           "IMMIG","HOMEPOS","ICTRES","repeated","female","PV1READ","PV1SCIE"]
FACTORS = [f for f in FACTORS if f in df.columns]

def wcorr(x, y, w):
    m = np.isfinite(x)&np.isfinite(y)&np.isfinite(w)
    x,y,w = x[m],y[m],w[m]
    mx,my = np.average(x,weights=w),np.average(y,weights=w)
    return np.average((x-mx)*(y-my),weights=w)/np.sqrt(
        np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w)), m.sum()

def zscore(s):
    sd = s.std()
    return (s - s.mean())/sd if sd and np.isfinite(sd) else s*0.0

# ---------- 1. БИВАРИАТНО ----------
print("="*72)
print("БИВАРИАТНАЯ СВЯЗЬ (взвеш.): r с BSMJ  |  std.mean-diff со STEM (в σ)")
print("="*72)
rows=[]
for f in FACTORS:
    r,n = wcorr(df[f].values.astype(float), df["BSMJ"].values, df[W].values)
    # связь со STEM: разница стандартизов. фактора между stem и не-stem
    d = df.dropna(subset=[f,"is_stem",W]); d=d[d["valid_occ"]]
    zf = zscore(d[f].astype(float))
    stem = d["is_stem"]==1
    smd = (np.average(zf[stem],weights=d[W][stem]) -
           np.average(zf[~stem],weights=d[W][~stem]))
    rows.append((f, r, smd, n))
tab = pd.DataFrame(rows, columns=["factor","r_BSMJ","stemSMD","n"]).set_index("factor")
print(tab.reindex(tab.r_BSMJ.abs().sort_values(ascending=False).index).round(3).to_string())

# ---------- 2. СОВМЕСТНАЯ OLS для BSMJ (стандартиз. коэф.) ----------
print("\n"+"="*72)
print("СОВМЕСТНАЯ OLS: BSMJ ~ стандартиз. факторы (взвеш.) — что удерживает эффект")
print("="*72)
# только хорошо покрытые (>250k) факторы; CURSUPP/SISCO исключены (разрежены/бинарны).
# EXPECEDU включаем, но помним: это параллельное ОЖИДАНИЕ, а не «причина» (см. вывод).
joint = ["HISEI","ESCS","math","MATHEFF","ANXMAT","INFOSEEK","MATHMOT",
         "GROSAGR","CURIOAGR","CREATEFF","BELONG","FAMSUP","TEACHSUP",
         "IMMIG","HOMEPOS","repeated","female","EXPECEDU"]
joint = [f for f in joint if f in df.columns]
d = df.dropna(subset=joint+["BSMJ",W]).copy()
X = d[joint].apply(zscore); X = sm.add_constant(X)
m = sm.WLS(d["BSMJ"], X, weights=d[W]).fit()
coef = (m.params.drop("const").rename("beta").to_frame()
        .assign(t=m.tvalues.drop("const")))
print(f"n={len(d):,}  R²={m.rsquared:.3f}")
print(coef.reindex(coef.beta.abs().sort_values(ascending=False).index).round(3).to_string())
