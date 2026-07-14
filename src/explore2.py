"""explore2.py — топ-профессии, доля STEM, гетерогенность по странам."""
from pathlib import Path
import numpy as np, pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT/"data"/"interim"/"pisa_full.parquet")
_, meta = pyreadstat.read_sav("data/raw/CY08MSP_STU_QQQ.SAV", metadataonly=True)
lab = {str(k): v for k, v in meta.variable_value_labels.get("OCOD3", {}).items()}

SPECIAL = {"9701","9702","9703","9704","9705","9997","9998","9999","0000"}
oc = df["OCOD3"].astype(str)
valid = ~oc.isin(SPECIAL) & oc.str.match(r"^\d{3,4}$")
df = df.assign(oc=oc, valid=valid)

# STEM = science/engineering (21) + ICT (25); health (22) — отдельно
smg = oc.str[:2]  # sub-major group
df["is_stem"] = valid & smg.isin(["21","25"])
df["is_health"] = valid & smg.eq("22")

def hr(t): print(f"\n{'='*60}\n{t}\n{'='*60}")
w = "W_FSTUWT"

hr("ДОЛЯ STEM-ОЖИДАНИЙ по полу (валидные ответы, взвеш.)")
v = df[df.valid & df.ST004D01T.isin([1,2])]
for g,name in [(1,"девочки"),(2,"мальчики")]:
    s = v[v.ST004D01T==g]
    stem = np.average(s["is_stem"], weights=s[w])
    hlt = np.average(s["is_health"], weights=s[w])
    print(f"  {name}: STEM={stem*100:.1f}%   health={hlt*100:.1f}%   (n={len(s):,})")

hr("ТОП-10 ожидаемых профессий по полу (валидные, взвеш.)")
for g,name in [(1,"ДЕВОЧКИ"),(2,"МАЛЬЧИКИ")]:
    s = v[v.ST004D01T==g]
    top = s.groupby("oc")[w].sum().sort_values(ascending=False).head(10)
    tot = s[w].sum()
    print(f"\n  {name}:")
    for code,wt in top.items():
        print(f"    {wt/tot*100:4.1f}%  {code}  {lab.get(code,'?')[:45]}")

hr("ГЕТЕРОГЕННОСТЬ ПО СТРАНАМ: r(BSMJ,HISEI) и STEM-gap (девочки-мальчики, п.п.)")
def wcorr(x,y,wt):
    mx,my=np.average(x,weights=wt),np.average(y,weights=wt)
    return np.average((x-mx)*(y-my),weights=wt)/np.sqrt(
        np.average((x-mx)**2,weights=wt)*np.average((y-my)**2,weights=wt))
rows=[]
for cnt,sub in df.groupby("CNT"):
    d=sub.dropna(subset=["BSMJ","HISEI",w])
    if len(d)<500: continue
    r=wcorr(d["HISEI"].values,d["BSMJ"].values,d[w].values)
    vv=sub[sub.valid & sub.ST004D01T.isin([1,2])]
    gf=vv[vv.ST004D01T==1]; gm=vv[vv.ST004D01T==2]
    gap=(np.average(gf["is_stem"],weights=gf[w])-np.average(gm["is_stem"],weights=gm[w]))*100
    rows.append((cnt,len(d),r,gap))
res=pd.DataFrame(rows,columns=["CNT","n","r_HISEI","stem_gap_pp"]).set_index("CNT")
focus=["KOR","JPN","SGP","ESP","PRT","IRL","USA","AUS","KAZ","MEX","COL","CRI","MDA","TUR","VNM"]
print(res.loc[[c for c in focus if c in res.index]].round(3).to_string())
print(f"\n  диапазон r по {len(res)} странам: {res.r_HISEI.min():.3f}..{res.r_HISEI.max():.3f}, медиана {res.r_HISEI.median():.3f}")
print(f"  диапазон STEM-gap: {res.stem_gap_pp.min():.1f}..{res.stem_gap_pp.max():.1f} п.п., медиана {res.stem_gap_pp.median():.1f}")
