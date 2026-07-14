"""
export.py — собирает все посчитанные величины в dashboard/data.js (window.DASH=...).
Никаких новых оценок «на глаз»: H1/H4 берутся из models_h1_h4.json (BRR+PV),
остальное — взвешенные агрегаты для графиков.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT/"data"/"interim"/"pisa_full.parquet")
W = "W_FSTUWT"
PANEL = ["ESP","KOR","USA","COL","KAZ"]
NAMES = {"ESP":"Испания","KOR":"Корея","USA":"США","COL":"Колумбия","KAZ":"Казахстан","POOL":"Все страны"}
SPECIAL = {"9701","9702","9703","9704","9705","9997","9998","9999","0000"}

_, meta = pyreadstat.read_sav("data/raw/CY08MSP_STU_QQQ.SAV", metadataonly=True)
OCLAB = {str(k): v for k, v in meta.variable_value_labels.get("OCOD3", {}).items()}

RU_OCC = {
    "011":"Офицеры вооружённых сил","031":"Военнослужащие (рядовой состав)",
    "1120":"Директора и топ-менеджеры","1223":"Руководители R&D",
    "2131":"Биологи","214":"Инженеры","2144":"Инженеры-механики",
    "2161":"Архитекторы","2163":"Дизайнеры одежды","2166":"Графические дизайнеры",
    "221":"Врачи","2211":"Врачи-терапевты","2212":"Врачи-специалисты",
    "2221":"Медсёстры","2250":"Ветеринары","2261":"Стоматологи","2262":"Фармацевты",
    "2330":"Учителя средней школы","234":"Учителя начальной школы","2342":"Воспитатели",
    "2421":"Бизнес-аналитики","2422":"Специалисты по госуправлению",
    "2511":"Системные аналитики","2512":"Разработчики ПО","2514":"Программисты",
    "2611":"Юристы","2634":"Психологи","2642":"Журналисты","2643":"Переводчики",
    "2652":"Музыканты","3334":"Риелторы","3359":"Госслужащие (средний уровень)",
    "3421":"Спортсмены","3422":"Спортивные тренеры","3432":"Дизайнеры интерьера",
    "4110":"Офисные служащие","5111":"Бортпроводники","5120":"Повара","5141":"Парикмахеры",
    "5142":"Косметологи","5411":"Пожарные","5412":"Полицейские","5419":"Работники охраны",
    "7212":"Сварщики","7231":"Автомеханики",
}

oc = df["OCOD3"].astype(str)
df["valid_occ"] = ~oc.isin(SPECIAL) & oc.str.match(r"^\d{3,4}$")
df["is_stem"] = (df["valid_occ"] & oc.str[:2].isin(["21","25"])).astype(float)
df["is_health"] = (df["valid_occ"] & oc.str[:2].eq("22")).astype(float)
df["prof"] = (df["valid_occ"] & oc.str[:1].isin(["1","2","3"])).astype(float)
df["female"] = df["ST004D01T"].map({1:1.0,2:0.0})
df["math"] = df[[f"PV{i}MATH" for i in range(1,11)]].mean(axis=1)


def wmean(s, w):
    m = np.isfinite(s) & np.isfinite(w); s, w = s[m], w[m]
    return float(np.average(s, weights=w)) if len(s) else None

def wsd(s, w):
    m = np.isfinite(s) & np.isfinite(w); s, w = s[m], w[m]
    mu = np.average(s, weights=w); return float(np.sqrt(np.average((s-mu)**2, weights=w)))

def sub(cnt):
    return df if cnt == "POOL" else df[df["CNT"] == cnt]


def per_country(cnt):
    d = sub(cnt)
    v = d[d.valid_occ]
    o = {}
    # HISEI → BSMJ: децили статуса родителей, взвеш. среднее ожидание
    dd = d.dropna(subset=["HISEI","BSMJ",W])
    q = pd.qcut(dd["HISEI"], 10, duplicates="drop")
    binned = []
    for _, g in dd.groupby(q, observed=True):
        binned.append({"x": round(wmean(g["HISEI"].values, g[W].values),1),
                       "y": round(wmean(g["BSMJ"].values, g[W].values),1)})
    o["hisei_bsmj"] = binned
    # STEM по квартилю математики × пол
    dm = v.dropna(subset=["math","female",W]).copy()
    dm["mq"] = pd.qcut(dm["math"], 4, labels=[1,2,3,4], duplicates="drop")
    stem_q = {"female":[], "male":[]}
    for qi in [1,2,3,4]:
        for g,key in [(1,"female"),(0,"male")]:
            s = dm[(dm.mq==qi)&(dm.female==g)]
            stem_q[key].append(round(wmean(s.is_stem.values, s[W].values)*100,1) if len(s) else None)
    o["stem_by_mathq"] = stem_q
    # доли по полу
    o["by_gender"] = {}
    for g,key in [(1,"female"),(0,"male")]:
        s = v[v.female==g]
        o["by_gender"][key] = {
            "stem": round(wmean(s.is_stem.values, s[W].values)*100,1),
            "health": round(wmean(s.is_health.values, s[W].values)*100,1),
            "prof": round(wmean(s.prof.values, s[W].values)*100,1),
            "bsmj": round(wmean(s.BSMJ.values, s[W].values),1),
        }
    # топ-10 профессий по полу
    o["top_occ"] = {}
    for g,key in [(1,"female"),(0,"male")]:
        s = v[v.female==g]
        tot = s[W].sum()
        top = s.groupby(oc[s.index])[W].sum().sort_values(ascending=False).head(10)
        o["top_occ"][key] = [{"code":c, "label":RU_OCC.get(c, OCLAB.get(c,"?")),
                              "pct":round(wt/tot*100,1)} for c,wt in top.items()]
    # сводка + «потолок»
    b = d.dropna(subset=["BSMJ",W])
    o["summary"] = {
        "name": NAMES[cnt], "cnt": cnt,
        "n_bsmj": int(len(b)),
        "mean_bsmj": round(wmean(b.BSMJ.values, b[W].values),1),
        "sd_bsmj": round(wsd(b.BSMJ.values, b[W].values),1),
        "ceiling": round(wmean((b.BSMJ>=80).astype(float).values, b[W].values)*100,1),
    }
    return o


def main():
    data = {"names": NAMES, "panel": PANEL,
            "countries": {c: per_country(c) for c in PANEL + ["POOL"]}}

    # H1/H4 из моделей (BRR+PV)
    models = json.loads((ROOT/"data"/"interim"/"models_h1_h4.json").read_text())
    data["models"] = models

    # гетерогенность по всем странам: r(HISEI,BSMJ)
    def wcorr(x,y,w):
        m=np.isfinite(x)&np.isfinite(y)&np.isfinite(w); x,y,w=x[m],y[m],w[m]
        mx,my=np.average(x,weights=w),np.average(y,weights=w)
        d=np.sqrt(np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w))
        return float(np.average((x-mx)*(y-my),weights=w)/d) if d else None
    het=[]
    for c,s in df.groupby("CNT"):
        dd=s.dropna(subset=["HISEI","BSMJ",W])
        if len(dd)<800 or c=="QAZ": continue
        het.append({"cnt":c,"r":round(wcorr(dd.HISEI.values,dd.BSMJ.values,dd[W].values),3),"n":int(len(dd))})
    data["heterogeneity"]=sorted(het,key=lambda z:z["r"])

    # факторный ландшафт (чистая OLS, стандартиз.) — воспроизводим кратко
    import statsmodels.api as sm
    def z(s): sd=s.std(); return (s-s.mean())/sd if sd else s*0
    rep=df[["ST127Q01TA","ST127Q02TA","ST127Q03TA"]]
    df["repeated"]=(((rep>1)&(rep<90)).any(axis=1)).astype(float); df.loc[rep.isna().all(axis=1),"repeated"]=np.nan
    clean=["HISEI","math","MATHEFF","ANXMAT","GROSAGR","CURIOAGR","CREATEFF","BELONG","FAMSUP","TEACHSUP","IMMIG","repeated","female"]
    dc=df.dropna(subset=clean+["BSMJ",W]).copy()
    X=sm.add_constant(dc[clean].apply(z))
    m=sm.WLS(dc["BSMJ"],X,weights=dc[W]).fit()
    LBL={"HISEI":"Статус родителей","math":"Успеваемость (матем.)","MATHEFF":"Мат. самоэффективность",
         "ANXMAT":"Мат. тревожность","GROSAGR":"Growth mindset","CURIOAGR":"Любознательность",
         "CREATEFF":"Креативная самоэфф.","BELONG":"Чувство принадлежности","FAMSUP":"Поддержка семьи",
         "TEACHSUP":"Поддержка учителя","IMMIG":"Мигрантский фон","repeated":"Второгодничество","female":"Пол (девочки)"}
    fac=[{"label":LBL[k],"beta":round(m.params[k],2)} for k in clean]
    data["factors"]=sorted(fac,key=lambda z:abs(z["beta"]),reverse=True)
    data["factors_r2"]=round(m.rsquared,3)

    # литература
    data["meta"]=json.loads((ROOT/"data"/"interim"/"meta_findings.json").read_text())

    out=ROOT/"dashboard"/"data.js"
    out.write_text("window.DASH = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"[saved] {out}  ({out.stat().st_size/1024:.0f} КБ)")


if __name__ == "__main__":
    main()
