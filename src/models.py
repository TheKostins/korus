"""
models.py — формальные тесты H1 и H4 с дизайном PISA (веса + BRR + пулинг PV).
    .venv/bin/python src/models.py            # панель + пул
    .venv/bin/python src/models.py ESP        # одна страна (отладка)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from pisa_stats import combine, zcols

ROOT = Path(__file__).resolve().parent.parent
PANEL = ["ESP", "KOR", "USA", "COL", "KAZ"]
SPECIAL = {"9701","9702","9703","9704","9705","9997","9998","9999","0000"}


def prep(df):
    oc = df["OCOD3"].astype(str)
    valid = ~oc.isin(SPECIAL) & oc.str.match(r"^\d{3,4}$")
    df = df.assign(
        female=df["ST004D01T"].map({1: 1.0, 2: 0.0}),
        is_stem=(valid & oc.str[:2].isin(["21", "25"])).astype(float),
        prof=(valid & oc.str[:1].isin(["1", "2", "3"])).astype(float),
        valid_occ=valid,
    )
    return df


def subset(df, cnt):
    d = df if cnt == "POOL" else df[df["CNT"] == cnt]
    return d


# ---- построители дизайна (const + предикторы) ----
def _mat(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])

def X_h1_total(d, pv):   return (["const","HISEI_z","female"], _mat(d, ["HISEI_z","female"]))
def X_h1_direct(d, pv):  return (["const","HISEI_z","female","math_z"],
                                 _mat(d.assign(math_z=d[f"MATHZ{pv}"]), ["HISEI_z","female","math_z"]))
def X_lvl(d, pv):        return (["const","female"], _mat(d, ["female"]))
def X_field(d, pv):      return (["const","female","math_z"],
                                 _mat(d.assign(math_z=d[f"MATHZ{pv}"]), ["female","math_z"]))
def X_field_ix(d, pv):
    dd = d.assign(math_z=d[f"MATHZ{pv}"]); dd = dd.assign(fxm=dd["female"]*dd["math_z"])
    return (["const","female","math_z","fem×math"], _mat(dd, ["female","math_z","fxm"]))


def run_country(df, cnt):
    d0 = subset(prep(df), cnt)
    out = {"CNT": cnt}

    # ===== H1: BSMJ ~ HISEI (+ math) =====
    h1 = d0.dropna(subset=["BSMJ","HISEI","female","W_FSTUWT"]).copy()
    h1 = zcols(h1, ["HISEI"] + [f"PV{i}MATH" for i in range(1,11)])
    h1 = h1.rename(columns={f"PV{i}MATH_z": f"MATHZ{i}" for i in range(1,11)})
    _, bt, st, _ = combine(h1, "BSMJ", X_h1_total, "ols", use_pv=False)
    _, bd, sd, _ = combine(h1, "BSMJ", X_h1_direct, "ols", use_pv=True)
    b_tot, b_dir = bt[1], bd[1]                       # коэф. HISEI_z
    out.update(H1_n=len(h1), H1_total=b_tot, H1_total_se=st[1],
               H1_direct=b_dir, H1_direct_se=sd[1],
               H1_pct_mediated=(b_tot-b_dir)/b_tot*100)

    # ===== H4 уровень: девочки НЕ ниже =====
    lv = d0.dropna(subset=["BSMJ","female","W_FSTUWT"]).copy()      # BSMJ⇔valid occ
    _, bl, sl, tl = combine(lv, "BSMJ", X_lvl, "ols", use_pv=False)
    _, bp, sp, tp = combine(lv, "prof", X_lvl, "logit", use_pv=False)
    out.update(H4_lvl_n=len(lv), H4_BSMJ_female=bl[1], H4_BSMJ_female_se=sl[1],
               H4_prof_female_logOR=bp[1], H4_prof_female_se=sp[1])

    # ===== H4 поле: девочки реже STEM при равном балле =====
    fd = d0[d0["valid_occ"]].dropna(subset=["female","W_FSTUWT"]).copy()
    fd = zcols(fd, [f"PV{i}MATH" for i in range(1,11)])
    fd = fd.rename(columns={f"PV{i}MATH_z": f"MATHZ{i}" for i in range(1,11)})
    _, bf, sf, tf = combine(fd, "is_stem", X_field, "logit", use_pv=True)
    _, bi, si, ti = combine(fd, "is_stem", X_field_ix, "logit", use_pv=True)
    out.update(H4_field_n=len(fd),
               H4_stem_female_logOR=bf[1], H4_stem_female_se=sf[1],
               H4_stem_math_logOR=bf[2],
               H4_stem_femXmath_logOR=bi[3], H4_stem_femXmath_se=si[3])
    return out


def main():
    df = pd.read_parquet(ROOT/"data"/"interim"/"pisa_full.parquet")
    targets = [a.upper() for a in sys.argv[1:]] or PANEL + ["POOL"]
    rows = [run_country(df, c) for c in targets]
    res = pd.DataFrame(rows).set_index("CNT")

    def show(cols, title, fmt):
        print(f"\n{'='*72}\n{title}\n{'='*72}")
        print(res[cols].to_string(float_format=fmt))

    pd.set_option("display.width", 200)
    show(["H1_n","H1_total","H1_total_se","H1_direct","H1_direct_se","H1_pct_mediated"],
         "H1 — прямое воспроизводство статуса (β HISEI, стандартиз.; BSMJ-пункты за 1σ)",
         lambda x: f"{x:.2f}")
    show(["H4_lvl_n","H4_BSMJ_female","H4_BSMJ_female_se","H4_prof_female_logOR","H4_prof_female_se"],
         "H4 УРОВЕНЬ — девочки НЕ ниже (BSMJ: пункты; prof: log-OR; + = девочки выше)",
         lambda x: f"{x:.2f}")
    show(["H4_field_n","H4_stem_female_logOR","H4_stem_female_se","H4_stem_math_logOR","H4_stem_femXmath_logOR","H4_stem_femXmath_se"],
         "H4 ПОЛЕ — девочки реже STEM при равном балле (log-OR; female − = меньше STEM)",
         lambda x: f"{x:.2f}")
    res.to_json(ROOT/"data"/"interim"/"models_h1_h4.json", orient="index", indent=2)
    print(f"\n[saved] data/interim/models_h1_h4.json")


if __name__ == "__main__":
    main()
