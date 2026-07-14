"""
pisa_stats.py — корректная оценка по дизайну PISA:
  • взвешивание финальным весом W_FSTUWT;
  • стандартные ошибки через BRR (80 реплик-весов, метод Фэя, k=0.5 → знаменатель 80·(1-0.5)²=20);
  • объединение 10 plausible values по правилам Рубина.
Быстрые взвешенные OLS и логит на numpy (замкнутая форма / IRLS), чтобы прогонять
до 10×81 = 810 подгонок на модель.
"""
import numpy as np

FAY = 0.5
BRR_DENOM = 80 * (1 - FAY) ** 2   # = 20.0
N_REP = 80
N_PV = 10


def wls_beta(y, X, w):
    XtW = X.T * w
    return np.linalg.solve(XtW @ X, XtW @ y)


def logit_beta(y, X, w, iters=30, tol=1e-8):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        v = np.clip(p * (1 - p), 1e-9, None)
        Wv = w * v
        z = eta + (y - p) / v                     # рабочий отклик IRLS
        XtW = X.T * Wv
        b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    return b


def combine(d, outcome, build_X, kind="ols", use_pv=False):
    """
    d        : DataFrame с колонками W_FSTUWT, W_FSTURWT1..80 и предикторами
    outcome  : имя целевой колонки
    build_X  : fn(d, pv) -> (names:list, X:ndarray)  (pv=None если PV не нужен)
    kind     : 'ols' | 'logit'
    use_pv   : брать ли 10 PV (пулинг Рубина) — иначе один прогон
    Возврат  : (names, beta, se, tval)
    """
    est = wls_beta if kind == "ols" else logit_beta
    y = d[outcome].to_numpy(float)
    w0 = d["W_FSTUWT"].to_numpy(float)
    repw = [d[f"W_FSTURWT{r}"].to_numpy(float) for r in range(1, N_REP + 1)]
    pvs = range(1, N_PV + 1) if use_pv else [None]

    thetas, Us = [], []
    for pv in pvs:
        names, X = build_X(d, pv)
        theta = est(y, X, w0)
        reps = np.array([est(y, X, wr) for wr in repw])
        U = ((reps - theta) ** 2).sum(axis=0) / BRR_DENOM   # выборочная дисперсия (BRR)
        thetas.append(theta); Us.append(U)

    thetas = np.array(thetas); Us = np.array(Us)
    theta_bar = thetas.mean(axis=0)
    U_bar = Us.mean(axis=0)
    if len(thetas) > 1:                                     # правила Рубина
        B = thetas.var(axis=0, ddof=1)
        T = U_bar + (1 + 1 / len(thetas)) * B
    else:
        T = U_bar
    se = np.sqrt(T)
    return names, theta_bar, se, theta_bar / se


def zcols(d, cols):
    """стандартизовать перечисленные колонки по (невзвеш.) выборке d, вернуть копию d."""
    d = d.copy()
    for c in cols:
        s = d[c].astype(float)
        sd = s.std()
        d[c + "_z"] = (s - s.mean()) / (sd if sd else 1.0)
    return d
