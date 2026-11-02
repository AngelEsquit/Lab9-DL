"""Baselines (StatsForecast: Naive, SeasonalNaive 24/168, MSTL) y LightGBM global recursivo. Evalua en prueba con backtest()."""
import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
from lab9 import *
import lightgbm as lgb
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, MSTL, AutoETS

df = load_panel(100); P = Panel(df); COLS = list(df.columns)
scale = mase_scale(df.iloc[:P.pos_train_end])
Y = df.reset_index(names="ds").melt(id_vars="ds", var_name="unique_id", value_name="y").sort_values(["unique_id", "ds"])


def run_stats():
    models = [Naive(), SeasonalNaive(24, alias="SNaive24"), SeasonalNaive(168, alias="SNaive168"),
              MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL")]
    sf = StatsForecast(models=models, freq="h", n_jobs=-1)
    t0 = time.time()
    # input_size acota la historia usada en cada ventana (8 semanas) para que MSTL sea tratable; refit=True en cada origen
    out = sf.cross_validation(df=Y, h=24, n_windows=92, step_size=24, refit=True, input_size=24 * 7 * 8)
    tt = time.time() - t0
    res = {}
    for name in ("Naive", "SNaive24", "SNaive168", "MSTL"):
        cache = long_to_cache(out, COLS, {"y_hat": name})
        bt = backtest(cache_predict_fn(cache), df)
        rep = full_report(bt, scale, name)
        rep.update(family="Baseline", L={"Naive": 1, "SNaive24": 24, "SNaive168": 168, "MSTL": 1344}[name], strategy="directa",
                   covariates=False, params_total=0, params_trainable=0, train_time=0.0, infer_time=tt if name == "MSTL" else 0.0,
                   hardware="CPU")
        res[name] = rep; save_preds(bt, name)
        print(name, {k: round(v, 4) for k, v in rep.items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas")}, flush=True)
    print("tiempo total cross_validation StatsForecast:", tt)
    return res


# ---------------------------------------------------------------- LightGBM global
LAGS = list(range(1, 25)) + [168]
cal_raw = np.column_stack([P.index.hour, P.index.dayofweek, P.index.month, P.cal_df["holiday"].values]).astype("float32")
Zn = ((df.values - P.mu_np) / P.sd_np).astype("float32")


def feats_matrix(t_rows, n_rows):
    """Filas (t,n): rezagos z[t-l, n] (l en LAGS) + hora, dia de la semana, mes, feriado de t."""
    return np.column_stack([Zn[t_rows - l, n_rows] for l in LAGS] + [cal_raw[t_rows]])


def train_lgb(months=None, train_end_pos=None, params=None, fixed_rounds=None):
    te = train_end_pos or P.pos_train_end
    lo = 168 if months is None else max(168, te - int(months * 30.4375 * 24))
    t = np.arange(lo, te)
    tt, nn_ = np.repeat(t, P.N), np.tile(np.arange(P.N), len(t))
    cat = [len(LAGS), len(LAGS) + 1, len(LAGS) + 2]
    dtr = lgb.Dataset(feats_matrix(tt, nn_), Zn[tt, nn_], categorical_feature=cat, free_raw_data=False)
    p = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, num_threads=os.cpu_count())
    p.update(params or {})
    t0 = time.time()
    if fixed_rounds is None:
        tv = np.arange(P.pos_train_end, P.pos_val_end)
        vv, vn = np.repeat(tv, P.N), np.tile(np.arange(P.N), len(tv))
        dva = lgb.Dataset(feats_matrix(vv, vn), Zn[vv, vn], reference=dtr)
        b = lgb.train(p, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        b = lgb.train(p, dtr, fixed_rounds)
    return b, time.time() - t0


def lgb_predict_fn(booster):
    mu, sd = P.mu_np, P.sd_np
    it = booster.best_iteration or None
    def predict(hist, origin):
        t = len(hist); assert P.index[t] == origin
        buf = np.zeros((P.N, 168 + 24), dtype="float32")
        buf[:, :168] = ((hist.values[-168:] - mu) / sd).T
        for k in range(24):
            pos = 168 + k
            lagmat = np.column_stack([buf[:, pos - l] for l in LAGS])
            X = np.column_stack([lagmat, np.tile(cal_raw[t + k], (P.N, 1))])
            buf[:, pos] = booster.predict(X, num_iteration=it)
        return np.clip(buf[:, 168:] * sd[:, None] + mu[:, None], 0, None)
    return predict


def val_mase_lgb(booster):
    vo = pd.date_range("2014-07-01", "2014-09-30", freq="1D")
    return metrics(backtest(lgb_predict_fn(booster), df, vo), scale)["MASE"]


GRID = {"lgb_base": dict(), "lgb_leaves15": dict(num_leaves=15), "lgb_lr0.1_leaves127": dict(learning_rate=0.1, num_leaves=127)}


def run_lgb():
    runs = load_json("runs_lgb.json")
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    for rid, prm in GRID.items():
        if rid in runs: continue
        b, tt = train_lgb(None, params=prm)
        runs[rid] = dict(params=prm, val_mase=val_mase_lgb(b), best_iter=b.best_iteration, train_time=tt)
        print(rid, runs[rid], flush=True); b.save_model(os.path.join(ROOT, "checkpoints", rid + ".txt"))
        save_json(runs, "runs_lgb.json")
    for mo in (3, 12, 30):
        rid = f"lgb_m{mo}"
        if rid in runs: continue
        b, tt = train_lgb(mo)
        runs[rid] = dict(months=mo, val_mase=val_mase_lgb(b), best_iter=b.best_iteration, train_time=tt)
        print(rid, runs[rid], flush=True); save_json(runs, "runs_lgb.json")
    best = min(GRID, key=lambda r: runs[r]["val_mase"])
    b = lgb.Booster(model_file=os.path.join(ROOT, "checkpoints", best + ".txt"))
    b.best_iteration = runs[best]["best_iter"]
    bt = backtest(lgb_predict_fn(b), df)
    rep = full_report(bt, scale, "LightGBM")
    rep.update(family="Arboles", L=168, strategy="recursiva", covariates=True, params_total=b.num_trees(), params_trainable=b.num_trees(),
               train_time=runs[best]["train_time"], infer_time=bt.attrs["infer_time"], hardware="CPU", best_cfg=best)
    save_preds(bt, "LightGBM")
    print("LightGBM", {k: round(v, 4) for k, v in rep.items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas")}, flush=True)
    return rep


if __name__ == "__main__":
    what = sys.argv[1:] or ["stats", "lgb"]
    res = load_json("final_baselines.json")
    if "stats" in what: res.update(run_stats())
    if "lgb" in what: res["LightGBM"] = run_lgb()
    save_json(res, "final_baselines.json")
