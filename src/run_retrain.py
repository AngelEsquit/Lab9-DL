"""Reentrenamiento mensual (seccion 4): al inicio de cada mes de prueba (1-oct, 1-nov, 1-dic) se reentrena con TODA la historia
disponible hasta las 23:00 del dia anterior (ventana expandible) y se pronostican los origenes diarios de ese mes.
Uso: python src/run_retrain.py mlp itrans mstl"""
import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
from lab9 import *

MONTHS = [("2014-10-01", "2014-10-31"), ("2014-11-01", "2014-11-30"), ("2014-12-01", "2014-12-31")]
df = load_panel(100); P = Panel(df); COLS = list(df.columns)
scale = mase_scale(df.iloc[:P.pos_train_end])


def finish(parts, name, ref, extra):
    bt = pd.concat(parts, ignore_index=True)
    assert bt.origin.nunique() == 92
    rep = full_report(bt, scale, name + " (mensual)")
    fixed = load_json(ref[0])[ref[1]]
    rep.update(extra, MASE_fixed=fixed["MASE"], MAE_fixed=fixed["MAE"], RMSE_fixed=fixed["RMSE"], WAPE_fixed=fixed["WAPE"])
    save_preds(bt, name + "_monthly")
    res = load_json("retrain.json"); res[name] = rep; save_json(res, "retrain.json")
    print(name, "mensual: MAE %.3f MASE %.4f | pesos fijos: MAE %.3f MASE %.4f" % (rep["MAE"], rep["MASE"], fixed["MAE"], fixed["MASE"]), flush=True)


def run_mlp():
    cfg = load_json("runs.json")["mlp_L168_win"]["cfg"]
    parts, tt = [], 0.
    for a, b in MONTHS:
        te = P.index.get_loc(pd.Timestamp(a))                       # exclusivo: ultima hora de entrenamiento = a-1h
        net, info = fit(P, {**cfg, "months": None}, train_end_pos=te, fixed_epochs=cfg["epochs"], verbose=False)
        tt += info["train_time"]
        orig = pd.date_range(a, b, freq="1D")
        parts.append(backtest(make_predict_fn(net, P, cfg["L"]), df, orig))
        print("mlp", a, "train_time %.0fs" % info["train_time"], "ventanas", info["n_train_windows"], flush=True)
    finish(parts, "MLP", ("final_own.json", "MLP"), dict(train_time_total=tt, n_retrains=3))


def run_itrans():
    from run_nf import mk, Y, NeuralForecast, VAL_SIZE
    parts, tt = [], 0.
    for a, b in MONTHS:
        ts = pd.Timestamp(a)
        nf = NeuralForecast(models=[mk("itransformer", 168)], freq="h")
        t0 = time.time(); nf.fit(Y[Y.ds < ts], val_size=VAL_SIZE); tt += time.time() - t0
        nd = len(pd.date_range(a, b, freq="1D"))
        out = nf.cross_validation(Y[Y.ds <= pd.Timestamp(b) + pd.Timedelta(hours=23)], n_windows=None, h=24, test_size=nd * 24,
                                  step_size=24, val_size=VAL_SIZE, refit=False)
        cache = long_to_cache(out, COLS, {"y_hat": "iTransformer"})
        parts.append(backtest(cache_predict_fn(cache), df, pd.date_range(a, b, freq="1D")))
        print("itransformer", a, "train_time %.0fs" % (time.time() - t0), flush=True)
    finish(parts, "iTransformer", ("final_nf.json", "iTransformer"), dict(train_time_total=tt, n_retrains=3))


def run_mstl():
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoETS
    Y = df.reset_index(names="ds").melt(id_vars="ds", var_name="unique_id", value_name="y").sort_values(["unique_id", "ds"])
    parts, tt = [], 0.
    for a, b in MONTHS:
        nd = len(pd.date_range(a, b, freq="1D"))
        sf = StatsForecast(models=[MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL")], freq="h", n_jobs=4)
        t0 = time.time()
        out = sf.cross_validation(df=Y[Y.ds <= pd.Timestamp(b) + pd.Timedelta(hours=23)], h=24, n_windows=nd, step_size=24,
                                  refit=False, input_size=24 * 7 * 8)
        tt += time.time() - t0
        cache = long_to_cache(out, COLS, {"y_hat": "MSTL"})
        parts.append(backtest(cache_predict_fn(cache), df, pd.date_range(a, b, freq="1D")))
        print("mstl", a, "%.0fs" % (time.time() - t0), flush=True)
    finish(parts, "MSTL", ("final_baselines.json", "MSTL"), dict(train_time_total=tt, n_retrains=3))


if __name__ == "__main__":
    for w in sys.argv[1:]: {"mlp": run_mlp, "itrans": run_itrans, "mstl": run_mstl}[w]()
