"""Sensibilidad a la semilla (secc. 8): mejor modelo propio (MLP) y mejor modelo de libreria (iTransformer) con 3 semillas."""
import sys; sys.path.insert(0, "src")
from lab9 import *
what = sys.argv[1:] or ["mlp", "itrans"]
res = load_json("seeds.json")
if "mlp" in what:
    df = load_panel(100); P = Panel(df); scale = mase_scale(df.iloc[:P.pos_train_end]); cfg = load_json("runs.json")["mlp_L168_win"]["cfg"]
    r = {"42": load_json("final_own.json")["MLP"]["MASE"]}
    for s in (43, 44):
        net, info = fit(P, {**cfg, "seed": s}, verbose=False)
        bt = backtest(make_predict_fn(net, P, cfg["L"]), df); r[str(s)] = float(metrics(bt, scale)["MASE"])
        print("mlp seed", s, r[str(s)], flush=True)
    res["MLP"] = dict(MASE_by_seed=r, mean=float(np.mean(list(r.values()))), std=float(np.std(list(r.values()), ddof=1)))
    save_json(res, "seeds.json")
if "itrans" in what:
    from run_nf import *
    r = {"1": load_json("final_nf.json")["iTransformer"]["MASE"]}
    for s in (2, 3):
        nf = NeuralForecast(models=[mk("itransformer", 168, random_seed=s)], freq="h")
        nf.fit(df_for(None, VAL_END), val_size=VAL_SIZE)
        out = nf.cross_validation(df_for(None, END), n_windows=None, h=24, test_size=92 * 24, step_size=24, val_size=VAL_SIZE, refit=False)
        bt = backtest(cache_predict_fn(long_to_cache(out, COLS, {"y_hat": "iTransformer"})), df)
        r[str(s)] = float(metrics(bt, scale)["MASE"]); print("itrans seed", s, r[str(s)], flush=True)
    res["iTransformer"] = dict(MASE_by_seed=r, mean=float(np.mean(list(r.values()))), std=float(np.std(list(r.values()), ddof=1)))
    save_json(res, "seeds.json")
for k, v in res.items(): print(k, v)
