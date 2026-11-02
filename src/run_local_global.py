"""5.3 Local contra global: el mejor modelo propio (MLP, mlp_L168_win) entrenado por cliente (10 clientes, semilla fija) vs. global."""
import sys; sys.path.insert(0, "src")
from lab9 import *

df = load_panel(100); COLS = list(df.columns)
cfg = load_json("runs.json")["mlp_L168_win"]["cfg"]
rng = np.random.RandomState(SEED)
CL = sorted(rng.choice(COLS, 10, replace=False).tolist())
scale = mase_scale(df.iloc[:Panel(df).pos_train_end])
glob = load_preds("MLP"); glob = glob[glob.unique_id.isin(CL)]
rows, parts, tt = [], [], 0.
for c in CL:
    dfc = df[[c]]; Pc = Panel(dfc)
    net, info = fit(Pc, cfg, verbose=False)
    tt += info["train_time"]
    bt = backtest(make_predict_fn(net, Pc, cfg["L"]), dfc)
    parts.append(bt)
    print(c, "local val_mase %.4f epochs %d t=%.0fs" % (info["best_val_mase"], info["best_epoch"], info["train_time"]), flush=True)
loc = pd.concat(parts, ignore_index=True)
# el panel global se evalua con backtest(); recuperamos y real desde el parquet del global
g = load_preds("MLP"); g = g[g.unique_id.isin(CL)].merge(loc[["unique_id", "ds", "origin", "y"]], on=["unique_id", "ds", "origin"])
mg, ml = metrics(g, scale), metrics(loc, scale)
pc = []
for c in CL:
    a = g[g.unique_id == c]; b = loc[loc.unique_id == c]
    sg, sl = (a.y_hat - a.y).abs().mean() / scale[c], (b.y_hat - b.y).abs().mean() / scale[c]
    pc.append(dict(unique_id=c, mean_kwh=float(df.iloc[:Panel(df).pos_train_end][c].mean()), MASE_global=float(sg), MASE_local=float(sl)))
res = dict(clients=CL, MASE_global=float(mg["MASE"]), MASE_local=float(ml["MASE"]), MAE_global=float(mg["MAE"]), MAE_local=float(ml["MAE"]),
           train_time_local_total=tt, n_models_local=10, per_client=pc)
save_json(res, "local_global.json")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "per_client"})
for r in pc: print(r)
