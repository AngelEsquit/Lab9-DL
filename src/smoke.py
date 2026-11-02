import sys; sys.path.insert(0, "src")
from lab9 import *
df = load_panel(100); P = Panel(df)
for L in (168,):
    tr = WindowDataset(P, P.origins("train", L), L); va = WindowDataset(P, P.origins("val", L), L); te = WindowDataset(P, P.origins("test", L), L)
    print("ventanas", len(tr), len(va), len(te))
x, xf, y, n = tr[5]; print(x.shape, xf.shape, y.shape)
for m in ["dlinear", "nlinear", "mlp", "lstm", "lstm_auto", "seq2seq"]:
    t0 = time.time()
    net, info = fit(P, dict(model=m, epochs=1, spe=100), verbose=False)
    print(m, info["history"][0]["val_mase"], info["params_total"], round(time.time()-t0, 1), "s")
    bt = backtest(make_predict_fn(net, P, 168), df, TEST_ORIGINS[:3])
    print("   bt", bt.shape, metrics(bt, mase_scale(df.iloc[:P.pos_train_end]))["MASE"], bt.attrs["infer_time"])
