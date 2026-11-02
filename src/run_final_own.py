"""Evaluacion final en prueba (una vez por modelo) de la mejor configuracion de cada familia propia, seleccionada por
MASE de validacion. Uso: python src/run_final_own.py test"""
import sys; sys.path.insert(0, "src")
from lab9 import *

df = load_panel(100); P = Panel(df)
scale = mase_scale(df.iloc[:P.pos_train_end])
NAMES = {"mlp": "MLP", "dlinear": "DLinear", "nlinear": "NLinear", "lstm": "LSTM (multi-salida)",
         "lstm_auto": "LSTM (autorregresivo)", "seq2seq": "Seq2seq LSTM"}
STRAT = {"mlp": "multi-salida", "dlinear": "multi-salida", "nlinear": "multi-salida", "lstm": "multi-salida",
         "lstm_auto": "autorregresiva", "seq2seq": "encoder-decoder"}
COV = {"mlp": True, "dlinear": False, "nlinear": False, "lstm": True, "lstm_auto": True, "seq2seq": True}


def build(cfg, ckpt):
    cfg = {**DEFAULT, **cfg}
    kw = {k: cfg[k] for k in ("hidden", "layers", "dropout")}
    if cfg["model"] in ("dlinear", "nlinear"): kw = {}
    net = Net(CORES[cfg["model"]](cfg["L"], P.C.shape[1], **kw), cfg["norm"]).to(P.dev)
    net.load_state_dict(torch.load(ckpt, map_location=P.dev)); net.eval()
    return net


def best_by_family(runs):
    best = {}
    for rid, r in runs.items():
        m = r["cfg"]["model"]
        if m not in best or r["best_val_mase"] < runs[best[m]]["best_val_mase"]: best[m] = rid
    return best


def evaluate(net, cfg, info, name, fam):
    if P.dev == "cuda": torch.cuda.reset_peak_memory_stats()
    bt = backtest(make_predict_fn(net, P, cfg["L"]), df)
    rep = full_report(bt, scale, name)
    m = cfg["model"]
    rep.update(family=fam, L=cfg["L"], strategy=STRAT[m], covariates=COV[m], params_total=info["params_total"],
               params_trainable=info["params_trainable"], train_time=info["train_time"], infer_time=bt.attrs["infer_time"],
               hardware=f"GPU {torch.cuda.get_device_name(0)}", gpu_peak_mb=info["gpu_peak_mb"], best_cfg=info["rid"],
               val_mase=info["best_val_mase"], months=cfg["months"] or 30)
    save_preds(bt, name)
    return rep


def run_test():
    runs = load_json("runs.json")
    fin = load_json("final_own.json")
    for m, rid in best_by_family(runs).items():
        name = NAMES[m]
        if name in fin: continue
        info = {**runs[rid], "rid": rid}
        net = build(runs[rid]["cfg"], os.path.join(ROOT, "checkpoints", rid + ".pt"))
        fam = {"mlp": "MLP y lineales", "dlinear": "MLP y lineales", "nlinear": "MLP y lineales", "lstm": "Recurrentes",
               "lstm_auto": "Recurrentes", "seq2seq": "Encoder-decoder"}[m]
        fin[name] = evaluate(net, runs[rid]["cfg"], info, name, fam)
        print(name, rid, {k: round(v, 4) for k, v in fin[name].items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas")}, flush=True)
        save_json(fin, "final_own.json")


if __name__ == "__main__":
    run_test()
