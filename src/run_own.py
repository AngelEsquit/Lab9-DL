"""Busqueda de hiperparametros (solo validacion) para MLP, DLinear, NLinear, LSTM, LSTM autoregresivo y seq2seq.
Resultados incrementales en results/runs.json; checkpoints en checkpoints/. Uso: python src/run_own.py [prefijo]"""
import sys; sys.path.insert(0, "src")
from lab9 import *
os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)

base_l = dict(model="lstm", L=168, hidden=64, layers=1, dropout=0.1)
R = {}
def add(i, **kw): R[i] = kw
# DLinear / NLinear
for L in (48, 168, 336): add(f"dlinear_L{L}", model="dlinear", L=L)
add("dlinear_L168_win", model="dlinear", L=168, norm="window")
add("dlinear_L336_win", model="dlinear", L=336, norm="window")
add("dlinear_lr3e-3", model="dlinear", L=336, lr=3e-3)
for mo in (3, 12, 30): add(f"dlinear_m{mo}", model="dlinear", L=168, months=mo)
for L in (168, 336): add(f"nlinear_L{L}", model="nlinear", L=L)
add("nlinear_L168_win", model="nlinear", L=168, norm="window")
# MLP
add("mlp_L168", model="mlp", L=168, hidden=256, layers=2)
add("mlp_L168_h512_l3", model="mlp", L=168, hidden=512, layers=3, dropout=0.2)
add("mlp_L336", model="mlp", L=336, hidden=256, layers=2)
add("mlp_L168_win", model="mlp", L=168, hidden=256, layers=2, norm="window")
# LSTM multi-salida
for L in (48, 168, 336): add(f"lstm_L{L}", **{**base_l, "L": L})
for h in (32, 128): add(f"lstm_h{h}", **{**base_l, "hidden": h})
add("lstm_2layers", **{**base_l, "layers": 2, "dropout": 0.2})
add("lstm_2layers_drop0.4", **{**base_l, "layers": 2, "dropout": 0.4})
add("lstm_lr3e-3", **{**base_l, "lr": 3e-3})
add("lstm_bs64", **{**base_l, "bs": 64, "spe": 1600})
add("lstm_win", **{**base_l, "norm": "window"})
for mo in (3, 12, 30): add(f"lstm_m{mo}", **{**base_l, "months": mo})
# Otras estrategias multi-paso con la misma base
add("lstm_auto_L168", **{**base_l, "model": "lstm_auto"})
add("lstm_auto_h128", **{**base_l, "model": "lstm_auto", "hidden": 128})
add("seq2seq_L168", **{**base_l, "model": "seq2seq", "epochs": 12, "spe": 500})
add("seq2seq_h128", **{**base_l, "model": "seq2seq", "hidden": 128, "epochs": 12, "spe": 500})
add("seq2seq_L48", **{**base_l, "model": "seq2seq", "L": 48, "epochs": 12, "spe": 500})
add("lstm_h128_lr3e-3", **{**base_l, "hidden": 128, "lr": 3e-3})
add("lstm_auto_h128_lr3e-3", **{**base_l, "model": "lstm_auto", "hidden": 128, "lr": 3e-3})
add("seq2seq_h128_lr3e-3", **{**base_l, "model": "seq2seq", "hidden": 128, "lr": 3e-3, "epochs": 12, "spe": 500})

if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    P = Panel(load_panel(100))
    for rid, cfg in R.items():
        if not rid.startswith(pref): continue
        runs = load_json("runs.json")
        if rid in runs: continue
        print("=====", rid, cfg, flush=True)
        net, info = fit(P, cfg)
        torch.save(net.state_dict(), os.path.join(ROOT, "checkpoints", rid + ".pt"))
        runs = load_json("runs.json"); runs[rid] = info; save_json(runs, "runs.json")
        print(f"-> {rid}: val_mase={info['best_val_mase']:.4f} ep={info['best_epoch']} t={info['train_time']:.0f}s", flush=True)
