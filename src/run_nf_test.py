"""Evaluacion UNICA en prueba (oct-dic 2014) del mejor modelo NeuralForecast segun MASE de validacion (o de los ids dados).
Los pesos ya entrenados (checkpoints/nf/<id>) NO se reentrenan ni se refitean: solo se predice. Uso: python src/run_nf_test.py [id ...]"""
import sys; sys.argv_ = sys.argv[1:]
from run_nf import *

NAMES = {"nhits": "NHITS", "tcn": "TCN", "patchtst": "PatchTST", "itransformer": "iTransformer", "deepar": "DeepAR"}

if __name__ == "__main__":
    runs = load_json("runs_nf.json"); fin = load_json("final_nf.json")
    ids = sys.argv_ or [min(runs, key=lambda r: runs[r]["val_mase"])]
    for rid in ids:
        info = runs[rid]; kind = info["kind"]
        nf = NeuralForecast.load(os.path.join(ROOT, "checkpoints", "nf", rid))
        d = df_for(None, END)
        t0 = time.time()
        out = nf.cross_validation(d, n_windows=None, h=24, test_size=92 * 24, step_size=24, val_size=VAL_SIZE, refit=False)
        tt = time.time() - t0
        alias = nf.models[0].__class__.__name__
        cache = long_to_cache(out, COLS, {"y_hat": alias if kind != "deepar" else "DeepAR-median"})
        bt = backtest(cache_predict_fn(cache), df)
        name = NAMES[kind]
        rep = full_report(bt, scale, name)
        rep.update(family="Transformer/NF", L=info["L"], strategy="multi-salida", covariates=False,
                   params_total=info["params_total"], params_trainable=info["params_trainable"], train_time=info["train_time"],
                   infer_time=tt, hardware=f"GPU {torch.cuda.get_device_name(0)}", gpu_peak_mb=info["gpu_peak_mb"], best_cfg=rid)
        save_preds(bt, name); fin[name] = rep; save_json(fin, "final_nf.json")
        print(name, rid, {k: round(v, 4) for k, v in rep.items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas")}, flush=True)
