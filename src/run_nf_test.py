"""Evaluacion UNICA en prueba (oct-dic 2014) de los modelos NeuralForecast. Para cada familia se elige, SOLO con el MASE de
validacion, entre las corridas candidatas (1000 y 3000 pasos); los pesos entrenados (checkpoints/nf/<id>, datos <= 2014-09-30)
NO se reentrenan: se predice origen por origen con nf.predict() usando solo datos hasta las 23:00 anteriores.
Uso: python src/run_nf_test.py [id ...]   (sin ids: seleccion automatica por validacion)"""
import sys; sys.argv_ = sys.argv[1:]
from run_nf import *

NAMES = {"nhits": "NHITS", "tcn": "TCN", "patchtst": "PatchTST", "itransformer": "iTransformer"}
CANDIDATES = {"nhits": ["nhits_L168_noex", "nhits_L168_noex_3k"], "tcn": ["tcn_L168_noex", "tcn_L168_noex_3k"],
              "patchtst": ["patchtst_L168", "patchtst_L168_3k"], "itransformer": ["itransformer_L168", "itransformer_L168_3k"]}

if __name__ == "__main__":
    runs = load_json("runs_nf.json"); fin = {}
    if sys.argv_: ids = sys.argv_
    else: ids = [min([r for r in c if r in runs], key=lambda r: runs[r]["val_mase"]) for c in CANDIDATES.values()]
    for rid in ids:
        info = runs[rid]; kind = info["kind"]; name = NAMES[kind]
        nf = NeuralForecast.load(os.path.join(ROOT, "checkpoints", "nf", rid))
        alias = nf.models[0].__class__.__name__
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); parts = []
        for o in TEST_ORIGINS:
            d = Y[(Y.ds < o) & (Y.ds >= o - pd.Timedelta(hours=400))][["unique_id", "ds", "y"]]
            p = nf.predict(df=d); p["cutoff"] = o - pd.Timedelta(hours=1); parts.append(p)
        torch.cuda.synchronize(); tt = time.time() - t0
        out = pd.concat(parts, ignore_index=True)
        cache = long_to_cache(out, COLS, {"y_hat": alias})
        bt = backtest(cache_predict_fn(cache), df)
        rep = full_report(bt, scale, name)
        rep.update(family="Transformer/NF", L=info["L"], strategy="multi-salida", covariates=False,
                   params_total=info["params_total"], params_trainable=info["params_trainable"], train_time=info["train_time"],
                   infer_time=tt, hardware=f"GPU {torch.cuda.get_device_name(0)}", gpu_peak_mb=info["gpu_peak_mb"], best_cfg=rid, val_mase=info["val_mase"])
        save_preds(bt, name); fin[name] = rep
        print(name, rid, {k: round(float(v), 4) for k, v in rep.items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas")}, flush=True)
    old = load_json("final_nf.json"); old.update(fin); save_json(old, "final_nf.json")
