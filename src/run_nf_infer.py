"""Tiempo de inferencia (92 origenes, 100 clientes) de los modelos NeuralForecast usando los checkpoints guardados.
(En run_nf_test.py el tiempo de cross_validation incluia un reajuste del modelo, por lo que no sirve como tiempo de inferencia.)"""
import sys; sys.path.insert(0, "src")
from run_nf import *
IDS = {"NHITS": "nhits_L168_noex", "TCN": "tcn_L168_noex", "PatchTST": "patchtst_L168", "iTransformer": "itransformer_L168"}
res = {}
for name, rid in IDS.items():
    nf = NeuralForecast.load(os.path.join(ROOT, "checkpoints", "nf", rid))
    torch.cuda.synchronize(); t0 = time.time()
    for o in TEST_ORIGINS:
        d = Y[(Y.ds < o) & (Y.ds >= o - pd.Timedelta(hours=400))]
        _ = nf.predict(df=d)
    torch.cuda.synchronize(); res[name] = time.time() - t0
    print(name, round(res[name], 1), "s", flush=True)
save_json(res, "infer_nf.json")
