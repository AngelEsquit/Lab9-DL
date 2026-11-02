"""Modelos de NeuralForecast: NHITS, TCN, PatchTST, iTransformer, DeepAR. Entrena una vez con datos <= 2014-09-30
(val_size = ultimas 2208 h para early stopping), selecciona por MASE en validacion y evalua UNA vez en prueba."""
import sys, gc, warnings, logging; sys.path.insert(0, "src")
warnings.filterwarnings("ignore"); logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
from lab9 import *
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TCN, PatchTST, iTransformer, DeepAR
from neuralforecast.losses.pytorch import MAE, DistributionLoss

df = load_panel(100); P = Panel(df)
COLS = list(df.columns)
CAL = list(P.cal_df.columns)
scale = mase_scale(df.iloc[:P.pos_train_end])
Y = df.reset_index(names="ds").melt(id_vars="ds", var_name="unique_id", value_name="y")
Y = Y.merge(P.cal_df.reset_index(names="ds"), on="ds")[["unique_id", "ds", "y"] + CAL].sort_values(["unique_id", "ds"])
VAL_SIZE = 2208
TK = dict(enable_progress_bar=True, logger=False, enable_checkpointing=False, enable_model_summary=False)
COMMON = dict(h=24, max_steps=1000, val_check_steps=100, early_stop_patience_steps=5, scaler_type="standard", random_seed=1, **TK)
fut = CAL

def mk(kind, L, **kw):
    if kind == "nhits":
        return NHITS(input_size=L, **{"futr_exog_list": fut, **kw.pop("ex", {})}, n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[24, 12, 1],
                     **{**dict(mlp_units=[[256, 256]] * 3, windows_batch_size=512), **COMMON, **kw})
    if kind == "tcn":
        return TCN(input_size=L, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32], encoder_hidden_size=64, decoder_hidden_size=128,
                   **{"futr_exog_list": fut, **kw.pop("ex", {})}, windows_batch_size=256, **{**COMMON, **kw})
    if kind == "patchtst":
        return PatchTST(input_size=L, encoder_layers=3, hidden_size=128, n_heads=8, linear_hidden_size=256, patch_len=16, stride=8,
                        windows_batch_size=512, **{**COMMON, "learning_rate": 1e-3, **kw})
    if kind == "itransformer":
        return iTransformer(input_size=L, n_series=len(COLS), hidden_size=128, n_heads=8, e_layers=2, d_ff=256,
                            windows_batch_size=32, **{**COMMON, "learning_rate": 1e-3, **kw})
    if kind == "deepar":
        return DeepAR(input_size=L, h_train=24, lstm_n_layers=2, lstm_hidden_size=64, **{"futr_exog_list": fut, **kw.pop("ex", {})},
                      loss=DistributionLoss("StudentT", level=[80]), valid_loss=MAE(),
                      **{"windows_batch_size": 512, **COMMON, "scaler_type": "standard", **kw})
    raise ValueError(kind)

# (id, familia, L, meses de historia, kwargs)
CONFIGS = [
    ("nhits_L168", "nhits", 168, None, {}), ("patchtst_L168", "patchtst", 168, None, {}),
    ("tcn_L168", "tcn", 168, None, {}),
    # v2: diagnostico (en v1 NHITS quedo sin converger en 1000 pasos y TCN divergio con picos de perdida)
    ("nhits_L168_v2", "nhits", 168, None, dict(learning_rate=3e-3, mlp_units=[[128, 128]] * 3, windows_batch_size=256)),
    ("tcn_L168_noex", "tcn", 168, None, dict(ex=dict(futr_exog_list=None))),
    ("nhits_L168_noex", "nhits", 168, None, dict(ex=dict(futr_exog_list=None))),
    ("itransformer_L168", "itransformer", 168, None, {}),
    ("deepar_L168_noex", "deepar", 168, None, dict(ex=dict(futr_exog_list=None), inference_windows_batch_size=16, valid_batch_size=16, max_steps=300, val_check_steps=50, windows_batch_size=128)),
    ("patchtst_m3", "patchtst", 168, 3, {}), ("patchtst_m12", "patchtst", 168, 12, {}),
    ("tcn_L168_v2", "tcn", 168, None, dict(learning_rate=3e-4, gradient_clip_val=1.0)),
]

def df_for(months, end):
    d = Y[Y.ds <= end]
    if months is not None:
        start = pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1) - pd.DateOffset(months=months)
        d = d[d.ds >= start]
    return d

def cv(nf, d, test_size):
    return nf.cross_validation(d, n_windows=None, h=24, test_size=test_size, step_size=24, val_size=VAL_SIZE, refit=False, use_fitted=True)

def mcol(out, name):
    return {c: c for c in out.columns if c.startswith(name)}

if __name__ == "__main__":
    only = sys.argv[1:]
    runs = load_json("runs_nf.json")
    for rid, kind, L, months, kw in CONFIGS:
        if (only and not any(rid.startswith(o) for o in only)) or rid in runs: continue
        print("=====", rid, flush=True)
        model = mk(kind, L, **kw)
        nf = NeuralForecast(models=[model], freq="h")
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        nf.fit(df_for(months, VAL_END), val_size=VAL_SIZE)
        train_time = time.time() - t0
        m = nf.models[0]
        tot, trn = n_params(m)
        out = cv(nf, df_for(months, VAL_END), VAL_SIZE)           # validacion: ventanas diarias jul-sep
        alias = m.__class__.__name__
        # MASE de validacion con el mismo calculo que los modelos propios
        cache = long_to_cache(out, COLS, {"y_hat": alias if kind != "deepar" else "DeepAR-median"})
        vo = pd.date_range("2014-07-01", "2014-09-30", freq="1D")
        val_bt = backtest(cache_predict_fn(cache), df, vo)
        vm = metrics(val_bt, scale)
        info = dict(kind=kind, L=L, months=months, val_mase=vm["MASE"], val_mae=vm["MAE"], params_total=tot,
                    params_trainable=trn, train_time=train_time, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                    steps=int(m.trainer.global_step) if hasattr(m, "trainer") and m.trainer is not None else None,
                    val_traj=[[int(a), float(b)] for a, b in getattr(m, "valid_trajectories", [])],
                    train_traj=[[int(a), float(b)] for a, b in getattr(m, "train_trajectories", [])])
        print(rid, {k: v for k, v in info.items() if "traj" not in k}, flush=True)
        os.makedirs(os.path.join(ROOT, "checkpoints", "nf"), exist_ok=True)
        nf.save(os.path.join(ROOT, "checkpoints", "nf", rid), overwrite=True, save_dataset=False)
        runs[rid] = info; save_json(runs, "runs_nf.json")
        del nf, model; gc.collect(); torch.cuda.empty_cache()
