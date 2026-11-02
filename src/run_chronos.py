"""Chronos-Bolt zero-shot: contexto = ultimas 512 h (sin entrenamiento), cuantiles P10/P50/P90."""
import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
from lab9 import *
from chronos import BaseChronosPipeline

df = load_panel(100); P = Panel(df); scale = mase_scale(df.iloc[:P.pos_train_end])
name = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-bolt-small"
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 512
pipe = BaseChronosPipeline.from_pretrained(name, device_map=DEV, torch_dtype=torch.float32)


def predict(hist, origin):
    ctx = torch.tensor(hist.values[-CTX:].T, dtype=torch.float32)                  # (N, CTX)
    q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
    q = q.clamp_min(0).numpy()                                                      # (N, 24, 3)
    return {"y_hat": q[..., 1], "p10": q[..., 0], "p90": q[..., 2]}


torch.cuda.reset_peak_memory_stats()
bt = backtest(predict, df)
rep = full_report(bt, scale, "Chronos-Bolt")
rep.update(family="Fundacional", L=CTX, strategy="directa (zero-shot)", covariates=False,
           params_total=sum(p.numel() for p in pipe.model.parameters()), params_trainable=0, train_time=0.0,
           infer_time=bt.attrs["infer_time"], hardware=f"GPU {torch.cuda.get_device_name(0)}",
           gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, model_id=name)
save_preds(bt, "Chronos-Bolt")
res = load_json("final_chronos.json"); res["Chronos-Bolt"] = rep; save_json(res, "final_chronos.json")
print({k: round(v, 4) for k, v in rep.items() if isinstance(v, float)})
