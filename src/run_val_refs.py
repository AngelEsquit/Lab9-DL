"""MASE de VALIDACION de referencia (SNaive168 y Chronos zero-shot) para la grafica MASE vs meses de historia (5.2)."""
import sys; sys.path.insert(0, "src")
from lab9 import *
from chronos import BaseChronosPipeline
df = load_panel(100); P = Panel(df); scale = mase_scale(df.iloc[:P.pos_train_end])
vo = pd.date_range("2014-07-01", "2014-09-30", freq="1D")
sn = lambda hist, o: np.clip(hist.values[-168:-168 + H].T, 0, None)
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=DEV, torch_dtype=torch.float32)
def ch(hist, o):
    q, _ = pipe.predict_quantiles(torch.tensor(hist.values[-512:].T, dtype=torch.float32), prediction_length=24, quantile_levels=[0.5])
    return q[..., 0].clamp_min(0).numpy()
out = dict(SNaive168=float(metrics(backtest(sn, df, vo), scale)["MASE"]), Chronos=float(metrics(backtest(ch, df, vo), scale)["MASE"]))
print(out); save_json(out, "val_refs.json")
