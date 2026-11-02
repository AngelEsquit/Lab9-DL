# Celdas de las secciones 6-12 del notebook. Se ejecuta desde build_notebook.py (usa md() y code()).
md('''
## 6. Panel de modelado (100 clientes) y módulo `src/lab9.py`
Por restricciones de cómputo (GPU de 6 GB) todos los modelos usan **el mismo subconjunto de 100 clientes**, elegido con semilla fija (42)
entre los 320 completos (`src/prep_data.py`). El módulo `src/lab9.py` contiene las implementaciones propias (Dataset de ventanas en GPU,
`backtest`, métricas, MLP, DLinear, NLinear, LSTM, LSTM autorregresivo y seq2seq, y el ciclo de entrenamiento `fit`).
Las celdas siguientes lo importan y **muestran su código fuente**; el entrenamiento pesado se corrió con los scripts de `src/` y sus
resultados (`results/*.json`) se cargan aquí.
''')
code('''
import sys, json, inspect
sys.path.insert(0, "../src" if os.path.basename(os.getcwd()) == "notebooks" else "src")
import lab9
from lab9 import *
RES_DIR = lab9.RES
df100 = load_panel(100); P = Panel(df100)
scale = mase_scale(df100.iloc[:P.pos_train_end])
print("clientes:", P.N, "| horas:", P.T)
for s in ("train", "val", "test"):
    print(f"ventanas {s} (L=168, H=24, stride 1): {len(P.origins(s, 168)) * P.N:,}")
''')
md('''
## 7. Implementaciones propias (PyTorch)
Todas reciben `past (B, L, 1+C)` (valor normalizado + calendario) y `fut (B, H, C)` (calendario futuro) y devuelven `(B, H)`.
`Net` añade la normalización (z-score por serie o por ventana, tipo RevIN sin parámetros afines) y la reversión.
''')
code('''
for obj in (lab9.MLP, lab9.DLinear, lab9.NLinear):
    print(inspect.getsource(obj))
''')
code('''
for obj in (lab9.LSTMMulti, lab9.LSTMAuto, lab9.Seq2Seq, lab9.Net):
    print(inspect.getsource(obj))
''')
md('''
### 7.1 Ciclo de entrenamiento y backtest
Pérdida L1, AdamW, cosine annealing, recorte de gradiente, early stopping **solo con validación**. La prueba se evalúa una vez por modelo.
''')
code('''
print(inspect.getsource(lab9.fit))
print(inspect.getsource(lab9.make_predict_fn))
''')
md("### 7.2 Ejemplo ejecutable: DLinear (3 epochs) y backtest en prueba")
code('''
cfg = dict(model="dlinear", L=168, epochs=3, spe=300, lr=3e-3)
net, info = fit(P, cfg, verbose=False)
bt100 = lab9.backtest(make_predict_fn(net, P, 168), df100)
print("MASE val:", round(info["best_val_mase"], 4), "| prueba:", lab9.metrics(bt100, scale).round(3).to_dict(), f"| {info['train_time']:.0f} s")
''')

md('''
## 8. Baselines, LightGBM y modelos de librería
Scripts: `src/run_baselines.py` (StatsForecast: Naive, SeasonalNaive 24/168, MSTL; LightGBM global recursivo con rezagos 1–24, 168 y calendario),
`src/run_nf.py` + `src/run_nf_test.py` (NeuralForecast: NHITS, TCN, PatchTST, iTransformer) y `src/run_chronos.py` (Chronos-Bolt zero-shot).
**Hallazgo:** al pasar el calendario como `futr_exog_list` NHITS y TCN divergían (MASE 1.66 y 29.5 en validación); sin exógenas convergen
(0.833 y 0.890), por lo que los modelos de NeuralForecast se reportan sin covariables. **DeepAR:** `NeuralForecast.DeepAR` no fue viable en la GPU de 6 GB (error de memoria en validación y tiempos excesivos), por lo que se
implementó **DeepAR propio en PyTorch** (`src/deepar.py`: LSTM de 2 capas autorregresivo, cabeza Student-t, calendario como covariables,
entrenado por máxima verosimilitud con teacher forcing y pronóstico por muestreo de 100 trayectorias; cuantiles P10/P50/P90 de las muestras).
Los modelos de NeuralForecast se entrenan una vez (datos hasta 2014-09-30) y se evalúan en prueba con `nf.predict` origen por origen.
''')
code('''
runs = load_json("runs.json"); runs_nf = load_json("runs_nf.json"); runs_lgb = load_json("runs_lgb.json")
for k, v in runs_nf.items(): print(f"{k:22s} val_MASE={v['val_mase']:.4f} params={v['params_total']:,} t={v['train_time']:.0f}s")
for k, v in runs_lgb.items(): print(f"{k:22s} val_MASE={v['val_mase']:.4f}")
for k, v in load_json("runs_deepar.json").items(): print(f"{k:22s} val_MASE={v['best_val_mase']:.4f} params={v['params_total']:,} t={v['train_time']:.0f}s")
''')

md('''
## 9. Iteraciones de hiperparámetros (5.2)
Selección **solo con validación** (2014-07-01 a 2014-09-30). Se registran configuración (L, meses de historia, normalización, hidden, capas,
dropout, lr, batch), pérdidas por epoch, MAE y MASE de validación, parámetros, tiempos y memoria pico de GPU (`runs.json`).
''')
code('''
rows = []
for rid, r in runs.items():
    c, hs = r["cfg"], r["history"]
    rows.append(dict(id=rid, L=c["L"], meses=c["months"] or 30, norm=c["norm"], hidden=c["hidden"], capas=c["layers"], dropout=c["dropout"], lr=c["lr"],
                     bs=c["bs"], epochs=len(hs), train_loss=hs[-1]["train_loss"], val_loss=hs[-1]["val_loss"], val_mae=min(h["val_mae"] for h in hs),
                     val_mase=r["best_val_mase"], params=r["params_total"], s_por_epoch=np.mean([h["epoch_time"] for h in hs]),
                     t_total_s=r["train_time"], gpu_mb=r["gpu_peak_mb"]))
it = pd.DataFrame(rows).sort_values("val_mase"); it.round(4)
''')
code('''
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for a, rid in zip(axs, ["dlinear_L168", "mlp_L168_win", "lstm_h128_lr3e-3"]):
    hs = runs[rid]["history"]; e = [x["epoch"] for x in hs]
    a.plot(e, [x["train_loss"] for x in hs], label="entrenamiento"); a.plot(e, [x["val_loss"] for x in hs], label="validación")
    a.set_title(rid); a.set_xlabel("epoch")
axs[0].legend(); plt.show()
''')
md('''
**¿Menor pérdida de entrenamiento implica mejor MASE de validación?** No necesariamente: ver la comparación de las curvas y de la columna
`val_mase` contra `train_loss` en la tabla de iteraciones (las LSTM alcanzan pérdidas de entrenamiento comparables a las de DLinear/MLP,
pero un MASE de validación peor).
''')

md("## 10. Estrategias multi-paso del LSTM (5.1) y local vs global (5.3)")
code('''
fin = {}
for f in ("final_baselines", "final_own", "final_nf", "final_chronos"): fin.update(load_json(f + ".json"))
h = np.arange(1, 25)
for k, lab in [("LSTM (multi-salida)", "multi-salida"), ("LSTM (autorregresivo)", "autorregresiva"), ("Seq2seq LSTM", "encoder-decoder")]:
    plt.plot(h, fin[k]["mae_by_h"], label=lab)
plt.xlabel("h"); plt.ylabel("MAE (kWh)"); plt.legend(); plt.title("MAE por paso del horizonte"); plt.show()
''')
code('''
lg = load_json("local_global.json")
print(f"10 clientes {lg['clients']}")
print(f"MASE global={lg['MASE_global']:.4f} | local={lg['MASE_local']:.4f} (entrenamiento local total {lg['train_time_local_total']:.0f} s en 10 modelos)")
pd.DataFrame(lg["per_client"]).round(4)
''')

md('''
## 11. Reentrenamiento mensual y semillas
`src/run_retrain.py`: al inicio de cada mes de prueba se reentrena con toda la historia disponible (ventana expandible) y se pronostica ese mes.
`src/run_seeds.py`: mejor modelo propio (MLP) y mejor modelo de librería (iTransformer) con 3 semillas.
''')
code('''
rt = load_json("retrain.json")
print(pd.DataFrame({k: {"MASE fijo": v["MASE_fixed"], "MASE mensual": v["MASE"], "MAE fijo": v["MAE_fixed"], "MAE mensual": v["MAE"]} for k, v in rt.items()}).T.round(4))
print(json.dumps(load_json("seeds.json"), indent=1))
''')

md("## 12. Comparación final sobre prueba (ordenada por MASE) y archivo de pronósticos")
code('''
cmp = pd.read_csv(os.path.join(RES_DIR, "comparison.csv")); cmp.round(4)
''')
code('''
import matplotlib.image as mpimg
FIGDIR = os.path.join(lab9.ROOT, "report", "figs")
for f in ("pareto.png", "mae_horizon.png", "months.png"):
    if os.path.exists(os.path.join(FIGDIR, f)):
        plt.figure(figsize=(8, 3.6)); plt.imshow(mpimg.imread(os.path.join(FIGDIR, f))); plt.axis("off"); plt.show()
pred = pd.read_csv(os.path.join(RES_DIR, "predictions_test.csv.gz"))
print(pred.shape, pred.modelo.nunique(), "modelos;", pred.columns.tolist()); pred.head()
''')
