"""Tabla comparativa (seccion 6), archivo unificado de pronosticos y graficas del reporte."""
import sys; sys.path.insert(0, "src")
from lab9 import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(ROOT, "report", "figs"); os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False, "text.color": INK})

R = {}
for f in ("final_baselines", "final_own", "final_nf", "final_chronos", "final_deepar"): R.update(load_json(f + ".json"))
ORDER = ["Naive", "SNaive24", "SNaive168", "MSTL", "LightGBM", "MLP", "DLinear", "NLinear", "LSTM (multi-salida)", "LSTM (autorregresivo)",
         "Seq2seq LSTM", "NHITS", "TCN", "PatchTST", "iTransformer", "DeepAR", "Chronos-Bolt"]
R = {k: R[k] for k in ORDER}
_inf = load_json("infer_nf.json")
for _k, _v in _inf.items(): R[_k]["infer_time"] = _v

# ---------------------------------------------------------------- tabla comparativa
rows = []
for k, r in R.items():
    rows.append(dict(Modelo=k, Familia=r["family"], L=r["L"], Estrategia=r["strategy"], Covariables="si" if r["covariates"] else "no",
                     Params=r["params_total"], Params_entren=r["params_trainable"], T_entren_s=r["train_time"], T_infer_s=r["infer_time"],
                     Hardware="GPU" if "GPU" in str(r["hardware"]) else "CPU", MAE=r["MAE"], RMSE=r["RMSE"], WAPE=r["WAPE"], MAPE=r["MAPE"],
                     MASE=r["MASE"], MASE_22_28dic=r["MASE_xmas"], Pinball=r.get("pinball", np.nan), Cobertura_P10_P90=r.get("coverage_p10_p90", np.nan)))
T = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
T.to_csv(os.path.join(RES, "comparison.csv"), index=False)
def tex(T):
    BS = chr(92)
    L = [BS + "begin{tabular}{@{}lllrrrrrrrrrr@{}}", BS + "toprule",
         "Modelo & $L$ & Estrategia & Cov. & Params & $t_{ent}$ (s) & $t_{inf}$ (s) & HW & MAE & RMSE & WAPE & MASE & MASE$_{22-28}$ " + BS + BS + " " + BS + "midrule"]
    for _, r in T.iterrows():
        n = r.Modelo.replace("(multi-salida)", "(MS)").replace("(autorregresivo)", "(AR)")
        par = f"{int(r.Params):,}".replace(",", BS + ",")
        L.append(f"{n} & {r.L} & {r.Estrategia.split(' ')[0]} & {r.Covariables} & {par} & {r.T_entren_s:.0f} & {r.T_infer_s:.1f} & {r.Hardware} & {r.MAE:.1f} & {r.RMSE:.1f} & {r.WAPE:.3f} & {r.MASE:.3f} & {r.MASE_22_28dic:.2f} " + BS + BS)
    return "\n".join(L + [BS + "bottomrule", BS + "end{tabular}"])
open(os.path.join(RES, "comparison_table.tex"), "w", encoding="utf-8").write(tex(T))

# ---------------------------------------------------------------- archivo unificado de pronosticos
parts = []
for k in ORDER:
    d = load_preds(k)[["unique_id", "ds", "y_hat"]].copy(); d.insert(2, "modelo", k); parts.append(d)
U = pd.concat(parts, ignore_index=True)
U.to_csv(os.path.join(RES, "predictions_test.csv.gz"), index=False, float_format="%.3f")
print("pronosticos:", U.shape, "modelos", U.modelo.nunique())

# ---------------------------------------------------------------- Fig 1: Pareto MASE vs tiempo de entrenamiento
fig, ax = plt.subplots(figsize=(3.4, 2.7))
P = T[T.T_entren_s > 0]
OFF = {"NLinear": (-31, -2), "DLinear": (-8, 6), "MLP": (-10, -11), "iTransformer": (4, -10), "LightGBM": (-10, 6), "PatchTST": (2, 6), "LSTM (multi-salida)": (-36, -3), "LSTM (autorregresivo)": (6, -3), "TCN": (-7, -11), "DeepAR": (5, -3), "NHITS": (-12, -12), "Seq2seq LSTM": (5, 2)}
for i, (_, r) in enumerate(P.iterrows()):
    ax.scatter(r.T_entren_s, r.MASE, s=22, color=SER[0], zorder=3, edgecolor="#fcfcfb", linewidth=1)
    ax.annotate(r.Modelo.replace("(multi-salida)", "MS").replace("(autorregresivo)", "AR").replace(" LSTM", ""), (r.T_entren_s, r.MASE),
                xytext=OFF.get(r.Modelo, (3, 2)), textcoords="offset points", fontsize=5.5, color=INK2)
ax.axhline(R["Chronos-Bolt"]["MASE"], color=SER[1], lw=1.2); ax.text(P.T_entren_s.min(), R["Chronos-Bolt"]["MASE"] - 0.008, "Chronos (zero-shot)", color=INK2, fontsize=6, va="top")
ax.axhline(R["SNaive24"]["MASE"], color=SER[3], lw=1.2); ax.text(P.T_entren_s.min(), R["SNaive24"]["MASE"] + 0.004, "SNaive24", color=INK2, fontsize=6)
ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300, 1000, 3000]); ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000"]); ax.minorticks_off(); ax.set_xlabel("tiempo de entrenamiento (s, log)"); ax.set_ylabel("MASE en prueba")
ax.set_ylim(0.74, 1.06); fig.tight_layout(); fig.savefig(os.path.join(FIG, "pareto.png"), dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: MAE por paso del horizonte
h = np.arange(1, 25)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
a = axs[0]
for c, k in zip(SER[:3], ["LSTM (multi-salida)", "LSTM (autorregresivo)", "Seq2seq LSTM"]):
    a.plot(h, R[k]["mae_by_h"], color=c, lw=1.6, label={"LSTM (multi-salida)": "multi-salida", "LSTM (autorregresivo)": "autorregresiva", "Seq2seq LSTM": "encoder-decoder"}[k])
a.set_title("Estrategias multi-paso (LSTM)", fontsize=8, loc="left"); a.set_xlabel("paso del horizonte h"); a.set_ylabel("MAE (kWh)"); a.legend(fontsize=6.5)
a = axs[1]
for c, k in zip(SER, ["SNaive24", "LightGBM", "MLP", "LSTM (multi-salida)", "NHITS", "PatchTST", "iTransformer", "Chronos-Bolt"]):
    a.plot(h, R[k]["mae_by_h"], color=c, lw=1.6, label=k.replace(" (multi-salida)", ""))
a.set_title("Modelos representativos", fontsize=8, loc="left"); a.set_xlabel("paso del horizonte h"); a.legend(fontsize=5.5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "mae_horizon.png"), dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: curvas de perdida de 3 iteraciones
runs = load_json("runs.json")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=False)
for a, rid in zip(axs, ["dlinear_L168", "mlp_L168_win", "lstm_h128_lr3e-3"]):
    hs = runs[rid]["history"]; e = [x["epoch"] for x in hs]
    a.plot(e, [x["train_loss"] for x in hs], color=SER[0], lw=1.6, label="entrenamiento")
    a.plot(e, [x["val_loss"] for x in hs], color=SER[1], lw=1.6, label="validacion")
    a.set_title(rid, fontsize=7.5, loc="left"); a.set_xlabel("epoch")
axs[0].set_ylabel("perdida L1 (z-score)"); axs[0].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "loss_curves.png"), dpi=220); plt.close(fig)
print("ok figs 1-3")

# ---------------------------------------------------------------- Fig 4: error por hora del dia y semana 22-28 dic
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
a = axs[0]
for c, k in zip(SER, ["SNaive24", "LightGBM", "MLP", "LSTM (multi-salida)", "NHITS", "PatchTST", "iTransformer", "Chronos-Bolt"]):
    a.plot(range(24), R[k]["mae_by_hour"], color=c, lw=1.5, label=k.replace(" (multi-salida)", ""))
a.set_xlabel("hora del día (ds)"); a.set_ylabel("MAE (kWh)"); a.set_title("Error por hora del día", fontsize=8, loc="left"); a.legend(fontsize=5.5, ncol=2)
a = axs[1]
names = [k for k in T.Modelo if k != "Naive"][::-1]
y = np.arange(len(names))
a.barh(y - 0.2, [R[k]["MASE"] for k in names], height=0.38, color=SER[0], label="prueba completa")
a.barh(y + 0.2, [R[k]["MASE_xmas"] for k in names], height=0.38, color=SER[1], label="22–28 dic")
a.set_yticks(y); a.set_yticklabels([k.replace("(multi-salida)", "MS").replace("(autorregresivo)", "AR") for k in names], fontsize=5.5)
a.set_xlabel("MASE"); a.set_title("Navidad: MASE normal vs. 22–28 dic", fontsize=8, loc="left"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "hour_xmas.png"), dpi=220); plt.close(fig)
print("ok fig4")
