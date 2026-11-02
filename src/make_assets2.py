"""Graficas/tablas restantes: MASE vs meses de historia, tabla de iteraciones, exploracion, local vs global."""
import sys; sys.path.insert(0, "src")
exec(open("src/make_assets.py", encoding="utf-8").read().split("R = {}")[0])    # estilo y rutas
BS = chr(92)
runs, rl, rn = load_json("runs.json"), load_json("runs_lgb.json"), load_json("runs_nf.json")
refs = load_json("val_refs.json")

# ------------------------------------------------------------ MASE (validacion) vs meses de historia
x = [3, 12, 30]
curves = {"DLinear": [runs[f"dlinear_m{m}"]["best_val_mase"] for m in x],
          "LSTM": [runs[f"lstm_m{m}"]["best_val_mase"] for m in x],
          "LightGBM": [rl[f"lgb_m{m}"]["val_mase"] for m in x]}
if all(f"patchtst_m{m}" in rn for m in (3, 12)):
    curves["PatchTST"] = [rn["patchtst_m3"]["val_mase"], rn["patchtst_m12"]["val_mase"], rn["patchtst_L168"]["val_mase"]]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for c, (k, v) in zip(SER, curves.items()):
    ax.plot(x, v, marker="o", ms=4, color=c, lw=1.6, label=k)
ax.axhline(refs["SNaive168"], color=SER[7], lw=1.2, ls="--", label="SNaive168"); ax.axhline(refs["Chronos"], color=SER[5], lw=1.2, ls="--", label="Chronos")
ax.set_xticks(x); ax.set_xlabel("meses de historia de entrenamiento"); ax.set_ylabel("MASE en validacion"); ax.set_ylim(0.7, 1.3)
ax.legend(fontsize=5.8, ncol=2, loc="upper right"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "months.png"), dpi=220); plt.close(fig)
json.dump(curves, open(os.path.join(RES, "months_curves.json"), "w"))

# ------------------------------------------------------------ tabla de iteraciones (seleccion)
sel = ["dlinear_L48", "dlinear_L168", "dlinear_L336", "dlinear_L168_win", "dlinear_lr3e-3", "dlinear_m3", "dlinear_m12", "nlinear_L168", "nlinear_L336",
       "mlp_L168", "mlp_L168_h512_l3", "mlp_L336", "mlp_L168_win", "lstm_L48", "lstm_L168", "lstm_L336", "lstm_h32", "lstm_h128", "lstm_2layers",
       "lstm_2layers_drop0.4", "lstm_lr3e-3", "lstm_bs64", "lstm_win", "lstm_m3", "lstm_m12", "lstm_auto_L168", "lstm_auto_h128", "seq2seq_L168",
       "seq2seq_h128", "lstm_h128_lr3e-3", "lstm_auto_h128_lr3e-3", "seq2seq_h128_lr3e-3"]
rows = []
for rid in sel:
    r = runs[rid]; c = r["cfg"]; hs = r["history"]
    rows.append(dict(id=rid, modelo=c["model"], L=c["L"], meses=c["months"] or 30, norm="z" if c["norm"] == "z" else "ventana", hidden=c["hidden"] if c["model"] in ("mlp", "lstm", "lstm_auto", "seq2seq") else "-",
                     capas=c["layers"] if c["model"] in ("mlp", "lstm", "lstm_auto", "seq2seq") else "-", dropout=c["dropout"], lr=c["lr"], bs=c["bs"], H=24,
                     epochs=len(hs), loss_train=hs[-1]["train_loss"], loss_val=hs[-1]["val_loss"], val_mae=min(h["val_mae"] for h in hs), val_mase=r["best_val_mase"],
                     params=r["params_total"], t_epoch=float(np.mean([h["epoch_time"] for h in hs])), t_total=r["train_time"], gpu_mb=r["gpu_peak_mb"]))
I = pd.DataFrame(rows); I.to_csv(os.path.join(RES, "iterations.csv"), index=False)
PDF_SEL = ["dlinear_L48", "dlinear_L168", "dlinear_L336", "dlinear_m3", "dlinear_lr3e-3", "nlinear_L336", "mlp_L168", "mlp_L168_win", "lstm_L168", "lstm_h128",
           "lstm_2layers", "lstm_win", "lstm_m3", "lstm_h128_lr3e-3", "lstm_auto_h128_lr3e-3", "seq2seq_h128_lr3e-3"]
def itex(I):
    L = [BS + "begin{tabular}{@{}lrrrrrrrrrr@{}}", BS + "toprule",
         "Iteración & $L$ & meses & $h$/cap. & lr & ép. & MAE$_v$ & MASE$_v$ & params & s/ép. & MB " + BS + BS + " " + BS + "midrule"]
    for _, r in I[I.id.isin(PDF_SEL)].iterrows():
        hc = f"{r.hidden}/{r.capas}" if r.hidden != "-" else "-"
        L.append(f"{r.id.replace('_', ' ')} & {r.L} & {r.meses} & {hc} & {r.lr:g} & {r.epochs} & {r.val_mae:.1f} & {r.val_mase:.3f} & {int(r.params)} & {r.t_epoch:.1f} & {r.gpu_mb:.0f} " + BS + BS)
    L.append(BS + "midrule")
    for rid, nm in [("itransformer_L168", "itransformer L168"), ("nhits_L168_noex", "nhits L168"), ("tcn_L168_noex", "tcn L168"), ("patchtst_L168", "patchtst L168"),
                    ("patchtst_m3", "patchtst m3"), ("patchtst_m12", "patchtst m12")]:
        if rid not in rn: continue
        r = rn[rid]
        L.append(f"{nm} & {r['L']} & {r['months'] or 30} & - & 1e-3 & 1000 pasos & {r['val_mae']:.1f} & {r['val_mase']:.3f} & {int(r['params_total'])} & {r['train_time']:.0f} (total) & {r['gpu_peak_mb']:.0f} " + BS + BS)
    return "\n".join(L + [BS + "bottomrule", BS + "end{tabular}"])
open(os.path.join(RES, "iterations_table.tex"), "w", encoding="utf-8").write(itex(I))

# ------------------------------------------------------------ exploracion: perfil hora / dia y navidad (panel completo)
pa = pd.read_pickle(os.path.join(ROOT, "data", "panel_all.pkl"))
tot = pa.mean(axis=1)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
tr = tot[:"2014-06-30"]
axs[0].plot(range(24), tr.groupby(tr.index.hour).mean().values, color=SER[0], lw=1.6); axs[0].set_xlabel("hora del dia"); axs[0].set_ylabel("kWh medio por cliente")
axs[0].set_title("Perfil por hora", fontsize=7.5, loc="left")
dm = tr.groupby(tr.index.dayofweek).mean().values
axs[1].bar(range(7), dm, color=SER[0], width=0.6); axs[1].set_xticks(range(7)); axs[1].set_xticklabels(list("LMXJVSD")); axs[1].set_title("Perfil por dia de la semana", fontsize=7.5, loc="left"); axs[1].set_ylim(dm.min() * 0.9, dm.max() * 1.02)
xm = tot["2014-12-15":"2015-01-04"] if tot.index[-1] > pd.Timestamp("2015-01-01") else tot["2014-12-15":"2014-12-31"]
d = tot["2014-12-01":"2014-12-31"].resample("D").mean()
axs[2].plot(d.index.day, d.values, color=SER[0], lw=1.6)
for dd in (8, 25): axs[2].axvline(dd, color=SER[1], lw=1, ls="--")
axs[2].text(8.4, d.max(), "feriado 8", fontsize=5.5, color=INK2, va="top"); axs[2].text(25.4, d.max(), "Navidad", fontsize=5.5, color=INK2, va="top")
axs[2].set_xlabel("dia de diciembre 2014"); axs[2].set_title("Consumo diario medio", fontsize=7.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore.png"), dpi=220); plt.close(fig)

# ------------------------------------------------------------ local vs global
lg = load_json("local_global.json"); pcl = pd.DataFrame(lg["per_client"]).sort_values("mean_kwh")
fig, ax = plt.subplots(figsize=(3.4, 2.4)); y = np.arange(len(pcl))
ax.barh(y - 0.18, pcl.MASE_global, height=0.34, color=SER[0], label="global"); ax.barh(y + 0.18, pcl.MASE_local, height=0.34, color=SER[1], label="local")
ax.set_yticks(y); ax.set_yticklabels([f"{u} ({m:.0f})" for u, m in zip(pcl.unique_id, pcl.mean_kwh)], fontsize=5.5); ax.set_xlabel("MASE en prueba (entre parentesis: kWh medio)")
ax.legend(fontsize=6.5); ax.axvline(1, color=GRID); fig.tight_layout(); fig.savefig(os.path.join(FIG, "local_global.png"), dpi=220); plt.close(fig)
print("ok", list(curves))
