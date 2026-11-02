"""Genera notebooks/Lab9_Series_de_Tiempo.ipynb (secciones 1-4 iniciales).
Uso: python notebooks/build_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# CC3092 - Laboratorio 9: Series de tiempo y forecasting
Dataset: ElectricityLoadDiagrams20112014 (UCI). Objetivo: pronóstico day-ahead (H = 24 h) del consumo por hora.

**Índice:** 1. Datos · 2. Exploración y preparación · 3. Dataset de ventanas · 4. Auditoría de fugas · 5. Backtest y métricas
(las secciones de modelos se agregarán después).
""")

# ---------------------------------------------------------------- 1. Datos
md("## 1. Datos")
code("""
import os, urllib.request, zipfile, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import holidays

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = "../data" if os.path.basename(os.getcwd()) == "notebooks" else "data"
os.makedirs(DATA_DIR, exist_ok=True)
ZIP_PATH = f"{DATA_DIR}/electricity.zip"
TXT_PATH = f"{DATA_DIR}/LD2011_2014.txt"

if not os.path.exists(TXT_PATH):
    url = ("https://archive.ics.uci.edu/static/public/321/"
           "electricityloaddiagrams20112014.zip")
    urllib.request.urlretrieve(url, ZIP_PATH)
    zipfile.ZipFile(ZIP_PATH).extract("LD2011_2014.txt", DATA_DIR)

raw = pd.read_csv(TXT_PATH, sep=";", decimal=",", index_col=0, parse_dates=True)
print(raw.shape)   # 140,256 filas x 370 clientes
raw.iloc[:3, :4]
""")

md("""
## 2. Exploración y preparación de la serie
**Pipeline:** datos crudos (kW cada 15 min) → series horarias (kWh) limpias → ventanas → tensores.

### 2.1 Agregación horaria y filtro de clientes
El valor crudo está en kW promediado sobre 15 min, por lo que la energía de cada registro es `kW * 0.25` kWh;
sumando los 4 registros de la hora se obtiene el consumo horario en kWh.
Se conservan los clientes con datos completos desde 2012-01-01 (sin ceros iniciales por cliente que aún no existía).
""")
code("""
START, END = "2012-01-01 00:00", "2014-12-31 23:00"
TRAIN_END, VAL_END = "2014-06-30 23:00", "2014-09-30 23:00"
H, L, STRIDE = 24, 168, 1

kwh_15 = raw * 0.25                       # kW -> kWh por intervalo de 15 min
# min_count evita convertir horas sin datos en 0 al sumar
hourly = kwh_15.resample("1h").sum(min_count=4)
hourly = hourly.loc[START:END]
print("Horas esperadas:", len(pd.date_range(START, END, freq="1h")), "| filas:", len(hourly))

# Los clientes que empiezan a medir después de 2012 aparecen con 0 (no NaN) en el archivo original:
# se consideran 'completos' si no tienen NaN y su primer valor no nulo ocurre en 2012-01-01.
first_nonzero = hourly.apply(lambda s: s.ne(0).idxmax() if s.ne(0).any() else pd.NaT)
complete = (first_nonzero <= pd.Timestamp(START) + pd.Timedelta(hours=1)) & hourly.notna().all()
panel = hourly.loc[:, complete]
print(f"Clientes tras filtro 2012-2014: {panel.shape[1]} de {raw.shape[1]}")
print(f"Observaciones por serie: {len(panel):,} | panel completo: {panel.size:,}")
""")

md("""
### 2.2 Cambio de horario (DST)
En el archivo original, el día de adelanto de hora (marzo) tiene una hora sin datos (la 01:00-02:00 no existe) y el de
atraso (octubre) tiene una hora duplicada. Al agregar con `resample("1h")` sobre un índice naive:
- **Marzo:** la hora 01:00 (inexistente) queda casi vacía -> se marca NaN y se **interpola linealmente** entre vecinas.
- **Octubre:** la hora 01:00 acumula dos horas reales (pico artificial ~2x) -> se **reemplaza por el promedio**
  de la misma hora en el mismo día de la semana anterior y posterior.

Se verifica abajo qué ocurre realmente en los datos.
""")
code("""
# Fechas de cambio de horario en Portugal (último domingo de marzo y octubre)
def last_sunday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)

dst_days = [(last_sunday(y, 3), last_sunday(y, 10)) for y in (2012, 2013, 2014)]
for spring, fall in dst_days:
    for d in (spring, fall):
        day = panel.loc[d:d + pd.Timedelta(hours=23)].sum(axis=1)
        print(d.date(), "| total horas 00-04:", np.round(day.iloc[:5].values, 0))
""")
code("""
panel = panel.copy()
for spring, fall in dst_days:
    # Marzo: la hora 01:00 casi no tiene registros (valor ~4% del normal) -> NaN e interpolacion lineal
    panel.loc[spring + pd.Timedelta(hours=1)] = np.nan
    # Octubre: la hora 01:00 acumula 2 horas reales (~2x) -> promedio de +-7 dias a la misma hora
    h_fall = fall + pd.Timedelta(hours=1)
    panel.loc[h_fall] = (panel.loc[h_fall - pd.Timedelta(days=7)].values
                         + panel.loc[h_fall + pd.Timedelta(days=7)].values) / 2
panel = panel.interpolate(limit_direction="both")
assert panel.isna().sum().sum() == 0
""")

md("""
### 2.3 Subconjunto de clientes (opcional)
Si el cómputo es limitado se usa un subconjunto fijo con semilla (mínimo 100) para **todos** los modelos.
""")
code("""
N_CLIENTS = None            # None = todos; o un entero >= 100
if N_CLIENTS is not None:
    rng = np.random.default_rng(SEED)
    cols = sorted(rng.choice(panel.columns, size=N_CLIENTS, replace=False))
    panel = panel[cols]
CLIENTS = list(panel.columns)
print(len(CLIENTS), "clientes en uso")
""")

md("### 2.4 Perfiles por hora, día de la semana y feriados de Portugal")
code("""
pt_holidays = holidays.country_holidays("PT", years=range(2012, 2015))
is_hol = pd.Series(panel.index.normalize().isin(pd.to_datetime(list(pt_holidays.keys()))), index=panel.index)
total = panel.mean(axis=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
total.groupby(total.index.hour).mean().plot(ax=ax[0], title="Perfil por hora del día")
total.groupby(total.index.dayofweek).mean().plot(kind="bar", ax=ax[1], title="Por día de la semana (0=lun)")
daily = total.resample("1D").mean()
daily.plot(ax=ax[2], lw=.6, title="Consumo diario promedio y feriados")
hd = daily[daily.index.isin(pd.to_datetime(list(pt_holidays.keys())))]
ax[2].scatter(hd.index, hd.values, c="r", s=10, label="feriado"); ax[2].legend()
plt.tight_layout(); plt.show()

christmas = total[(total.index.month == 12) & (total.index.day.isin([24, 25, 26]))].mean()
newyear = total[((total.index.month == 12) & (total.index.day == 31)) | ((total.index.month == 1) & (total.index.day <= 2))].mean()
print(f"Consumo medio global: {total.mean():.2f} | Navidad: {christmas:.2f} | Año Nuevo: {newyear:.2f}")
""")

md("### 2.5 Escalas entre clientes y normalización")
code("""
mean_by_client = panel.loc[:TRAIN_END].mean()
print(mean_by_client.describe()[["min", "50%", "max"]])
print("Razón max/min:", mean_by_client.max() / max(mean_by_client.min(), 1e-9))
""")
md("""
**Decisión (a completar con los resultados):** las escalas difieren en varios órdenes de magnitud, por lo que
una normalización global dominaría la pérdida con los clientes grandes. Se usa **z-score por serie con
estadísticos calculados solo en entrenamiento** (media y desviación de 2012-01-01 a 2014-06-30), y para modelos
con RevIN, normalización por ventana. Las métricas se calculan siempre en kWh originales.
""")
code("""
mu = panel.loc[:TRAIN_END].mean()
sd = panel.loc[:TRAIN_END].std().replace(0, 1.0)
panel_z = (panel - mu) / sd            # usado como entrada; se invierte para evaluar
""")

md("""
### 2.6 Covariables de calendario conocidas a futuro
`hora`, `día de la semana` y `mes` se codifican con **seno/coseno** (preservan la ciclicidad, sin explosión de
dimensiones del one-hot) y `feriado` como binaria. Todas se conocen a futuro, por lo que pueden entrar al decoder.
""")
code("""
def calendar_features(idx):
    f = pd.DataFrame(index=idx)
    for name, val, per in [("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / per)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / per)
    f["holiday"] = idx.normalize().isin(pd.to_datetime(list(pt_holidays.keys()))).astype(float)
    return f.astype("float32")

cal = calendar_features(panel.index)
cal.head(3)
""")

# ---------------------------------------------------------------- 3. Dataset
md("""
## 3. Dataset de ventanas deslizantes
Cada muestra: `X` = (L, features) con el consumo normalizado + calendario, `y` = (H,) siguiente 24 h.
El origen `t` es la primera hora a pronosticar; la entrada usa solo `[t-L, t-1]`.
Se agregan también las covariables de calendario de `[t, t+H-1]` (`x_fut`), conocidas a futuro.
""")
code("""
class WindowDataset(Dataset):
    \"\"\"Ventanas (X: L x F, y: H) sobre todas las series. Origenes t en [start, end] (inclusive) con paso `stride`.
    El target de cada ventana cae en [t, t+H-1]; se asigna a un split segun su ORIGEN t, y el target
    nunca excede el limite superior del split.\"\"\"
    def __init__(self, values, cal, index, start, end, L=168, H=24, stride=1):
        self.v = torch.tensor(values.values, dtype=torch.float32)       # (T, N)
        self.c = torch.tensor(cal.values, dtype=torch.float32)          # (T, C)
        self.L, self.H = L, H
        i0 = max(index.get_indexer([pd.Timestamp(start)])[0], L)
        i1 = index.get_indexer([pd.Timestamp(end)])[0] - H + 1          # target dentro del split
        self.origins = np.arange(i0, i1 + 1, stride)
        self.N = self.v.shape[1]
    def __len__(self): return len(self.origins) * self.N
    def __getitem__(self, k):
        t, n = self.origins[k // self.N], k % self.N
        past = self.v[t - self.L:t, n:n + 1]                            # (L, 1)
        x = torch.cat([past, self.c[t - self.L:t]], dim=1)              # (L, 1+C)
        x_fut = self.c[t:t + self.H]                                    # (H, C) calendario futuro
        y = self.v[t:t + self.H, n]                                     # (H,)
        return x, x_fut, y, n

idx = panel.index
ds_train = WindowDataset(panel_z, cal, idx, START, TRAIN_END, L, H, STRIDE)
ds_val   = WindowDataset(panel_z, cal, idx, "2014-07-01 00:00", VAL_END, L, H, STRIDE)
ds_test  = WindowDataset(panel_z, cal, idx, "2014-10-01 00:00", END, L, H, STRIDE)
print(f"Ventanas (L={L}, H={H}, stride={STRIDE}) -> train: {len(ds_train):,} | val: {len(ds_val):,} | test: {len(ds_test):,}")
x, xf, y, n = ds_train[0]; print(x.shape, xf.shape, y.shape)
""")
md("""
> Nota: los primeros `L` pasos del bloque de validación y prueba pueden usar historia del bloque anterior como *entrada*
> (es información pasada legítima); lo que no puede cruzarse es el **target**. Aquí el `start` se limita para que
> el origen del primer target quede dentro del split, y el historial de entrada puede venir del split previo.
""")

# ---------------------------------------------------------------- 4. Fugas
md("""
## 4. Auditoría de fugas
Cada entrada debe poder calcularse en el instante de emitir el pronóstico (00:00 con datos hasta 23:00).
""")
code("""
# (1) Split aleatorio de ventanas -> se usa split CRONOLOGICO por origen/target
assert panel.index[ds_train.origins[-1] + H - 1] <= pd.Timestamp(TRAIN_END)
assert panel.index[ds_val.origins[0]] > pd.Timestamp(TRAIN_END) - pd.Timedelta(hours=H)  # sin solape de targets con train
assert panel.index[ds_val.origins[-1] + H - 1] <= pd.Timestamp(VAL_END)
assert panel.index[ds_test.origins[0]] > pd.Timestamp(VAL_END) - pd.Timedelta(hours=H)
print("(1) Splits cronologicos, targets sin solape entre train/val/test: OK")

# (2) Escalado con toda la serie -> mu y sd salen solo de TRAIN
assert np.allclose(mu.values, panel.loc[:TRAIN_END].mean().values)
mu_full = panel.mean()
print("(2) mu usa solo train; diferencia vs. media global (promedio relativo): "
      f"{((mu - mu_full).abs() / mu_full).mean():.3%}")

# (3) Early stopping con la prueba -> la seleccion usa solo val; test se evalua una vez por modelo
print("(3) Early stopping/seleccion de hiperparametros: solo con ds_val (ver secciones de entrenamiento)")

# (4) Covariables mal alineadas: el calendario se calcula del indice y x_fut se toma en [t, t+H-1]
k = 100
t = ds_train.origins[k // ds_train.N]
x, xf, y, n = ds_train[k]
assert torch.allclose(x[:, 0], torch.tensor(panel_z.values[t - L:t, n], dtype=torch.float32))   # pasado
assert torch.allclose(y, torch.tensor(panel_z.values[t:t + H, n], dtype=torch.float32))         # futuro
assert torch.allclose(xf, torch.tensor(cal.values[t:t + H]))      # calendario futuro (conocido)
assert panel_z.index[t - 1] < panel_z.index[t]                    # la entrada termina antes del origen
print("(4) Entrada = [t-L, t-1]; target = [t, t+H-1]; solo calendario (determinista) se usa a futuro: OK")
""")

# ---------------------------------------------------------------- 5. Backtest
md("""
## 5. Protocolo de validación y métricas
`backtest(model)` se implementa **una sola vez**: origen móvil diario en prueba (92 orígenes, a las 00:00 de cada día
del 2014-10-01 al 2014-12-31); cada modelo expone `predict(history_df, origin) -> array (n_clients, H)` en kWh.
""")
code("""
TEST_ORIGINS = pd.date_range("2014-10-01 00:00", "2014-12-31 00:00", freq="1D")
assert len(TEST_ORIGINS) == 92

def backtest(predict_fn, panel_kwh=panel, origins=TEST_ORIGINS, H=H):
    \"\"\"predict_fn(history: DataFrame hasta origin-1h, origin) -> ndarray (n_clientes, H) en kWh.
    Devuelve DataFrame largo (unique_id, ds, origin, h, y, y_hat).\"\"\"
    rows = []
    for o in origins:
        hist = panel_kwh.loc[:o - pd.Timedelta(hours=1)]            # solo datos hasta las 23:00
        pred = np.asarray(predict_fn(hist, o))                      # (N, H)
        ds = pd.date_range(o, periods=H, freq="1h")
        real = panel_kwh.loc[ds].values.T                           # (N, H)
        for i, c in enumerate(panel_kwh.columns):
            rows.append(pd.DataFrame({"unique_id": c, "ds": ds, "origin": o, "h": np.arange(1, H + 1),
                                      "y": real[i], "y_hat": pred[i]}))
    return pd.concat(rows, ignore_index=True)

def metrics(df, train_panel=None, m=24):
    e = df.y_hat - df.y
    out = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
           "WAPE": e.abs().sum() / df.y.abs().sum()}
    nz = df.y.abs() > 1e-6
    out["MAPE"] = (e[nz].abs() / df.y[nz].abs()).mean()             # explota con consumo ~ 0
    if train_panel is not None:                                      # MASE: escala = MAE naive estacional en train
        scale = train_panel.diff(m).abs().mean()                     # por cliente
        mae_c = e.abs().groupby(df.unique_id).mean()
        out["MASE"] = (mae_c / scale.reindex(mae_c.index)).mean()
    return pd.Series(out)

# Ejemplo con seasonal naive 24 h (verifica el pipeline)
def snaive24(hist, origin): return hist.iloc[-24:].values.T
bt = backtest(snaive24)
metrics(bt, panel.loc[:TRAIN_END])
""")
code("""
# Error por paso del horizonte, por hora del dia y semana 22-28 dic
def breakdown(bt):
    ae = (bt.y_hat - bt.y).abs()
    return (ae.groupby(bt.h).mean(), ae.groupby(bt.ds.dt.hour).mean(),
            bt[(bt.ds >= "2014-12-22") & (bt.ds < "2014-12-29")])
by_h, by_hour, xmas = breakdown(bt)
print("MAE semana 22-28 dic:", (xmas.y_hat - xmas.y).abs().mean())
""")

exec(open(Path(__file__).parent / "_models_cells.py", encoding="utf-8").read())

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).parent / "Lab9_Series_de_Tiempo.ipynb"
nbf.write(nb, out)
print("escrito", out)
