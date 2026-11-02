"""Datos crudos -> panel horario limpio (kWh) cacheado en data/panel_all.pkl. Misma logica que el notebook."""
import numpy as np, pandas as pd
START, END = "2012-01-01 00:00", "2014-12-31 23:00"
raw = pd.read_csv("data/LD2011_2014.txt", sep=";", decimal=",", index_col=0, parse_dates=True)
print(raw.shape)
hourly = (raw * 0.25).resample("1h").sum(min_count=4).loc[START:END]
print("filas horarias", len(hourly), "esperadas", len(pd.date_range(START, END, freq="1h")))
first_nonzero = hourly.apply(lambda s: s.ne(0).idxmax() if s.ne(0).any() else pd.NaT)
complete = (first_nonzero <= pd.Timestamp(START) + pd.Timedelta(hours=1)) & hourly.notna().all()
panel = hourly.loc[:, complete].copy()
print("clientes completos:", panel.shape[1], "obs/serie:", len(panel), "panel:", panel.size)
def last_sunday(y, m):
    d = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)
for y in (2012, 2013, 2014):
    for m in (3, 10):
        d = last_sunday(y, m); print(d.date(), np.round(panel.loc[d:d+pd.Timedelta(hours=4)].sum(axis=1).values))

# DST: marzo -> hora 01:00 casi vacia (NaN + interpolacion); octubre -> hora 01:00 duplicada (promedio de +-7 dias)
for y in (2012, 2013, 2014):
    sp, fa = last_sunday(y, 3), last_sunday(y, 10)
    panel.loc[sp + pd.Timedelta(hours=1)] = np.nan
    h = fa + pd.Timedelta(hours=1)
    panel.loc[h] = (panel.loc[h - pd.Timedelta(days=7)].values + panel.loc[h + pd.Timedelta(days=7)].values) / 2
panel = panel.interpolate(limit_direction="both")
assert panel.isna().sum().sum() == 0
panel.to_pickle("data/panel_all.pkl")
rng = np.random.default_rng(42)
cols = sorted(rng.choice(panel.columns, size=100, replace=False))
panel[cols].to_pickle("data/panel_100.pkl")
print("guardado; subconjunto", len(cols))
