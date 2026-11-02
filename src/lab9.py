"""Modulo comun del Lab 9: panel, calendario, Dataset de ventanas, backtest/metricas, modelos propios y trainer."""
import json, time, os, random
import numpy as np, pandas as pd, torch, torch.nn as nn, holidays
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, H = 42, 24
TRAIN_END, VAL_END, END = "2014-06-30 23:00", "2014-09-30 23:00", "2014-12-31 23:00"
TEST_ORIGINS = pd.date_range("2014-10-01 00:00", "2014-12-31 00:00", freq="1D")
assert len(TEST_ORIGINS) == 92
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ panel y calendario
def load_panel(n=100):
    return pd.read_pickle(os.path.join(ROOT, "data", f"panel_{n}.pkl"))


PT_HOL = holidays.country_holidays("PT", years=range(2011, 2016))


def calendar_features(idx):
    """hora, dia de la semana y mes en seno/coseno + feriado (binaria). Todas deterministas -> conocidas a futuro."""
    f = pd.DataFrame(index=idx)
    for name, val, per in [("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / per)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / per)
    f["holiday"] = idx.normalize().isin(pd.to_datetime(list(PT_HOL.keys()))).astype(float)
    return f.astype("float32")


class Panel:
    """Panel en GPU: kwh (T,N), z-score por serie con estadisticos de TRAIN, calendario (T,C) y escala MASE."""
    def __init__(self, df, dev=DEV):
        self.df, self.index, self.cols = df, df.index, list(df.columns)
        self.T, self.N = df.shape
        self.pos_train_end = self.index.get_loc(pd.Timestamp(TRAIN_END)) + 1      # exclusivo
        self.pos_val_end = self.index.get_loc(pd.Timestamp(VAL_END)) + 1
        tr = df.iloc[:self.pos_train_end]
        self.mu_np, self.sd_np = tr.mean().values, tr.std().replace(0, 1.0).values
        self.mase_scale_np = tr.diff(24).abs().mean().values                         # MAE naive estacional (train)
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.kwh, self.mu, self.sd = t(df.values), t(self.mu_np), t(self.sd_np)
        self.Z = (self.kwh - self.mu) / self.sd
        self.cal_df = calendar_features(df.index)
        self.C = t(self.cal_df.values)
        self.scale = t(self.mase_scale_np)
        self.dev = dev

    def origins(self, split, L, H=H, months=None, train_end_pos=None):
        """Posiciones t (primera hora a pronosticar). Entrada [t-L,t-1], target [t,t+H-1] dentro del split."""
        te = train_end_pos or self.pos_train_end
        if split == "train":
            lo = L
            if months is not None:
                start = self.index[te - 1] + pd.Timedelta(hours=1) - pd.DateOffset(months=months)
                lo = max(L, self.index.get_loc(start) + L if start in self.index else L)
            return np.arange(lo, te - H + 1)
        if split == "val":
            return np.arange(self.pos_train_end, self.pos_val_end - H + 1)
        if split == "test":
            return np.arange(self.pos_val_end, self.T - H + 1)
        raise ValueError(split)


class WindowDataset(Dataset):
    """Ventanas deslizantes sobre todas las series: X (L, 1+C), x_fut (H, C), y (H,). Origenes con paso `stride`.
    Ademas de __getitem__ ofrece `batch(k)` vectorizado en GPU, equivalente a apilar __getitem__."""
    def __init__(self, panel, origins, L, H=H, stride=1):
        self.p, self.L, self.H = panel, L, H
        self.origins = np.asarray(origins)[::stride]
        self.ar_L = torch.arange(-L, 0, device=panel.dev)
        self.ar_H = torch.arange(0, H, device=panel.dev)
        self.org_t = torch.tensor(self.origins, device=panel.dev)

    def __len__(self): return len(self.origins) * self.p.N

    def batch(self, k):
        k = torch.as_tensor(k, device=self.p.dev)
        t, n = self.org_t[k // self.p.N], k % self.p.N
        ti, tf = t[:, None] + self.ar_L, t[:, None] + self.ar_H
        past = torch.cat([self.p.Z[ti, n[:, None]].unsqueeze(-1), self.p.C[ti]], dim=-1)   # (B, L, 1+C)
        return past, self.p.C[tf], self.p.Z[tf, n[:, None]], n

    def __getitem__(self, k):
        x, xf, y, n = self.batch(torch.tensor([k]))
        return x[0], xf[0], y[0], int(n[0])


# ------------------------------------------------------------------ backtest y metricas
def backtest(predict_fn, panel_kwh, origins=TEST_ORIGINS, H=H):
    """Protocolo unico. predict_fn(history_df hasta origin-1h, origin) -> (N,H) kWh, o dict con 'y_hat' y cuantiles
    {'p10':(N,H),...}. Devuelve DataFrame largo (unique_id, ds, origin, h, y, y_hat[, p10, p90...])."""
    cols, N = list(panel_kwh.columns), panel_kwh.shape[1]
    outs, t0 = [], time.time()
    for o in origins:
        hist = panel_kwh.loc[:o - pd.Timedelta(hours=1)]
        assert hist.index[-1] == o - pd.Timedelta(hours=1)
        r = predict_fn(hist, o)
        outs.append(r if isinstance(r, dict) else {"y_hat": r})
    infer_time = time.time() - t0
    n_o = len(origins)
    d = {"unique_id": np.tile(np.array(cols), n_o * H),
         "origin": np.repeat(np.array(origins), N * H)}
    # orden de filas: origen, h, cliente
    ds = np.concatenate([np.repeat(pd.date_range(o, periods=H, freq="1h").values, N) for o in origins])
    d["ds"] = ds
    d["h"] = np.tile(np.repeat(np.arange(1, H + 1), N), n_o)
    real = np.concatenate([panel_kwh.loc[pd.date_range(o, periods=H, freq="1h")].values.reshape(-1) for o in origins])
    d["y"] = real
    for key in outs[0]:
        d[key] = np.concatenate([np.asarray(o[key]).T.reshape(-1) for o in outs])    # (N,H)->(H,N)->flat
    df = pd.DataFrame(d)
    df.attrs["infer_time"] = infer_time
    return df


def mase_scale(train_df): return train_df.diff(24).abs().mean()


def metrics(df, scale):
    """MAE, RMSE, WAPE, MAPE (solo y>0), MASE (media por cliente de MAE/escala naive estacional de train)."""
    e = df.y_hat - df.y
    ae = e.abs()
    nz = df.y.abs() > 1e-6
    mae_c = ae.groupby(df.unique_id).mean()
    return pd.Series({"MAE": ae.mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "WAPE": ae.sum() / df.y.abs().sum(),
                      "MAPE": (ae[nz] / df.y[nz].abs()).mean(), "MASE": (mae_c / scale.reindex(mae_c.index)).mean()})


def pinball(y, q_pred, q):
    d = y - q_pred
    return np.maximum(q * d, (q - 1) * d)


def full_report(df, scale, name=None):
    """Metricas globales + error por h, por hora del dia y semana 22-28 dic."""
    ae = (df.y_hat - df.y).abs()
    xm = df[(df.ds >= "2014-12-22") & (df.ds < "2014-12-29")]
    rep = metrics(df, scale).to_dict()
    rep["MASE_xmas"] = metrics(xm, scale)["MASE"]
    rep["MAE_xmas"] = (xm.y_hat - xm.y).abs().mean()
    rep["mae_by_h"] = ae.groupby(df.h).mean().tolist()
    rep["mae_by_hour"] = ae.groupby(df.ds.dt.hour).mean().tolist()
    rep["mae_by_client"] = ae.groupby(df.unique_id).mean().to_dict()
    rep["rmse_by_client"] = ((df.y_hat - df.y) ** 2).groupby(df.unique_id).mean().pow(.5).to_dict()
    if "p10" in df and "p90" in df:
        rep["coverage_p10_p90"] = float(((df.y >= df.p10) & (df.y <= df.p90)).mean())
        rep["pinball"] = float(np.mean([pinball(df.y, df[c], q).mean() for c, q in
                                        [("p10", .1), ("y_hat", .5), ("p90", .9)]]))
    if name: rep["name"] = name
    return rep


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------------------ modelos propios (nucleos)
# Cada nucleo: forward(past (B,L,1+C), fut (B,H,C)) -> (B,H) en espacio normalizado; train_loss(past,fut,y,crit).
class Core(nn.Module):
    def train_loss(self, past, fut, y, crit): return crit(self(past, fut), y)


class MLP(Core):
    def __init__(self, L, C, H=H, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        dims, mods = L + H * C, []
        for _ in range(layers):
            mods += [nn.Linear(dims, hidden), nn.ReLU(), nn.Dropout(dropout)]; dims = hidden
        self.net = nn.Sequential(*mods, nn.Linear(dims, H))

    def forward(self, past, fut): return self.net(torch.cat([past[..., 0], fut.flatten(1)], dim=1))


class DLinear(Core):
    """Descomposicion tendencia (media movil, kernel 25) + resto; una capa lineal L->H por componente."""
    def __init__(self, L, C, H=H, kernel=25):
        super().__init__()
        self.k = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1)
        self.lin_s, self.lin_t = nn.Linear(L, H), nn.Linear(L, H)

    def forward(self, past, fut):
        x = past[..., 0]
        pad = torch.cat([x[:, :1].repeat(1, (self.k - 1) // 2), x, x[:, -1:].repeat(1, (self.k - 1) // 2)], 1)
        trend = self.pool(pad.unsqueeze(1)).squeeze(1)
        return self.lin_s(x - trend) + self.lin_t(trend)


class NLinear(Core):
    """Resta el ultimo valor de la ventana, aplica una capa lineal y lo suma de vuelta."""
    def __init__(self, L, C, H=H):
        super().__init__(); self.lin = nn.Linear(L, H)

    def forward(self, past, fut):
        x = past[..., 0]; last = x[:, -1:]
        return self.lin(x - last) + last


class LSTMMulti(Core):
    """Multi-salida: LSTM sobre la ventana, ultimo estado oculto (+calendario futuro) -> capa densa con H salidas."""
    def __init__(self, L, C, H=H, hidden=64, layers=1, dropout=0.1, cell="lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(1 + C, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + H * C, H))

    def forward(self, past, fut):
        out, _ = self.rnn(past)
        return self.head(torch.cat([out[:, -1], fut.flatten(1)], 1))


class LSTMAuto(Core):
    """Autorregresivo: paso de entrada [z_s, cal_{s+1}] -> z_{s+1}. Entrena con teacher forcing sobre toda la
    secuencia (L+H); en inferencia realimenta su propia prediccion durante H pasos."""
    def __init__(self, L, C, H=H, hidden=64, layers=1, dropout=0.1, cell="lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(1 + C, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.)
        self.out = nn.Linear(hidden, 1)
        self.H = H

    def forward(self, past, fut):
        x = past[..., 0]
        cal = torch.cat([past[..., 1:], fut], 1)                       # (B, L+H, C)
        L = x.shape[1]
        inp = torch.cat([x[:, :, None], cal[:, 1:L + 1]], -1)           # pasos s=0..L-1 con cal_{s+1}
        o, st = self.rnn(inp)
        z = self.out(o[:, -1])                                          # z_L
        preds = [z]
        for k in range(1, self.H):
            o, st = self.rnn(torch.cat([z[:, None], cal[:, L + k:L + k + 1]], -1), st)
            z = self.out(o[:, -1]); preds.append(z)
        return torch.cat(preds, 1)

    def train_loss(self, past, fut, y, crit):
        full = torch.cat([past[..., 0], y], 1)                          # (B, L+H) teacher forcing
        cal = torch.cat([past[..., 1:], fut], 1)
        inp = torch.cat([full[:, :-1, None], cal[:, 1:]], -1)
        o, _ = self.rnn(inp)
        pred = self.out(o).squeeze(-1)                                  # predice full[:,1:]
        return crit(pred[:, 24:], full[:, 1:][:, 24:])                   # descarta 24 pasos de calentamiento


class Seq2Seq(Core):
    """Encoder LSTM sobre el pasado; decoder LSTM paso a paso con [y_prev, calendario futuro]. Teacher forcing."""
    def __init__(self, L, C, H=H, hidden=64, layers=1, dropout=0.1, tf=0.5):
        super().__init__()
        d = dropout if layers > 1 else 0.
        self.enc = nn.LSTM(1 + C, hidden, layers, batch_first=True, dropout=d)
        self.dec = nn.LSTM(1 + C, hidden, layers, batch_first=True, dropout=d)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.H, self.tf = H, tf

    def run(self, past, fut, y=None):
        _, st = self.enc(past)
        prev, preds = past[:, -1:, 0:1], []
        for k in range(self.H):
            o, st = self.dec(torch.cat([prev, fut[:, k:k + 1]], -1), st)
            p = self.out(o)                                             # (B,1,1)
            preds.append(p[:, 0])
            use_tf = y is not None and self.training and random.random() < self.tf
            prev = y[:, k:k + 1, None] if use_tf else p
        return torch.cat(preds, 1)

    def forward(self, past, fut): return self.run(past, fut)
    def train_loss(self, past, fut, y, crit): return crit(self.run(past, fut, y), y)


CORES = {"mlp": MLP, "dlinear": DLinear, "nlinear": NLinear, "lstm": LSTMMulti, "gru": lambda *a, **k: LSTMMulti(*a, cell="gru", **k),
         "lstm_auto": LSTMAuto, "seq2seq": Seq2Seq}


class Net(nn.Module):
    """Envuelve un nucleo y maneja la normalizacion: 'z' (ya aplicada al panel) o 'window' (RevIN por ventana)."""
    def __init__(self, core, norm="z"):
        super().__init__(); self.core, self.norm = core, norm

    def _prep(self, past):
        if self.norm != "window":
            return past, 0., 1.
        x = past[..., 0]
        m, s = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-3
        past = torch.cat([((x - m) / s)[..., None], past[..., 1:]], -1)
        return past, m, s

    def forward(self, past, fut):
        p, m, s = self._prep(past)
        return self.core(p, fut) * s + m

    def loss(self, past, fut, y, crit):
        p, m, s = self._prep(past)
        return self.core.train_loss(p, fut, (y - m) / s, crit)


def n_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ trainer
DEFAULT = dict(model="dlinear", L=168, months=None, norm="z", hidden=64, layers=1, dropout=0.1, lr=1e-3,
               bs=256, epochs=20, spe=800, patience=4, seed=SEED, wd=1e-4, loss="l1")


@torch.no_grad()
def val_metrics(net, panel, ds, bs=1024):
    """MAE (kWh) y MASE en validacion: errores absolutos medios por cliente."""
    net.eval()
    tot = torch.zeros(panel.N, device=panel.dev); cnt = torch.zeros(panel.N, device=panel.dev)
    zl = 0.
    for i in range(0, len(ds), bs):
        past, fut, y, n = ds.batch(np.arange(i, min(i + bs, len(ds))))
        p = net(past, fut)
        e = (p - y).abs().mean(1)
        tot.index_add_(0, n, e); cnt.index_add_(0, n, torch.ones_like(e))
        zl += e.sum().item()
    mae_z_c = tot / cnt
    mae_kwh_c = mae_z_c * panel.sd
    return dict(val_loss=zl / len(ds), val_mae=mae_kwh_c.mean().item(), val_mase=(mae_kwh_c / panel.scale).mean().item())


def fit(panel, cfg, train_end_pos=None, verbose=True, fixed_epochs=None):
    """Entrena un modelo global sobre todas las series. Early stopping y seleccion solo con validacion.
    Si fixed_epochs, entrena exactamente ese numero de epochs sin validacion (reentrenamiento mensual)."""
    cfg = {**DEFAULT, **cfg}
    seed_all(cfg["seed"])
    L, C = cfg["L"], panel.C.shape[1]
    kw = {k: cfg[k] for k in ("hidden", "layers", "dropout") if k in cfg}
    m = cfg["model"]
    if m in ("dlinear", "nlinear"): kw = {}
    if m == "mlp": kw = dict(hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"])
    net = Net(CORES[m](L, C, **kw), cfg["norm"]).to(panel.dev)
    ds_tr = WindowDataset(panel, panel.origins("train", L, months=cfg["months"], train_end_pos=train_end_pos), L)
    ds_va = WindowDataset(panel, panel.origins("val", L), L, stride=4)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.L1Loss() if cfg["loss"] == "l1" else nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs or cfg["epochs"], eta_min=cfg["lr"] * 0.05)
    tot, trn = n_params(net)
    hist, best, best_state, bad = [], 1e9, None, 0
    if panel.dev == "cuda": torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    n_ep = fixed_epochs or cfg["epochs"]
    for ep in range(1, n_ep + 1):
        net.train(); t0 = time.time(); tl = 0.
        for _ in range(cfg["spe"]):
            k = torch.randint(0, len(ds_tr), (cfg["bs"],), device=panel.dev)
            past, fut, y, _n = ds_tr.batch(k)
            loss = net.loss(past, fut, y, crit)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item()
        sched.step()
        rec = dict(epoch=ep, train_loss=tl / cfg["spe"], epoch_time=time.time() - t0)
        if fixed_epochs is None:
            rec.update(val_metrics(net, panel, ds_va))
            if rec["val_mase"] < best - 1e-5:
                best, bad = rec["val_mase"], 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        if verbose: print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
        if fixed_epochs is None and bad >= cfg["patience"]: break
    if best_state is not None: net.load_state_dict(best_state)
    info = dict(cfg=cfg, history=hist, params_total=tot, params_trainable=trn, train_time=time.time() - t_start,
                best_val_mase=best, best_epoch=int(np.argmin([h["val_mase"] for h in hist]) + 1) if fixed_epochs is None else n_ep,
                gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if panel.dev == "cuda" else 0.,
                n_train_windows=len(ds_tr), n_val_windows=len(ds_va))
    return net, info


def make_predict_fn(net, panel, L):
    """Adaptador al protocolo backtest: usa solo `hist` (hasta origin-1h) + calendario determinista."""
    def predict(hist, origin):
        net.eval()
        t = len(hist)
        assert panel.index[t] == origin
        with torch.no_grad():
            x = torch.tensor(hist.values[-L:], dtype=torch.float32, device=panel.dev)            # (L,N) kWh
            z = (x - panel.mu) / panel.sd
            past = torch.cat([z.T.unsqueeze(-1), panel.C[t - L:t].unsqueeze(0).expand(panel.N, -1, -1)], -1)
            fut = panel.C[t:t + H].unsqueeze(0).expand(panel.N, -1, -1)
            out = net(past, fut) * panel.sd[:, None] + panel.mu[:, None]
        return out.clamp_min(0).cpu().numpy()
    return predict


def infer_with_timing(net, panel, L):
    fn = make_predict_fn(net, panel, L)
    return fn


# ------------------------------------------------------------------ utilidades para modelos de libreria
def cache_predict_fn(cache):
    """cache: {origin Timestamp: {'y_hat': (N,H), 'p10':..., 'p90':...}} -> predict_fn compatible con backtest().
    Los modelos de libreria predicen todos los origenes en una sola llamada (sin ver datos posteriores a cada corte)."""
    def predict(hist, origin): return cache[pd.Timestamp(origin)]
    return predict


def long_to_cache(df, cols, ycols, ds_col="ds", cutoff_col="cutoff", H=H):
    """DataFrame largo (unique_id, ds, cutoff, <ycols>) -> cache {origin: {key: (N,H)}} ordenado como `cols`."""
    out = {}
    df = df.copy()
    df["origin"] = df[cutoff_col] + pd.Timedelta(hours=1)
    for o, g in df.groupby("origin"):
        g = g.sort_values(["unique_id", ds_col])
        d = {}
        for key, c in ycols.items():
            a = g.pivot(index="unique_id", columns=ds_col, values=c).reindex(cols)
            d[key] = np.clip(a.values, 0, None)
        out[pd.Timestamp(o)] = d
    return out


def save_preds(bt, name):
    d = os.path.join(RES, "preds"); os.makedirs(d, exist_ok=True)
    keep = [c for c in ("unique_id", "ds", "origin", "h", "y_hat", "p10", "p90") if c in bt]
    bt[keep].astype({"y_hat": "float32"}).to_parquet(os.path.join(d, f"{name}.parquet"), index=False)


def load_preds(name):
    return pd.read_parquet(os.path.join(RES, "preds", f"{name}.parquet"))
