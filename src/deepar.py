"""DeepAR (implementacion propia en PyTorch): LSTM autorregresivo con cabeza Student-t, calendario como covariables y
z-score por serie. Entrena maximizando la log-verosimilitud con teacher forcing y pronostica por muestreo ancestral.
(NeuralForecast.DeepAR no fue viable en la GPU de 6 GB: error de memoria en validacion y tiempos excesivos.)
Uso: python src/deepar.py [id ...]   (ids de CFGS)"""
import sys; sys.path.insert(0, "src")
from lab9 import *
import lab9
import torch.nn.functional as F
from torch.distributions import StudentT


class DeepAR(Core):
    def __init__(self, L, C, H=H, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(1 + C, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.)
        self.head = nn.Linear(hidden, 3)                     # mu, sigma, nu
        self.H = H

    def _dist(self, o):
        mu, s, nu = self.head(o).unbind(-1)
        return mu, F.softplus(s) + 1e-3, 2.0 + F.softplus(nu)

    def train_loss(self, past, fut, y, crit):
        full = torch.cat([past[..., 0], y], 1)               # (B, L+H) teacher forcing
        cal = torch.cat([past[..., 1:], fut], 1)
        inp = torch.cat([full[:, :-1, None], cal[:, 1:]], -1)
        o, _ = self.rnn(inp)
        mu, sg, nu = self._dist(o)
        nll = -StudentT(nu, mu, sg).log_prob(full[:, 1:])
        return nll[:, 24:].mean()                            # descarta 24 pasos de calentamiento

    def _encode(self, past, fut):
        x = past[..., 0]
        cal = torch.cat([past[..., 1:], fut], 1)
        L = x.shape[1]
        inp = torch.cat([x[:, :, None], cal[:, 1:L + 1]], -1)
        o, st = self.rnn(inp)
        return o[:, -1], st, cal, L

    @torch.no_grad()
    def forward(self, past, fut):
        """Pronostico puntual (trayectoria de la media, sin muestreo): para validacion."""
        o, st, cal, L = self._encode(past, fut)
        mu, _, _ = self._dist(o); out = [mu]
        for k in range(1, self.H):
            o, st = self.rnn(torch.cat([mu[:, None, None], cal[:, L + k:L + k + 1]], -1), st)
            mu = self._dist(o[:, -1])[0]; out.append(mu)
        return torch.stack(out, 1)

    @torch.no_grad()
    def sample(self, past, fut, S=100):
        """Muestreo ancestral: (B, S, H) en espacio normalizado."""
        o, st, cal, L = self._encode(past, fut)
        B = o.shape[0]
        rep = lambda t: t.repeat_interleave(S, dim=1) if t.dim() == 3 and t.shape[0] != B else t
        st = tuple(t.repeat_interleave(S, dim=1) for t in st)       # (layers, B*S, hidden)
        o = o.repeat_interleave(S, dim=0)
        cal = cal.repeat_interleave(S, dim=0)
        mu, sg, nu = self._dist(o); z = StudentT(nu, mu, sg).sample(); out = [z]
        for k in range(1, self.H):
            oo, st = self.rnn(torch.cat([z[:, None, None], cal[:, L + k:L + k + 1]], -1), st)
            mu, sg, nu = self._dist(oo[:, -1]); z = StudentT(nu, mu, sg).sample(); out.append(z)
        return torch.stack(out, 1).view(B, S, self.H)


lab9.CORES["deepar"] = DeepAR
CFGS = {"deepar_base": dict(model="deepar", L=168, hidden=64, layers=2, dropout=0.1),
        "deepar_lr3e-3": dict(model="deepar", L=168, hidden=64, layers=2, dropout=0.1, lr=3e-3),
        "deepar_h128_lr3e-3": dict(model="deepar", L=168, hidden=128, layers=2, dropout=0.1, lr=3e-3)}


def make_sample_predict_fn(net, panel, L, S=100, seed=0):
    def predict(hist, origin):
        net.eval(); t = len(hist); assert panel.index[t] == origin
        torch.manual_seed(seed + t)
        x = torch.tensor(hist.values[-L:], dtype=torch.float32, device=panel.dev)
        z = (x - panel.mu) / panel.sd
        past = torch.cat([z.T.unsqueeze(-1), panel.C[t - L:t].unsqueeze(0).expand(panel.N, -1, -1)], -1)
        fut = panel.C[t:t + H].unsqueeze(0).expand(panel.N, -1, -1)
        sm = net.core.sample(past, fut, S) * panel.sd[:, None, None] + panel.mu[:, None, None]    # (N,S,H) kWh
        q = torch.quantile(sm.clamp_min(0), torch.tensor([0.1, 0.5, 0.9], device=panel.dev), dim=1)   # (3,N,H)
        return {"y_hat": q[1].cpu().numpy(), "p10": q[0].cpu().numpy(), "p90": q[2].cpu().numpy()}
    return predict


if __name__ == "__main__":
    P = Panel(load_panel(100)); df = P.df; scale = mase_scale(df.iloc[:P.pos_train_end])
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    ids = sys.argv[1:] or list(CFGS)
    for rid in ids:
        runs = load_json("runs_deepar.json")
        if rid in runs: continue
        print("=====", rid, CFGS[rid], flush=True)
        net, info = fit(P, CFGS[rid], verbose=True)
        torch.save(net.state_dict(), os.path.join(ROOT, "checkpoints", rid + ".pt"))
        runs = load_json("runs_deepar.json"); runs[rid] = info; save_json(runs, "runs_deepar.json")
        print(f"-> {rid}: val_mase={info['best_val_mase']:.4f} ep={info['best_epoch']} t={info['train_time']:.0f}s", flush=True)
    runs = load_json("runs_deepar.json")
    best = min(runs, key=lambda r: runs[r]["best_val_mase"]); cfg = runs[best]["cfg"]
    net = Net(DeepAR(cfg["L"], P.C.shape[1], hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"]), cfg["norm"]).to(P.dev)
    net.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", best + ".pt"), map_location=P.dev)); net.eval()
    torch.cuda.reset_peak_memory_stats()
    bt = backtest(make_sample_predict_fn(net, P, cfg["L"]), df)
    rep = full_report(bt, scale, "DeepAR")
    rep.update(family="Probabilistico", L=cfg["L"], strategy="autorregresiva (muestreo, 100 trayectorias)", covariates=True,
               params_total=runs[best]["params_total"], params_trainable=runs[best]["params_trainable"], train_time=runs[best]["train_time"],
               infer_time=bt.attrs["infer_time"], hardware=f"GPU {torch.cuda.get_device_name(0)}", gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
               best_cfg=best, val_mase=runs[best]["best_val_mase"], impl="propia (PyTorch)")
    save_preds(bt, "DeepAR")
    fin = load_json("final_deepar.json"); fin["DeepAR"] = rep; save_json(fin, "final_deepar.json")
    print("DeepAR", {k: round(float(v), 4) for k, v in rep.items() if k in ("MAE", "RMSE", "WAPE", "MASE", "MASE_xmas", "coverage_p10_p90", "pinball")}, flush=True)
