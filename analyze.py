import json, os, numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.linewidth": 0.6, "legend.fontsize": 7.5,
                     "legend.framealpha": 0.9, "figure.dpi": 150, "mathtext.fontset": "stix"})
os.makedirs("figs3", exist_ok=True)

def load(f):
    return json.load(open(f)) if os.path.exists(f) else []

DATA = {}
for f in ("res_default.json", "res_c1.json", "res_ext.json", "res_ext2.json", "res_big.json"):
    for c in load(f):
        _p = Prm(**c["pd"]); _p.K = c["K"]
        for r in c["res"]:
            r.setdefault("bud", float(budget(_p)))
        DATA[c["name"]] = c
# default-suite entries keep their names "C1 fs+par" ... ; alias
ALIAS = {"C1": "C1 fs+par", "C2": "C2 fs+full", "C3": "C3 los+par", "C4": "C4 los+full"}

def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

def feas(r, m):
    if m == "local":
        return True
    return r[m].get("fly", 0) <= r["bud"] * (1 + 1e-7) if "bud" in r else True

def stat(c, m, ref=None):
    """mean energy, or saving (%) of m relative to ref, with 95% CI; only placements where both feasible."""
    res = c["res"]
    rows = [r for r in res if feas(r, m) and (ref is None or feas(r, ref))]
    if ref is None:
        return ci([r[m]["J"] for r in rows]) + (len(rows),)
    return ci([100 * (1 - r[m]["J"] / r[ref]["J"]) for r in rows]) + (len(rows),)

NUM = {}
COL = dict(local="#c0392b", hover="#2471a3", circle="#27ae60", ratelb="#e67e22", proposed="#000000")
LAB = dict(local="Local computing", hover="Hover at depot", circle="Fixed circle",
           ratelb="Rate-bound SCA", proposed="Proposed")
MK = dict(local="x", hover="s", circle="^", ratelb="v", proposed="o")
LS = dict(local="-", hover="--", circle="-.", ratelb=":", proposed="-")

def sweep_names(prefix, key, vals, fmt="{}"):
    out = []
    for v in vals:
        out.append(f"{prefix} {key}={fmt.format(v)}")
    return out

def curve(names, xs, methods=("local", "hover", "circle", "ratelb", "proposed")):
    out = {}
    for m in methods:
        mu, hw, n = [], [], []
        for nm in names:
            if nm not in DATA:
                mu.append(np.nan); hw.append(np.nan); n.append(0); continue
            a, b, k = stat(DATA[nm], m)
            mu.append(a); hw.append(b); n.append(k)
        out[m] = (np.array(mu), np.array(hw), n)
    return out

def plot_curves(ax, xs, cv, methods=("local", "hover", "circle", "ratelb", "proposed")):
    for m in methods:
        mu, hw, _ = cv[m]
        ok = ~np.isnan(mu)
        ax.errorbar(np.array(xs)[ok], mu[ok], yerr=hw[ok], color=COL[m], marker=MK[m], ms=3.5, lw=1.1,
                    ls=LS[m], capsize=1.5, elinewidth=0.6, label=LAB[m])

# ------------------------------------------------------------------ Fig: model illustration
def fig_model():
    p = Prm(power="full")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    V = np.linspace(0, 20, 300)
    a = ax[0]
    a.plot(V, prop_power(V ** 2, p)[0], color="k", label="Full rotary-wing model [12]")
    a.plot(V, prop_power(V ** 2, Prm(power="par"))[0], color="#c0392b", ls="--", label="Parasitic term only")
    a.axhline(prop_power(0, p)[0], color="#2471a3", ls=":", lw=1, label="Hover power")
    Vm = V[np.argmin(prop_power(V ** 2, p)[0])]
    a.plot([Vm], [prop_power(Vm ** 2, p)[0]], "o", color="k", ms=3)
    a.set_ylim(0, 225)
    a.annotate(f"min {prop_power(Vm**2,p)[0]:.0f} W at {Vm:.1f} m/s", (Vm, prop_power(Vm**2, p)[0]),
               (6.0, 198), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
    a.set_xlabel("Speed $V$ (m/s)"); a.set_ylabel("Propulsion power (W)"); a.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    a.set_title("(a)", fontsize=9)
    a = ax[1]
    d = np.linspace(0, 600, 600)
    for ch, st, nm in (("fs", "-", "Free-space"), ("los", "--", "LoS probability")):
        pp = Prm(channel=ch)
        a.plot(d, ftime(d ** 2, pp), st, color="k" if ch == "fs" else "#c0392b", label=nm)
    pp = Prm(channel="los"); f = ftime(d ** 2, pp)
    d2 = np.gradient(np.gradient(f, d ** 2), d ** 2)
    bad = d[d2 > 0]
    a.axvspan(bad.min(), bad.max(), color="#f5b7b1", alpha=0.5, label="LoS: $f$ not concave")
    a.set_xlabel("Horizontal distance (m)"); a.set_ylabel("Time per Mbit $f$ (s/Mbit)"); a.legend(loc="upper left")
    a.set_title("(b)", fontsize=9)
    fig.tight_layout(); fig.savefig("figs3/fig_model.png", dpi=300); plt.close(fig)
    NUM["nonconcave_range_m"] = [float(bad.min()), float(bad.max())]
    NUM["Pmin"] = [float(Vm), float(prop_power(Vm ** 2, p)[0]), float(prop_power(0, p)[0])]

# ------------------------------------------------------------------ Fig: sweeps
def fig_L():
    Ls = [10, 15, 20, 25, 30]
    names = [("C1 fs+par" if L == 20 else f"C1 L={L}") for L in Ls]
    cv = curve(names, Ls)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    plot_curves(ax, Ls, cv); ax.set_xlabel("Task size per terminal $L$ (Mbit)"); ax.set_ylabel("Total energy (J)")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig("figs3/fig_L.png", dpi=300); plt.close(fig)
    NUM["L"] = {L: {m: stat(DATA[n], m) for m in ("local", "hover", "circle", "ratelb", "proposed")} |
                {"prop_vs_local": stat(DATA[n], "proposed", "local"), "prop_vs_hover": stat(DATA[n], "proposed", "hover"),
                 "prop_vs_rl": stat(DATA[n], "proposed", "ratelb")}
                for L, n in zip(Ls, names) if n in DATA}

def fig_T():
    Ts = [40, 60, 80, 100]
    names = [("C1 fs+par" if T == 60 else f"C1 T={T}") for T in Ts]
    cv = curve(names, Ts)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    plot_curves(ax, Ts, cv); ax.set_xlabel("Deadline $T$ (s)"); ax.set_ylabel("Total energy (J)")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig("figs3/fig_T.png", dpi=300); plt.close(fig)
    NUM["T"] = {T: {"prop_vs_local": stat(DATA[n], "proposed", "local"), "prop_vs_hover": stat(DATA[n], "proposed", "hover"),
                    "prop": stat(DATA[n], "proposed"), "local": stat(DATA[n], "local"),
                    "off": float(np.mean([r["proposed"]["off"] for r in DATA[n]["res"]]))}
                for T, n in zip(Ts, names) if n in DATA}

def fig_K():
    Ks = [2, 4, 8, 12, 16]
    def nm(prefix, K):
        if K == 4:
            return ALIAS[prefix]
        return f"{prefix} K={K}"
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for a, prefix, ttl in ((ax[0], "C1", "(a) Free-space, parasitic power"), (ax[1], "C4", "(b) LoS, full power model")):
        names = [nm(prefix, K) for K in Ks]
        ok = [K for K, n in zip(Ks, names) if n in DATA]
        cv = curve([nm(prefix, K) for K in ok], ok)
        plot_curves(a, ok, cv, methods=("local", "hover", "circle", "ratelb", "proposed"))
        a.set_xlabel("Number of terminals $K$"); a.set_ylabel("Total energy (J)"); a.set_title(ttl, fontsize=8.5)
        a.legend(loc="upper left")
    fig.tight_layout(); fig.savefig("figs3/fig_K.png", dpi=300); plt.close(fig)
    for prefix in ("C1", "C2", "C3", "C4"):
        NUM[f"K_{prefix}"] = {}
        for K in Ks:
            n = nm(prefix, K)
            if n in DATA:
                NUM[f"K_{prefix}"][K] = dict(n=len(DATA[n]["res"]), prop=stat(DATA[n], "proposed"),
                                            vs_local=stat(DATA[n], "proposed", "local"),
                                            vs_hover=stat(DATA[n], "proposed", "hover"),
                                            vs_rl=stat(DATA[n], "proposed", "ratelb"),
                                            time=float(np.mean([r["proposed"]["time"] for r in DATA[n]["res"]])),
                                            iters=float(np.mean([r["proposed"]["iters"] for r in DATA[n]["res"]])))

def tables():
    # default table
    NUM["default"] = {}
    for c, name in ALIAS.items():
        d = DATA.get(name)
        if not d:
            continue
        res = d["res"]
        e = {m: stat(d, m) for m in ("local", "hover", "circle", "ratelb", "proposed")}
        e["prop_vs_local"] = stat(d, "proposed", "local")
        e["prop_vs_hover"] = stat(d, "proposed", "hover")
        e["prop_vs_circle"] = stat(d, "proposed", "circle")
        e["prop_vs_rl"] = stat(d, "proposed", "ratelb")
        e["circle_vs_hover"] = stat(d, "circle", "hover")
        e["rl_vs_hover"] = stat(d, "ratelb", "hover")
        e["fly_prop"] = float(np.mean([r["proposed"]["fly"] for r in res]))
        e["fly_hover"] = float(np.mean([r["hover"]["fly"] for r in res]))
        e["t_prop"] = float(np.mean([r["proposed"]["time"] for r in res]))
        e["t_rl"] = float(np.mean([r["ratelb"]["time"] for r in res]))
        e["iters"] = float(np.mean([r["proposed"]["iters"] for r in res]))
        e["off"] = float(np.mean([r["proposed"]["off"] for r in res]))
        e["tx"] = float(np.mean([r["proposed"]["tx"] for r in res]))
        e["loc"] = float(np.mean([r["proposed"]["loc"] for r in res]))
        e["uav"] = float(np.mean([r["proposed"]["uav"] for r in res]))
        e["n"] = len(res)
        # win rate proposed <= ratelb
        e["win_vs_rl"] = float(np.mean([r["proposed"]["J"] <= r["ratelb"]["J"] + 1e-9 for r in res]))
        e["max_rel_diff_rl"] = float(np.max([abs(r["proposed"]["J"] - r["ratelb"]["J"]) / r["ratelb"]["J"] for r in res]) * 100)
        # monotone check
        e["monotone"] = bool(all(all(np.diff(r["proposed"]["hist"]) <= 1e-9) for r in res if "hist" in r["proposed"]))
        NUM["default"][c] = e
    # Emax table (C1 + C3)
    NUM["Emax"] = {}
    for pref in ("C1", "C3"):
        NUM["Emax"][pref] = {}
        for E in (5, 10, 25, 50, 100, 150):
            n = ALIAS[pref] if E == 50 else f"{pref} Emax={E}"
            if n in DATA:
                d = DATA[n]
                NUM["Emax"][pref][E] = {m: stat(d, m) for m in ("hover", "circle", "ratelb", "proposed")} | \
                    {"vs_hover": stat(d, "proposed", "hover"), "vs_circle": stat(d, "proposed", "circle")}
    # N
    NUM["N"] = {}
    for N in (10, 20, 30):
        n = ALIAS["C1"] if N == 20 else f"C1 N={N}"
        if n in DATA:
            NUM["N"][N] = dict(prop=stat(DATA[n], "proposed"), vs_hover=stat(DATA[n], "proposed", "hover"))
    # eps sweep
    NUM["eps"] = {}
    for pref in ("C2", "C4"):
        NUM["eps"][pref] = {}
        for eps in (-0.2, -0.1, 0.0, 0.1):
            n = ALIAS[pref] if eps == 0.0 else f"{pref} eps={eps}"
            if n not in DATA:
                continue
            d = DATA[n]
            if eps == 0.0:   # compare on the same 60 placements as the other budgets
                d = dict(d); d["res"] = d["res"][:60]
            res = d["res"]
            bud = np.mean([r["bud"] for r in res])
            nh = sum(feas(r, "hover") for r in res)
            ent = dict(n=len(res), bud=float(bud), hover_feasible=nh / len(res),
                       prop=stat(d, "proposed"), circle=stat(d, "circle"), ratelb=stat(d, "ratelb"),
                       fly_prop=float(np.mean([r["proposed"]["fly"] for r in res])),
                       vs_local=stat(d, "proposed", "local"),
                       vs_circle=stat(d, "proposed", "circle"),
                       vs_rl=stat(d, "proposed", "ratelb"),
                       prop_feasible=float(np.mean([feas(r, "proposed") for r in res])))
            if nh > 1:
                ent["vs_hover"] = stat(d, "proposed", "hover")
            NUM["eps"][pref][eps] = ent
    # L / K for other configs
    for pref in ("C2", "C3", "C4"):
        NUM[f"L_{pref}"] = {}
        for L in (10, 20, 30):
            n = ALIAS[pref] if L == 20 else f"{pref} L={L}"
            if n in DATA:
                NUM[f"L_{pref}"][L] = dict(vs_local=stat(DATA[n], "proposed", "local"), vs_hover=stat(DATA[n], "proposed", "hover"),
                                           vs_rl=stat(DATA[n], "proposed", "ratelb"), prop=stat(DATA[n], "proposed"), local=stat(DATA[n], "local"))

def fig_pos():
    P = load("res_pos.json")
    if not P:
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.1))
    NUM["pos"] = {}
    for a, (ch, ttl) in zip(ax, (("fs", "(a) Free-space channel"), ("los", "(b) LoS-probability channel"))):
        rows = [c for c in P if c["channel"] == ch]
        rad = [c["rad"] for c in rows]
        for key, lab, col, mk, ls in (("prop_true", "Proposed, true positions", "k", "o", "-"),
                                      ("prop_est", "Proposed, estimated positions", "#c0392b", "s", "--"),
                                      ("hover_est", "Hover, estimated positions", "#2471a3", "^", ":")):
            mu = []; hw = []
            for c in rows:
                a_, b_ = ci([r[key] for r in c["res"]]); mu.append(a_); hw.append(b_)
            a.errorbar(rad, mu, yerr=hw, color=col, marker=mk, ms=3.5, ls=ls, lw=1.1, capsize=1.5, elinewidth=0.6, label=lab)
        a.set_xlabel("Position-error radius (m)"); a.set_ylabel("Total energy (J)"); a.set_title(ttl, fontsize=8.5)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=6.8, ncol=1)
        NUM["pos"][ch] = {}
        for c in rows:
            inc = [100 * (r["prop_est"] / r["prop_true"] - 1) for r in c["res"]]
            vsh = [100 * (1 - r["prop_est"] / r["hover_est"]) for r in c["res"]]
            NUM["pos"][ch][c["rad"]] = dict(inc=ci(inc), vs_hover=ci(vsh), n=len(c["res"]),
                                            prop_est=ci([r["prop_est"] for r in c["res"]])[0])
    fig.tight_layout(); fig.savefig("figs3/fig_pos.png", dpi=300); plt.close(fig)

def fig_traj():
    W = placements(1)[0][:4]
    cfgs = [("fs", "par", "(a) Free-space, parasitic"), ("fs", "full", "(b) Free-space, full power"),
            ("los", "par", "(c) LoS, parasitic"), ("los", "full", "(d) LoS, full power")]
    fig, ax = plt.subplots(1, 4, figsize=(6.9, 2.0))
    z = np.zeros((20, 2))
    for a, (ch, pw, ttl) in zip(ax, cfgs):
        p = Prm(channel=ch, power=pw)
        rs = [bcd(W, p, q0, "tangent") for q0 in (z, circle_init(W, p, 1), circle_init(W, p, -1))]
        r = min(rs, key=lambda x: x["J"])
        q = path(r["q"])
        a.plot(q[:, 0], q[:, 1], "k.-", ms=2.5, lw=0.8, label="UAV")
        a.plot(W[:, 0], W[:, 1], "r*", ms=7, label="Terminals")
        a.plot([0], [0], "bs", ms=4, label="Depot")
        a.set_title(ttl, fontsize=7.5); a.set_aspect("equal"); a.set_xlim(-320, 320); a.set_ylim(-320, 320)
        a.tick_params(labelsize=6)
        if a is ax[0]:
            a.legend(loc="lower left", fontsize=5.5)
    fig.tight_layout(); fig.savefig("figs3/fig_traj.png", dpi=300); plt.close(fig)
    # convergence curves from default runs
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for c, col in zip(("C1", "C2", "C3", "C4"), ("k", "#c0392b", "#2471a3", "#27ae60")):
        d = DATA.get(ALIAS[c])
        if not d:
            continue
        for r in d["res"][:8]:
            h = r["proposed"]["hist"]
            ax.plot(range(len(h)), np.array(h) / h[0], color=col, lw=0.6, alpha=0.6)
        ax.plot([], [], color=col, label="M"+c[1])
    ax.set_xlabel("BCD iteration"); ax.set_ylabel("Objective / initial objective"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig("figs3/fig_conv.png", dpi=300); plt.close(fig)

if __name__ == "__main__":
    fig_model()
    for f in (fig_L, fig_T, fig_K, fig_pos, fig_traj):
        try:
            f()
        except Exception as e:
            print("skip", f.__name__, repr(e))
    tables()
    json.dump(NUM, open("nums.json", "w"), indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print("ok", list(NUM.keys()))
