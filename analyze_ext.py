"""Analysis of the two extensions: (1) propulsion energy in the objective, (2) heterogeneous task sizes."""
import json, sys, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "font.family": "serif"})

def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

MUS = (0.0, 0.001, 0.01, 0.1, 1.0)
S = {}
R = json.load(open("res_joint.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, ch, ttl in zip(axs, ("fs", "los"), ("(a) free-space channel", "(b) line-of-sight channel")):
    pts = []
    for mu in MUS:
        rs = [r for r in R if r["ch"] == ch and r["mu"] == mu]
        c = np.mean([r["joint"]["J"] for r in rs]); f = np.mean([r["joint"]["fly"] for r in rs]) / 1e3
        pts.append((mu, c, f))
        S[f"{ch}_{mu}"] = dict(n=len(rs),
            comm=ci([r["joint"]["J"] for r in rs]), fly=ci([r["joint"]["fly"] for r in rs]),
            hover_comm=ci([r["hover"]["J"] for r in rs]), hover_fly=ci([r["hover"]["fly"] for r in rs]),
            loop_comm=ci([r["loop"]["J"] for r in rs]), loop_fly=ci([r["loop"]["fly"] for r in rs]),
            co_comm=ci([r["comm_only"]["J"] for r in rs]), co_fly=ci([r["comm_only"]["fly"] for r in rs]),
            tot_joint=ci([r["joint"]["Jtot"] for r in rs]), tot_hover=ci([r["hover"]["Jtot"] for r in rs]),
            tot_co=ci([r["comm_only"]["Jtot"] for r in rs]), tot_loop=ci([r["loop"]["Jtot"] for r in rs]),
            sv_hover=ci([100 * (1 - r["joint"]["Jtot"] / r["hover"]["Jtot"]) for r in rs]),
            sv_co=ci([100 * (1 - r["joint"]["Jtot"] / r["comm_only"]["Jtot"]) for r in rs]),
            sv_loop=ci([100 * (1 - r["joint"]["Jtot"] / r["loop"]["Jtot"]) for r in rs]),
            wins_loop=int(sum(r["joint"]["Jtot"] < r["loop"]["Jtot"] - 1e-6 for r in rs)))
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "o-", color="#1f4e79", label="proposed, joint objective")
    ax.annotate("$\\mu$=0", pts[0][1:], textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.annotate("$\\mu$=0.001", pts[1][1:], textcoords="offset points", xytext=(8, 6), fontsize=7)
    ax.set_ylim(7.6, 10.75)
    h = S[f"{ch}_0.0"]
    ax.plot(h["hover_comm"][0], h["hover_fly"][0] / 1e3, "s", color="#c0392b", label="hover at depot")
    ax.plot(h["loop_comm"][0], h["loop_fly"][0] / 1e3, "^", color="#2e8b57", label="fixed loop at 10.2 m/s")
    ax.set_xlabel("Communication + computing energy (J)"); ax.set_title(ttl, fontsize=9)
axs[0].set_ylabel("Propulsion energy (kJ)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig_pareto.png", dpi=250)

if len(sys.argv) > 1:
    H = json.load(open("res_het.json"))
    cfgs = sorted({(r["ch"], r["pw"], r["K"]) for r in H})
    S["het"] = {}
    for ch, pw, K in cfgs:
        key = f"{ch}-{pw}-K{K}"
        S["het"][key] = {}
        for het in (False, True):
            rs = [r for r in H if (r["ch"], r["pw"], r["K"], r["het"]) == (ch, pw, K, het)]
            if not rs: continue
            S["het"][key][str(het)] = dict(n=len(rs),
                vs_local=ci([100 * (1 - r["proposed"]["J"] / r["local"]["J"]) for r in rs]),
                vs_hover=ci([100 * (1 - r["proposed"]["J"] / r["hover"]["J"]) for r in rs]),
                vs_circle=ci([100 * (1 - r["proposed"]["J"] / r["circle"]["J"]) for r in rs]),
                vs_ratelb=ci([100 * (1 - r["proposed"]["J"] / r["ratelb"]["J"]) for r in rs]),
                E=ci([r["proposed"]["J"] for r in rs]), off=ci([r["proposed"]["off"] for r in rs]),
                t=ci([r["proposed"]["time"] for r in rs]))
json.dump(S, open("summary_ext.json", "w"), indent=1)
print(json.dumps(S, indent=1)[:200])
