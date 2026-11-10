"""Position-error experiment: plan with estimated terminal positions, execute with the true channel."""
import sys, json, numpy as np
from multiprocessing import Pool
from sim import *

def execute(q, tau_plan, w_est, w_true, p):
    Rest = rates_for(q, w_est, p); Rtr = rates_for(q, w_true, p)
    d = p.delta
    b = tau_plan * d * Rest                      # planned Mbit
    need = b / Rtr                               # seconds needed on the true channel
    scale = np.minimum(1.0, d / np.maximum(need.sum(0), 1e-12))
    b = b * scale[None, :]
    O = b.sum(1)
    tx = p.Pu * (b / Rtr).sum()
    return tx + p.cl * ((p.L - O) ** 3).sum() + p.cs * O.sum() ** 3

def task(args):
    idx, rad, chan, pw = args
    p = Prm(channel=chan, power=pw)
    w = placements(idx + 1)[idx][:p.K]
    rng = np.random.default_rng(10_000 + idx)
    ang = rng.uniform(0, 2 * np.pi, p.K); rr = rad * np.sqrt(rng.uniform(0, 1, p.K))
    w_est = w + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
    z = np.zeros((p.N, 2))
    qc = circle_init(w_est, p, 1)
    res = [bcd(w_est, p, z0, "tangent") for z0 in (z, qc)]
    r = min(res, key=lambda x: x["J"])
    prop_est = execute(r["q"], r["tau"], w_est, w, p)
    hov = evaluate(z, w_est, p)
    hov_est = execute(z, hov["tau"], w_est, w, p)
    # reference: planning with the true positions
    rt = min([bcd(w, p, z0, "tangent") for z0 in (z, circle_init(w, p, 1))], key=lambda x: x["J"])
    return dict(idx=idx, prop_est=float(prop_est), prop_true=float(rt["J"]), hover_est=float(hov_est))

if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2]
    allres = []
    with Pool(2) as pool:
        for chan, pw in (("fs", "par"), ("los", "par")):
            for rad in (0, 10, 25, 50, 100, 200):
                res = pool.map(task, [(i, rad, chan, pw) for i in range(n)], chunksize=1)
                allres.append(dict(channel=chan, power=pw, rad=rad, res=res))
                print(chan, rad, "done", flush=True)
                json.dump(allres, open(out, "w"))
