import sys, json, time, numpy as np
from multiprocessing import Pool
from sim import *

def hetL(idx, K, lo=10.0, hi=30.0):
    return np.random.default_rng(777 + idx).uniform(lo, hi, K)

def jtask(a):
    idx, ch, mu = a
    p = Prm(K=4, channel=ch, power="full", eps=1.0, mu=mu)
    w = placements(idx + 1)[idx][:4]
    try:
        o = run_joint(w, p)
    except Exception as e:
        return dict(idx=idx, ch=ch, mu=mu, err=str(e))
    r = dict(idx=idx, ch=ch, mu=mu)
    for m, v in o.items():
        r[m] = dict(Jtot=float(v["Jtot"]), J=float(v["J"]), fly=float(v["fly"]), tx=float(v["tx"]),
                    off=float(v["off"]), time=float(v.get("time", 0)))
    return r

def htask(a):
    idx, ch, pw, K, het = a
    p = Prm(K=K, channel=ch, power=pw, eps=0.0)
    if het:
        p.L = hetL(idx, K)
    w = placements(idx + 1)[idx][:K]
    try:
        o = run_all(w, p, ("local", "hover", "circle", "proposed", "ratelb"))
    except Exception as e:
        return dict(idx=idx, err=str(e))
    r = dict(idx=idx, ch=ch, pw=pw, K=K, het=het, Lk=(list(map(float, p.L)) if het else None))
    for m, v in o.items():
        r[m] = dict(J=float(v["J"]), fly=float(v.get("fly", 0)), off=float(v.get("off", 0)), time=float(v.get("time", 0)))
    return r

if __name__ == "__main__":
    which = sys.argv[1]
    n = int(sys.argv[2])
    out = []
    with Pool(2) as pool:
        if which == "joint":
            for ch in ("fs", "los"):
                for mu in (0.0, 0.001, 0.01, 0.1, 1.0):
                    t = time.time()
                    res = pool.map(jtask, [(i, ch, mu) for i in range(n)], chunksize=1)
                    out += [r for r in res if "err" not in r]
                    print(ch, mu, len(res), f"{time.time()-t:.0f}s", flush=True)
                    json.dump(out, open("res_joint.json", "w"))
        else:
            cfgs = [(ch, pw, 4) for ch in ("fs", "los") for pw in ("par", "full")] + [("fs", "par", 8), ("los", "full", 8)]
            for ch, pw, K in cfgs:
                for het in (False, True):
                    t = time.time()
                    nn = n if K == 4 else max(20, n // 2)
                    res = pool.map(htask, [(i, ch, pw, K, het) for i in range(nn)], chunksize=1)
                    out += [r for r in res if "err" not in r]
                    print(ch, pw, K, het, len(res), f"{time.time()-t:.0f}s", flush=True)
                    json.dump(out, open("res_het.json", "w"))
