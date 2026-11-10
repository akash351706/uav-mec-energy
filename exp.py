import sys, json, time, numpy as np
from multiprocessing import Pool
from sim import *

METHODS = ("local", "hover", "circle", "proposed", "ratelb")

def task(args):
    idx, pd, K = args
    p = Prm(**pd); p.K = K
    w = placements(idx+1)[idx][:K]
    t = time.time()
    try:
        o = run_all(w, p, METHODS)
    except Exception as e:
        return dict(idx=idx, err=str(e))
    r = dict(idx=idx, bud=float(budget(p)))
    for m, v in o.items():
        r[m] = dict(J=float(v["J"]), fly=float(v.get("fly", 0.0)), off=float(v.get("off", 0.0)),
                    tx=float(v.get("tx", 0.0)), loc=float(v.get("loc", 0.0)), uav=float(v.get("uav", 0.0)),
                    iters=int(v.get("iters", 0)), time=float(v.get("time", 0.0)))
        if m == "proposed" and "hist" in v:
            r[m]["hist"] = [float(h) for h in v["hist"]]
    r["wall"] = time.time() - t
    return r

def run(name, pd, K, n, pool, start=0):
    t = time.time()
    res = pool.map(task, [(i, pd, K) for i in range(start, n)], chunksize=1)
    ok = [r for r in res if "err" not in r]
    print(f"[{name}] K={K} n={len(ok)}/{n} {time.time()-t:.0f}s", flush=True)
    return dict(name=name, pd=pd, K=K, res=ok)

if __name__ == "__main__":
    suite = sys.argv[1]
    out = sys.argv[2]
    n = int(sys.argv[3])
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    jobs = json.load(open(suite))
    allres = []
    with Pool(2) as pool:
        for j in jobs:
            allres.append(run(j["name"], j["pd"], j["K"], j.get("n", n), pool, start))
            json.dump(allres, open(out, "w"))
