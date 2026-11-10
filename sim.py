"""Simulator for deadline-constrained energy minimisation in UAV-aided MEC.

Units: bits in Mbit, rates in Mbit/s, times in s, r = 1/R in s/Mbit, energy in J.
"""
import time
import numpy as np
from dataclasses import dataclass, replace
from scipy.optimize import minimize


@dataclass
class Prm:
    K: int = 4
    N: int = 20
    T: float = 60.0
    H: float = 100.0
    V: float = 20.0
    B: float = 0.1e6
    Pu: float = 1.0
    beta0_dB: float = -60.0
    sig2_dBm: float = -110.0
    L: float = 20.0           # Mbit per terminal
    C: float = 2000.0         # cycles / bit
    ku: float = 1e-27
    ks: float = 4e-29
    d0: float = 0.6
    rho: float = 1.225
    s: float = 0.05
    A: float = 0.503
    Emax: float = 50.0        # J, parasitic-model cap
    eps: float = 0.0          # full model: budget = (1+eps) * hover energy
    channel: str = "fs"       # 'fs' or 'los'
    power: str = "par"        # 'par' (parasitic only) or 'full'
    P0: float = 79.86
    Pi: float = 88.63
    Utip: float = 120.0
    v0: float = 4.03
    a_los: float = 9.61
    b_los: float = 0.16
    etaL: float = 1.0
    etaN: float = 20.0

    @property
    def delta(self):
        return self.T / self.N

    @property
    def gamma(self):
        return self.Pu * 10 ** (self.beta0_dB / 10) / 10 ** ((self.sig2_dBm - 30) / 10)

    @property
    def cd(self):
        return 0.5 * self.d0 * self.rho * self.s * self.A

    @property
    def cl(self):
        return self.ku * (self.C * 1e6) ** 3 / self.T ** 2

    @property
    def cs(self):
        return self.ks * (self.C * 1e6) ** 3 / self.T ** 2


# ---------------------------------------------------------------- channel
def rate(x, p):
    """Rate in Mbit/s for squared horizontal distance x (array)."""
    x = np.asarray(x, float)
    u = p.H ** 2 + x
    if p.channel == "fs":
        g = 1.0 / u
    else:
        theta = np.degrees(np.arctan2(p.H, np.sqrt(x)))
        pl = 1.0 / (1.0 + p.a_los * np.exp(-p.b_los * (theta - p.a_los)))
        loss_db = pl * p.etaL + (1 - pl) * p.etaN
        g = 10 ** (-loss_db / 10) / u
    return p.B / 1e6 * np.log2(1.0 + p.gamma * g)


def drate(x, p):
    x = np.asarray(x, float)
    if p.channel == "fs":
        u = p.H ** 2 + x
        return -(p.B / 1e6) / np.log(2) * p.gamma / (u * (u + p.gamma))
    h = 1e-5 * (p.H ** 2 + x) + 1e-6
    return (rate(x + h, p) - rate(np.maximum(x - h, 0.0), p)) / (x + h - np.maximum(x - h, 0.0))


def ftime(x, p):          # s / Mbit
    return 1.0 / rate(x, p)


def dftime(x, p):
    R = rate(x, p)
    return -drate(x, p) / R ** 2


# ---------------------------------------------------------------- propulsion
def prop_power(s2, p):
    """Power (W) as function of s2 = V^2, and derivative dP/ds2."""
    s2 = np.asarray(s2, float)
    sr = np.sqrt(s2)
    par = p.cd * s2 * sr
    dpar = 1.5 * p.cd * sr
    if p.power == "par":
        return par, dpar
    q = np.sqrt(1.0 + s2 ** 2 / (4 * p.v0 ** 4))
    h = 1.0 / (q + s2 / (2 * p.v0 ** 2))                # = q - s2/(2 v0^2)
    dq = (s2 / (4 * p.v0 ** 4)) / q
    dh = -h ** 2 * (dq + 1.0 / (2 * p.v0 ** 2))
    blade = p.P0 * (1 + 3 * s2 / p.Utip ** 2)
    ind = p.Pi * np.sqrt(h)
    P = blade + ind + par
    dP = 3 * p.P0 / p.Utip ** 2 + p.Pi * dh / (2 * np.sqrt(h)) + dpar
    return P, dP


def path(q):
    z = np.zeros((1, 2))
    return np.vstack([z, q, z])


def fly_energy(q, p):
    d = np.diff(path(q), axis=0)
    s2 = (d ** 2).sum(1) / p.delta ** 2
    return p.delta * prop_power(s2, p)[0].sum()


def budget(p):
    if p.power == "par":
        return p.Emax
    return (1 + p.eps) * prop_power(0.0, p)[0] * (p.N + 1) * p.delta


# ---------------------------------------------------------------- Step A
def energy_terms(tau, Rm, p):
    O = p.delta * (Rm * tau).sum(1)
    tx = p.Pu * p.delta * tau.sum()
    loc = p.cl * ((p.L - O) ** 3).sum()
    uav = p.cs * O.sum() ** 3
    return tx, loc, uav


def stepA(Rm, p, tau0=None):
    K, N = Rm.shape
    d = p.delta

    def obj(z):
        t = z.reshape(K, N)
        O = d * (Rm * t).sum(1)
        return p.Pu * d * z.sum() + p.cl * ((p.L - O) ** 3).sum() + p.cs * O.sum() ** 3

    def grad(z):
        t = z.reshape(K, N)
        O = d * (Rm * t).sum(1)
        g = p.Pu * d - 3 * p.cl * (p.L - O)[:, None] ** 2 * d * Rm + 3 * p.cs * O.sum() ** 2 * d * Rm
        return g.ravel()

    # constraints: slot time (N), per-terminal volume (K)
    A1 = np.zeros((N, K * N))
    for n in range(N):
        A1[n, n::N] = -1.0
    def cons(z):
        t = z.reshape(K, N)
        O = d * (Rm * t).sum(1)
        return np.concatenate([1.0 - t.sum(0), p.L - O])

    def cjac(z):
        J2 = -d * Rm
        A2 = np.zeros((K, K * N))
        for k in range(K):
            A2[k, k * N:(k + 1) * N] = J2[k]
        return np.vstack([A1, A2])

    best = None
    starts = [tau0.ravel() if tau0 is not None else None, np.zeros(K * N)]
    for z0 in starts:
        if z0 is None:
            continue
        res = minimize(obj, z0, jac=grad, method="SLSQP", bounds=[(0, 1)] * (K * N),
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options={"maxiter": 300, "ftol": 1e-11})
        t = np.clip(res.x.reshape(K, N), 0, 1)
        # repair feasibility
        cs_ = t.sum(0)
        t = t / np.maximum(cs_, 1.0)
        O = d * (Rm * t).sum(1)
        over = O > p.L
        if over.any():
            t[over] *= (p.L / O[over])[:, None]
        val = obj(t.ravel())
        if best is None or val < best[0] - 1e-12:
            best = (val, t)
        if tau0 is not None and res.success:
            break
    return best[1], best[0]


# ---------------------------------------------------------------- Step B
def stepB(q, tau, w, p, mode="tangent", radius=None):
    K, N = tau.shape
    d = p.delta
    x0 = ((q[None, :, :] - w[:, None, :]) ** 2).sum(2)          # K,N
    Rm0 = rate(x0, p)
    b = tau * d * Rm0                                            # Mbit
    act = b > 1e-9
    if mode == "tangent":
        f0 = ftime(x0, p)
        a = dftime(x0, p)
        phi = lambda x: f0 + a * (x - x0)
        dphi = lambda x: a
        xcap = None
    else:  # first-order rate lower bound (SCA in the style of refs [4],[14])
        dR = drate(x0, p)
        Rlb = lambda x: Rm0 + dR * (x - x0)
        phi = lambda x: 1.0 / Rlb(x)
        dphi = lambda x: -dR / Rlb(x) ** 2
        xcap = x0 + 0.9 * Rm0 / (-dR)
    bud = budget(p)
    Vd2 = (p.V * d) ** 2

    def X(z):
        qq = z.reshape(N, 2)
        return ((qq[None, :, :] - w[:, None, :]) ** 2).sum(2), qq

    def obj(z):
        x, _ = X(z)
        return p.Pu * (b * phi(x) * act).sum()

    def grad(z):
        x, qq = X(z)
        coef = b * dphi(x) * act                                   # K,N
        diff = qq[None, :, :] - w[:, None, :]
        g = 2 * p.Pu * (coef[:, :, None] * diff).sum(0)
        return g.ravel()

    def cons(z):
        x, qq = X(z)
        tm = d - (b * phi(x) * act).sum(0)
        P = path(qq)
        dd = np.diff(P, axis=0)
        sp = Vd2 - (dd ** 2).sum(1)
        s2 = (dd ** 2).sum(1) / d ** 2
        en = (bud - d * prop_power(s2, p)[0].sum()) / bud
        out = [tm, sp, [en]]
        if radius is not None:
            out.append(radius ** 2 - ((qq - q) ** 2).sum(1))
        if xcap is not None:
            out.append(((xcap - x) / 1e4 * act).ravel()[act.ravel()])
        return np.concatenate([np.atleast_1d(o) for o in out])

    def cjac(z):
        x, qq = X(z)
        diff = qq[None, :, :] - w[:, None, :]
        coef = b * dphi(x) * act
        rows = []
        # time rows (N)
        Jt = np.zeros((N, 2 * N))
        gtm = -2 * (coef[:, :, None] * diff).sum(0)               # N,2
        for n in range(N):
            Jt[n, 2 * n:2 * n + 2] = gtm[n]
        rows.append(Jt)
        # speed rows (N+1)
        P = path(qq)
        dd = np.diff(P, axis=0)
        Js = np.zeros((N + 1, 2 * N))
        for n in range(N + 1):
            if n >= 1:
                Js[n, 2 * (n - 1):2 * (n - 1) + 2] += 2 * dd[n]   # d(-|dd|^2)/dq_n  (q index n-1 -> q_n as end)
            if n <= N - 1:
                Js[n, 2 * n:2 * n + 2] += -2 * dd[n] * 0
        # exact speed jac: segment n connects P[n] -> P[n+1]; q_k = P[k], variable index k-1
        Js[:] = 0
        for n in range(N + 1):
            if n + 1 <= N:           # P[n+1] is a variable (index n)
                Js[n, 2 * n:2 * n + 2] += -2 * dd[n]
            if n >= 1:               # P[n] is a variable (index n-1)
                Js[n, 2 * (n - 1):2 * (n - 1) + 2] += 2 * dd[n]
        rows.append(Js)
        # energy row
        s2 = (dd ** 2).sum(1) / d ** 2
        dP = prop_power(s2, p)[1]
        gseg = 2 * dP[:, None] * dd / d                           # dE/d(dd_n), n=0..N  (E = d*sum P)
        Je = np.zeros((1, 2 * N))
        for n in range(N + 1):
            if n + 1 <= N:
                Je[0, 2 * n:2 * n + 2] += -gseg[n]
            if n >= 1:
                Je[0, 2 * (n - 1):2 * (n - 1) + 2] += gseg[n]
        rows.append(Je / bud)
        if radius is not None:
            Jr = np.zeros((N, 2 * N))
            for n in range(N):
                Jr[n, 2 * n:2 * n + 2] = -2 * (qq[n] - q[n])
            rows.append(Jr)
        if xcap is not None:
            idx = np.flatnonzero(act.ravel())
            Jx = np.zeros((len(idx), 2 * N))
            for j, ii in enumerate(idx):
                k, n = divmod(ii, N)
                Jx[j, 2 * n:2 * n + 2] = -2 * diff[k, n] / 1e4
            rows.append(Jx)
        return np.vstack(rows)

    res = minimize(obj, q.ravel(), jac=grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                   options={"maxiter": 200, "ftol": 1e-10})
    qn = res.x.reshape(N, 2)

    def feasible(qq):
        dd = np.diff(path(qq), axis=0)
        return (not ((dd ** 2).sum(1) > Vd2 * (1 + 1e-9)).any()) and fly_energy(qq, p) <= bud * (1 + 1e-9)

    # line-search repair towards the (feasible) current point if the solver ended slightly infeasible
    for alpha in (1.0, 0.9, 0.75, 0.5, 0.25):
        cand = q + alpha * (qn - q)
        if feasible(cand):
            return cand
    return None


# ---------------------------------------------------------------- evaluation
def rates_for(q, w, p):
    x = ((q[None, :, :] - w[:, None, :]) ** 2).sum(2)
    return rate(x, p)


def evaluate(q, w, p, tau0=None):
    Rm = rates_for(q, w, p)
    tau, J = stepA(Rm, p, tau0)
    tx, loc, uav = energy_terms(tau, Rm, p)
    O = p.delta * (Rm * tau).sum(1)
    return dict(J=tx + loc + uav, tx=tx, loc=loc, uav=uav, off=float(O.sum() / (p.K * p.L)),
                fly=fly_energy(q, p), tau=tau)


def local_energy(p):
    return p.K * p.cl * p.L ** 3


def bcd(w, p, q0, mode="tangent", maxit=15, tol=1e-5):
    """Block coordinate descent with safeguarded trajectory step.

    Step B is tried with a trust-region back-off (None -> 90 -> 45 -> 20 m); the first candidate
    that lowers the true objective after re-optimising the bits is accepted.
    """
    plain = (mode == "tangent" and p.channel == "fs" and p.power == "par")
    radii = [None] if plain else [1.5 * p.V * p.delta, 0.75 * p.V * p.delta, 0.33 * p.V * p.delta]
    q = q0.copy()
    cur = evaluate(q, w, p)
    hist = [cur["J"]]
    for it in range(maxit):
        accepted = False
        for rad in radii:
            qn = stepB(q, cur["tau"], w, p, mode, rad)
            if qn is None:
                continue
            new = evaluate(qn, w, p, cur["tau"])
            if new["J"] < cur["J"] - 1e-9:
                rel = (cur["J"] - new["J"]) / cur["J"]
                q, cur = qn, new
                hist.append(cur["J"])
                accepted = True
                break
        if not accepted or rel < tol:
            break
    cur["q"] = q
    cur["hist"] = hist
    cur["iters"] = len(hist) - 1
    return cur


def circle_init(w, p, direction=1):
    """Largest feasible loop through the depot, centred towards the terminal centroid."""
    N = p.N
    cen = w.mean(0)
    u = cen / (np.linalg.norm(cen) + 1e-12)
    bud = budget(p)
    best = None
    for r in np.linspace(0, p.V * (N + 1) * p.delta / (2 * np.pi), 120):
        c = r * u
        ang0 = np.arctan2(-u[1], -u[0])
        th = ang0 + direction * 2 * np.pi * np.arange(1, N + 1) / (N + 1)
        q = c + r * np.stack([np.cos(th), np.sin(th)], 1)
        dd = np.diff(path(q), axis=0)
        if ((dd ** 2).sum(1) > (p.V * p.delta) ** 2 * (1 + 1e-9)).any():
            continue
        if fly_energy(q, p) > bud * (1 + 1e-9):
            continue
        best = q
    return best if best is not None else np.zeros((N, 2))


def run_all(w, p, methods=("local", "hover", "circle", "proposed", "ratelb")):
    out = {}
    N = p.N
    z = np.zeros((N, 2))
    if "local" in methods:
        out["local"] = dict(J=local_energy(p))
    if "hover" in methods or True:
        out["hover"] = evaluate(z, w, p)
        out["hover"]["fly"] = fly_energy(z, p)
    if "circle" in methods or "proposed" in methods or "ratelb" in methods:
        cands = []
        for dr in (1, -1):
            qc = circle_init(w, p, dr)
            cands.append((evaluate(qc, w, p), qc))
        ev, qc = min(cands, key=lambda c: c[0]["J"])
        out["circle"] = ev
        out["circle"]["fly"] = fly_energy(qc, p)
    for m, mode in (("proposed", "tangent"), ("ratelb", "ratelb")):
        if m in methods:
            t0 = time.time()
            r1 = bcd(w, p, z, mode)
            r2 = bcd(w, p, qc, mode)
            r = r1 if r1["J"] <= r2["J"] else r2
            r["time"] = time.time() - t0
            r["iters"] = r1["iters"]
            out[m] = r
    return out


def placements(n, Kmax=24, seed=2026, side=300.0):
    return [np.random.default_rng(seed + i).uniform(-side, side, (Kmax, 2)) for i in range(n)]
