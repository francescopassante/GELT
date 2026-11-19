"""The learned spectrum (notes/learned_spectrum.md): K-state bases against the GEVP. Offline.

A K-state net (``train_glueball.py`` with ``GLUEBALL_N_STATES=K``) is a learned
K-dimensional *span*: its K outputs, fixed by training. The classical GEVP is a
span too — of 4, 7 or 35 smeared loops — whose best k-dimensional subspace is
chosen on the data. This script puts both through the same variational
readout, on the same configurations, inside the same jackknife:

  λ_n(Δ)     Ritz values of T^Δ in a fixed span: the eigenvalues of
             C(Δ) v = λ C(0) v (GEVP at t0 = 0) restricted to the span.
  Ê_n(Δ)     −log λ_n(Δ)/Δ. The n-th Ritz value bounds the n-th level
             (Hylleraas–Undheim–MacDonald), so Ê_n ≥ E_n up to noise and up to
             the backward-propagating e^{−E(Nt−2Δ)} term (≤ 0.2% at Δ ≤ 2 here,
             common to every operator). Lower is better, with no fit, no window
             and no saturation. **Ê_n(1) is the reading; Ê_n(2) is printed
             beside it.**
  m_eff      log λ_n(Δ)/λ_n(Δ+1), the usual GEVP effective masses (log ratio,
             no cosh correction: ≤ 0.004 at Δ ≤ 4 for the ground state).

Spans compared:
  learned    the K outputs of a K-state net — fixed by training, out of sample;
             also its validation-projected ground state alone ("@val").
  single     the K = 1 net of the same ensemble, when its dump exists.
  classical  `published` (4), `deep` (7), `full` (35, the strongest) and, at
             high statistics, the E-component bases `E1`, `E2` (35 each). Each
             (i) in sample (":in") — the whole span on the evaluation
                 configurations, the GEVP as practised, optimistic for 35
                 operators on 400 configurations; and
             (ii) cross-fitted (":cf", ":cf1") — its top-k GEVP subspace at
                 (t0, td) = (0, 1) fitted on the even blocks and scored on the
                 odd ones and vice versa, the two scores averaged; out of sample
                 like the network. k = K (":cf", --k= overrides) and k = 1
                 (":cf1", the projected ground state).
Labels:      f_A1(n), the C(0)-metric share of learned level n inside the
             classical A1 span (`full`); an E or T2 level has none (symmetry),
             an A1 level most. At high statistics also f_E(n), inside the
             classical E span (both components). The net is not O_h-invariant
             by construction, so the levels it learns are labelled, not assumed.

Every operator is first scaled to unit variance (a fixed rescaling: no Ritz value
moves) so that the truncation eps of the whitening (--eps, relative to the
largest eigenvalue of C(0)) is dimensionless. Every difference is formed inside
every jackknife sample. Blocks never straddle two chains.

Inputs, any mix:
  K-state dumps (positional, ``…_k<K>…_test_obars.pt``; default: every one in
      dumps/): each is paired with its ensemble's fair-fight cache
      (``dumps/su2_fair_fight_obars_<run5|ens1|ens2>.pt``) and, if present, with
      that ensemble's K = 1 net and its untrained K-state twin (``_rnd<k>``), all
      gated on the same test configurations.
  --hs=a.pt,b.pt      ``spectrum_hs_eval.py`` outputs, one chain each, merged.

Flags: --block=<n> (default: the dump's jack_block; 100 for --hs), --eps=1e-4,
--k=<k>, --dmax=8, --two-state=1 (--hs only: the shared-mass two-state ΔA₀ of
each single operator against `full` cross-fitted, through two_state_fit.fit2 on
[1, 8]), --out=<stem> (default results/glueball/spectrum_readout).
Readings: notes/learned_spectrum.md §5, printed at the end, combined over
ensembles by inverse variance; writes ``<stem>.pt`` and ``<stem>.txt``.

Run:  python scripts/spectrum_readout.py dumps/*_k2*_test_obars.pt
      python scripts/spectrum_readout.py --hs=dumps/spectrum_hs_seed2101.pt
"""

import glob
import io
import math
import os
import sys
from contextlib import redirect_stdout

_KNOWN = ("--hs=", "--block=", "--eps=", "--k=", "--dmax=", "--two-state=", "--out=")
_bad = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith(_KNOWN)]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {list(_KNOWN)}")


def _flag(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")),
                default)


import torch  # noqa: E402

from gelt.glueball import _whiten, gevp_vectors  # noqa: E402

torch.set_default_dtype(torch.float64)
HS = [p for p in _flag("hs", "").split(",") if p]
BLOCK = _flag("block", "")
EPS = float(_flag("eps", "1e-4"))
K_CLASSICAL = _flag("k", "")
DMAX = int(_flag("dmax", "8"))
TWO_STATE = _flag("two-state", "0") not in ("0", "", "false", "False")
OUT = _flag("out", "results/glueball/spectrum_readout")
POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("--")]
ENS_TAG = {0: "run5", 1: "ens1", 2: "ens2"}
A1_MIN, NONA1_MAX = 0.5, 0.1  # label thresholds on f_A1 (notes/learned_spectrum.md §5)
TWO_STATE_WINDOW = (1, 8)


# ── Block sums: every correlator of every subset of blocks, from one pass ─────
def _blocked(O, starts, block):
    idx = torch.cat([torch.arange(s, s + block) for s in starts])
    return O[:, idx].reshape(O.shape[0], len(starts), block, O.shape[-1])


class Blocks:
    """Per-block sums of one operator family: C on any block subset in O(1).

    ``O`` : ``(n, B, Nt)``; ``starts`` the first configuration of each block.
    C(Δ) = S_Δ/n − μμᵀ on a periodic time axis, symmetrised — what
    ``gelt.glueball.connected_correlator_matrix`` gives on the same configurations.
    """

    def __init__(self, O, starts, block, dmax):
        Ob = _blocked(O, starts, block)
        self.S = torch.stack([torch.einsum("ibct,jbct->bij", Ob.roll(-d, dims=3), Ob)
                              for d in range(dmax + 2)], dim=1)  # (nb, D, n, n)
        self.s = Ob.sum(dim=(2, 3)).T  # (nb, n)
        self.cnt = block * O.shape[-1]

    def corr(self, keep):
        """C(Δ), Δ = 0 … dmax+1, on the blocks where ``keep`` is True."""
        n = self.cnt * int(keep.sum())
        mu = self.s[keep].sum(0) / n
        C = self.S[keep].sum(0) / n - mu[None, :, None] * mu[None, None, :]
        return 0.5 * (C + C.transpose(-1, -2))


class Cross:
    """Per-block Δ = 0 cross sums of two families (the labels)."""

    def __init__(self, A, B, starts, block):
        Ab = _blocked(A, starts, block).flatten(2)
        Bb = _blocked(B, starts, block).flatten(2)
        self.S = torch.einsum("ibx,jbx->bij", Ab, Bb)
        self.sa, self.sb = Ab.sum(-1).T, Bb.sum(-1).T
        self.cnt = Ab.shape[-1]

    def corr(self, keep):
        n = self.cnt * int(keep.sum())
        return (self.S[keep].sum(0) / n
                - (self.sa[keep].sum(0) / n)[:, None] * (self.sb[keep].sum(0) / n)[None, :])


# ── The variational readout ───────────────────────────────────────────────────
def ritz(C, V=None):
    """λ_n(Δ) of span(V) (C's whole span, truncated at EPS, if V is None), (D, k)."""
    if V is not None:
        C = V.T @ C @ V
    W = _whiten(C[0], EPS if V is None else 1e-10, True)
    out = []
    for d in range(C.shape[0]):
        M = W.T @ C[d] @ W
        out.append(torch.linalg.eigvalsh(0.5 * (M + M.T)).flip(-1))
    return torch.stack(out)


def top_subspace(C, k):
    """The top-k GEVP vectors at (t0, td) = (0, 1): the k-level subspace a basis offers."""
    _, V = gevp_vectors(C, t0=0, td=1, eps=EPS, truncate=True)
    return V[:, :k]


def observables(lam, nlev, prefix):
    """Ê_n(1), Ê_n(2), m_eff^(n)(Δ = 1…4) for n < nlev, keyed ``prefix|q[n]``."""
    out = {}
    for n in range(nlev):
        ln = lam[:, n] if n < lam.shape[1] else torch.full((lam.shape[0],), float("nan"))
        out[f"{prefix}|E1[{n}]"] = -math.log(ln[1]) if ln[1] > 0 else float("nan")
        out[f"{prefix}|E2[{n}]"] = -0.5 * math.log(ln[2]) if ln[2] > 0 else float("nan")
        for d in range(1, 5):
            good = bool(ln[d] > 0) and bool(ln[d + 1] > 0)
            out[f"{prefix}|m{d}[{n}]"] = math.log(ln[d] / ln[d + 1]) if good else float("nan")
    return out


def share_in(CLL0, CLF0, CFF0, v):
    """C(0)-metric share of the operator Σ_k v_k O_k inside span(F)."""
    W = _whiten(CFF0, EPS, True)  # W Wᵀ is the projector onto span(F) in this metric
    x = W.T @ (CLF0.T @ v)
    return float((x @ x) / (v @ CLL0 @ v))


# ── Jackknife ─────────────────────────────────────────────────────────────────
class Result:
    """Central values and delete-one-block samples of every key."""

    def __init__(self, fn, nb):
        every = torch.ones(nb, dtype=torch.bool)
        even = torch.arange(nb) % 2 == 0
        full = fn(every, even, ~even)
        samples = []
        for j in range(nb):
            keep = every.clone()
            keep[j] = False
            samples.append(fn(keep, even & keep, ~even & keep))
        self.keys = list(full)
        self.ix = {k: i for i, k in enumerate(self.keys)}
        self.c = torch.tensor([full[k] for k in self.keys])
        self.v = torch.tensor([[s.get(k, float("nan")) for k in self.keys] for s in samples])

    @staticmethod
    def _err(x):
        nb = x.shape[0]
        return ((nb - 1) / nb * ((x - x.nanmean()) ** 2).nansum()).sqrt().item()

    def has(self, k):
        return k in self.ix and math.isfinite(self.c[self.ix[k]].item())

    def get(self, k):
        i = self.ix[k]
        return self.c[i].item(), self._err(self.v[:, i])

    def diff(self, a, b):
        ia, ib = self.ix[a], self.ix[b]
        return (self.c[ia] - self.c[ib]).item(), self._err(self.v[:, ia] - self.v[:, ib])


def _fmt(v, e):
    return f"{v:+.4f} ± {e:.4f} ({v / e:+.1f}σ)" if e > 0 else f"{v:+.4f}"


# ── One evaluation sample ─────────────────────────────────────────────────────
def analyse(st, k_classical):
    """Every variational observable of every family of one sample, jackknifed.

    Family names: "learned:<…>" are K-state spans, "single:<…>" single
    operators, everything else a classical basis (in sample and cross-fitted).
    """
    raw = st["fams"]
    fams = {n: O / O.std(dim=(1, 2), keepdim=True) for n, O in raw.items()}
    starts, block = st["starts"], st["block"]
    bs = {n: Blocks(O, starts, block, DMAX) for n, O in fams.items()}
    learned = [n for n in fams if n.startswith("learned:")]
    singles = [n for n in fams if n.startswith("single:")]
    classical = [n for n in fams if n not in learned and n not in singles]
    # The validation vector acts on the raw outputs; in the unit-variance basis
    # it is V·std (componentwise).
    v_val = {n: st["V"][n][:, 0] * raw[n].std(dim=(1, 2)) for n in learned if n in st["V"]}
    xa1 = {n: Cross(fams[n], fams["full"], starts, block) for n in learned} if "full" in fams else {}
    e_names = [n for n in ("E1", "E2") if n in fams]
    if e_names:
        eall = torch.cat([fams[n] for n in e_names], 0)
        xe = {n: Cross(fams[n], eall, starts, block) for n in learned}
        ee = Cross(eall, eall, starts, block)
    nlev = max([fams[n].shape[0] for n in learned] + [2])

    def fn(keep, ha, hb):
        out = {}
        for n in learned + singles:
            C = bs[n].corr(keep)
            out.update(observables(ritz(C, torch.eye(C.shape[-1])), nlev, n))
            if n in v_val:
                out.update(observables(ritz(C, v_val[n][:, None]), 1, n + "@val"))
            if n in xa1:
                _, Vl = gevp_vectors(C, t0=0, td=1, eps=1e-10, truncate=True)
                CF0, CLF0 = bs["full"].corr(keep)[0], xa1[n].corr(keep)
                for lvl in range(Vl.shape[1]):
                    out[f"{n}|fA1[{lvl}]"] = share_in(C[0], CLF0, CF0, Vl[:, lvl])
                    if e_names:
                        out[f"{n}|fE[{lvl}]"] = share_in(C[0], xe[n].corr(keep),
                                                         ee.corr(keep), Vl[:, lvl])
        for n in classical:
            C = bs[n].corr(keep)
            k = min(k_classical, fams[n].shape[0])
            out.update(observables(ritz(C), nlev, f"{n}:in"))
            CA, CB = bs[n].corr(ha), bs[n].corr(hb)
            for tag, kk in ((":cf", k), (":cf1", 1)):
                lam = 0.5 * (ritz(CB, top_subspace(CA, kk)) + ritz(CA, top_subspace(CB, kk)))
                out.update(observables(lam, nlev if kk > 1 else 1, n + tag))
        return out

    return Result(fn, len(starts))


def two_state(st):
    """--two-state: shared-mass two-state ΔA₀ of each single operator vs `full`(cf1).

    two_state_fit.fit2 — the record's S1-task-5 estimator — on [1, 8], with the
    classical arm the cross-fitted projected ground state (the vector from one
    half applied to the other, each half normalised to unit C(0)), so neither
    arm is fitted on the configurations it is scored on. σ_Δ from a first
    jackknife pass, fixed in the second (the repo's convention).
    """
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import two_state_fit as tsf
    finally:
        sys.argv = argv
    raw = st["fams"]
    fams = {n: O / O.std(dim=(1, 2), keepdim=True) for n, O in raw.items()}
    Nt = raw["full"].shape[-1]
    starts, block = st["starts"], st["block"]
    bf = Blocks(fams["full"], starts, block, DMAX)
    arms = [n for n in fams if n.startswith("single:")] + [
        n for n in fams if n.startswith("learned:")]
    bsing = {n: Blocks(fams[n], starts, block, DMAX) for n in arms}
    v_val = {n: st["V"][n][:, 0] * raw[n].std(dim=(1, 2)) for n in arms if n in st["V"]}

    def pad(c):
        out = torch.zeros(Nt)
        out[: c.shape[0]] = c
        return out

    def corrs(keep, ha, hb):
        CA, CB = bf.corr(ha), bf.corr(hb)
        va, vb = top_subspace(CA, 1)[:, 0], top_subspace(CB, 1)[:, 0]
        cf = 0.5 * (torch.einsum("i,dij,j->d", va, CB, va) / (va @ CB[0] @ va)
                    + torch.einsum("i,dij,j->d", vb, CA, vb) / (vb @ CA[0] @ vb))
        out = {"full": pad(cf)}
        for n in arms:
            C = bsing[n].corr(keep)
            v = v_val.get(n, torch.ones(1)) if C.shape[-1] > 1 or n in v_val else torch.ones(1)
            c = torch.einsum("i,dij,j->d", v, C, v)
            out[n] = pad(c / c[0])
        return out

    nb = len(starts)
    every = torch.ones(nb, dtype=torch.bool)
    even = torch.arange(nb) % 2 == 0
    jk = []
    for j in range(nb):
        keep = every.clone()
        keep[j] = False
        jk.append(corrs(keep, even & keep, ~even & keep))
    sig = {n: ((nb - 1) / nb * ((torch.stack([s[n] for s in jk]) - torch.stack(
        [s[n] for s in jk]).mean(0)) ** 2).sum(0)).sqrt().clamp_min(1e-12) for n in jk[0]}
    full_c = corrs(every, even, ~even)

    def fn(keep, ha, hb):
        c = full_c if bool(keep.all()) else corrs(keep, ha, hb)
        out = {}
        for n in arms:
            r = tsf.fit2([c[n], c["full"]], [sig[n], sig["full"]], TWO_STATE_WINDOW)
            out[f"{n}|A0"], out["full:cf1|A0#" + n] = r["A0"][0], r["A0"][1]
            out[f"{n}|ts_m0"], out[f"{n}|ts_m1"] = r["m0"], r["m1"]
            out[f"{n}|ts_edge"] = float(r["edge"])
        return out

    return Result(fn, nb)


# ── Inputs ────────────────────────────────────────────────────────────────────
def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _same_configs(a, b):
    """Two Obar_basis from the same test configurations (float32 pipelines)."""
    return a.shape == b.shape and float((a - b).abs().max() / a.std()) < 1e-4


def test_split_sets(paths):
    """One sample per trained K-state dump: its cache, its K = 1 net, its twin."""
    sets = []
    for p in paths:
        d = _load(p)
        meta = d["meta"]
        if "gelt_obars" not in d:
            print(f"  ** {p}: no gelt_obars — not a K-state dump, skipped")
            continue
        if meta.get("random_init"):
            continue  # read beside its trained twin
        seed = int(meta.get("ensemble_seed") or 0)
        tag = ENS_TAG.get(seed)
        cache_p = f"dumps/su2_fair_fight_obars_{tag}.pt"
        if tag is None or not os.path.exists(cache_p):
            print(f"  ** {p}: no fair-fight cache for ensemble {seed} — skipped")
            continue
        cache = _load(cache_p)
        if not _same_configs(d["Obar_basis"].double(), cache["bases"]["published"].double()):
            raise SystemExit(f"{p}: its published basis is not {cache_p}'s — not the same "
                             f"test configurations")
        name = f"learned:{os.path.basename(p).replace('_test_obars.pt', '')}"
        fams = {name: d["gelt_obars"].double()}
        V = {name: d["learned_gevp"]["V"].double()} if "learned_gevp" in d else {}
        for arm in ("published", "deep", "full"):
            fams[arm] = cache["bases"][arm].double()
        ens = "" if seed == 0 else f"_ens{seed}"
        single = f"dumps/best_glueball_gelt_sm0-2-4-6{ens}_test_obars.pt"
        if os.path.exists(single):
            s = _load(single)
            if _same_configs(s["Obar_basis"].double(), d["Obar_basis"].double()):
                fams[f"single:{os.path.basename(single).replace('_test_obars.pt', '')}"] = \
                    s["gelt_obar"].double()[None]
        twins = set(glob.glob(os.path.join(os.path.dirname(p) or ".", "*_rnd*_test_obars.pt")))
        for q in sorted(twins | set(glob.glob("dumps/*_rnd*_test_obars.pt"))):
            t = _load(q)
            tm = t["meta"]
            if ("gelt_obars" in t and tm.get("random_init")
                    and int(tm.get("ensemble_seed") or 0) == seed
                    and tm.get("n_states") == meta["n_states"]
                    and list(tm["input_smear_levels"]) == list(meta["input_smear_levels"])
                    and _same_configs(t["Obar_basis"].double(), d["Obar_basis"].double())):
                fams[f"learned:{os.path.basename(q).replace('_test_obars.pt', '')}"] = \
                    t["gelt_obars"].double()
                break
        block = int(BLOCK or meta.get("jack_block", 10))
        B = fams["full"].shape[1]
        sets.append({"tag": tag, "dump": p, "K": int(meta["n_states"]), "fams": fams, "V": V,
                     "starts": list(range(0, B - B % block, block)), "block": block})
    return sets


def hs_set(paths):
    """Merge spectrum_hs_eval chains; build the A1 and the E classical bases."""
    blobs = [_load(p) for p in paths]
    levels, shapes = list(blobs[0]["levels"]), [tuple(s) for s in blobs[0]["shapes"]]
    for b in blobs[1:]:
        if list(b["levels"]) != levels or [tuple(s) for s in b["shapes"]] != shapes:
            raise SystemExit("the chains were evaluated with different classical tables")
    names = sorted(set.intersection(*[set(b["nets"]) for b in blobs]))
    block = int(BLOCK or 100)
    starts, off = [], 0
    for b in blobs:
        N = b["table"].shape[3]
        starts += [off + s for s in range(0, N - N % block, block)]
        off += N
    table = torch.cat([b["table"].double() for b in blobs], dim=3)  # (lv, sh, 3, N, Lt)
    a1 = table.sum(2)  # Σ planes: the cubic scalar, (lv, sh, N, Lt)
    one = shapes.index((1, 1))
    fams = {"published": a1[[levels.index(x) for x in (0, 2, 4, 6)], one],
            "deep": a1[:, one], "full": a1.reshape(-1, *a1.shape[2:])}
    # The two components of E for each (level, shape): plane differences.
    e1 = (table[:, :, 0] - table[:, :, 1]) / math.sqrt(2)
    e2 = (table[:, :, 0] + table[:, :, 1] - 2 * table[:, :, 2]) / math.sqrt(6)
    fams["E1"], fams["E2"] = e1.reshape(-1, *e1.shape[2:]), e2.reshape(-1, *e2.shape[2:])
    V = {}
    for n in names:
        series = torch.cat([b["nets"][n].double() for b in blobs], dim=1)  # (K, N, Lt)
        if series.shape[0] == 1:
            fams[f"single:{n}"] = series
        else:
            fams[f"learned:{n}"] = series
            val_V = blobs[0]["net_info"][n].get("val_V")
            if val_V is not None:
                V[f"learned:{n}"] = torch.as_tensor(val_V).double()
    K = max([f.shape[0] for k, f in fams.items() if k.startswith("learned:")] + [1])
    return {"tag": "hs", "dump": ",".join(paths), "K": K, "fams": fams, "V": V,
            "starts": starts, "block": block}


# ── Report and readings ───────────────────────────────────────────────────────
def label_of(res, n, lvl):
    if not res.has(f"{n}|fA1[{lvl}]"):
        return "unlabelled"
    f = res.get(f"{n}|fA1[{lvl}]")[0]
    return "A1" if f >= A1_MIN else "non-A1" if f <= NONA1_MAX else "mixed"


def report(st, res, k_cl):
    fams = st["fams"]
    learned = [n for n in fams if n.startswith("learned:")]
    singles = [n for n in fams if n.startswith("single:")]
    classical = [n for n in fams if n not in learned and n not in singles]
    nlev = max([fams[n].shape[0] for n in learned] + [2])
    print(f"\n=== {st['tag']}: {st['dump']}")
    print(f"    {len(st['starts'])} blocks of {st['block']} configurations, eps {EPS:g}, "
          f"classical cross-fit k = {k_cl}; columns Ê(1) Ê(2) m_eff(1) m_eff(2) m_eff(3), "
          f"error ×1e4 in brackets")
    rows = ([(n, n) for n in learned] + [(n + "@val", n + "@val") for n in learned]
            + [(n, n) for n in singles]
            + [(f"{n} in sample", f"{n}:in") for n in classical]
            + [(f"{n} cross-fit k={k_cl}", f"{n}:cf") for n in classical]
            + [(f"{n} cross-fit k=1", f"{n}:cf1") for n in classical])
    for lvl in range(nlev):
        print(f"  level {lvl}")
        for label, key in rows:
            if not res.has(f"{key}|E1[{lvl}]"):
                continue
            cells = [res.get(f"{key}|{q}[{lvl}]") for q in ("E1", "E2", "m1", "m2", "m3")]
            tail = ""
            if res.has(f"{key}|fA1[{lvl}]"):
                tail += f"  f_A1 {res.get(f'{key}|fA1[{lvl}]')[0]:.3f}"
            if res.has(f"{key}|fE[{lvl}]"):
                tail += f"  f_E {res.get(f'{key}|fE[{lvl}]')[0]:.3f}"
            print(f"    {label.split(':', 1)[-1][:52]:<52} " + "  ".join(
                f"{v:.4f}({1e4 * e:4.0f})" if math.isfinite(v) else f"{'—':^13}"
                for v, e in cells) + tail)


def _verdict(v, e, better, worse, level):
    if v <= -3 * e:
        return better
    if v >= 3 * e:
        return worse
    return level if abs(v) < 2 * e else "between 2σ and 3σ: no verdict"


def readings(st, res, ts=None):
    """Lines to print and (reading id, comparison, K) → (value, error) to combine."""
    fams, lines, vals = st["fams"], [], {}
    learned = [n for n in fams if n.startswith("learned:")]
    trained = [n for n in learned if "_rnd" not in n]
    untrained = [n for n in learned if "_rnd" in n]
    singles = [n for n in fams if n.startswith("single:")]

    def put(rid, what, K, v, e):
        vals[(rid, what, K)] = (v, e)

    for n in trained:
        Kn = fams[n].shape[0]
        labels = [label_of(res, n, lvl) for lvl in range(Kn)]
        lines.append(f"{n.split(':', 1)[1]}: level labels {labels}")
        for u in untrained:
            if fams[u].shape[0] == Kn and res.has(f"{u}|E1[1]"):
                v, e = res.diff(f"{u}|E1[1]", f"{n}|E1[1]")
                put("RS1", "untrained − trained, Ê₁(1)", Kn, v, e)
                lines.append(f"  RS1 Ê₁(1) untrained − trained = {_fmt(v, e)} → "
                             + ("learned, not architectural" if v >= 5 * e else "NOT ≥ 5σ"))
        a1 = [lvl for lvl in range(Kn) if labels[lvl] == "A1"]
        if len(a1) >= 2 and res.has("full:cf|E1[1]"):
            lvl = a1[1]
            for arm, what in (("full:cf", "cross-fitted — the reading"),
                              ("full:in", "in sample — context")):
                v, e = res.diff(f"{n}|E1[{lvl}]", f"{arm}|E1[1]")
                v2, e2 = res.diff(f"{n}|E2[{lvl}]", f"{arm}|E2[1]")
                put("RS3" if arm == "full:cf" else "RS3ctx", f"A1* Ê(1) learned − {arm}", Kn, v, e)
                lines.append(f"  RS3 Ê(1) learned A1 level {lvl} − full level 1, {what}: "
                             f"{_fmt(v, e)}   [Ê(2) {_fmt(v2, e2)}] → "
                             + _verdict(v, e, "learned basis BETTER on the excited 0++",
                                        "classical better", "matches"))
        else:
            lines.append("  RS3 not read: fewer than two A1-labelled learned levels")
        for s in singles:
            v, e = res.diff(f"{n}|E1[0]", f"{s}|E1[0]")
            put("RS4", "Ê₀(1) K-span − K=1 net", Kn, v, e)
            lines.append(f"  RS4 Ê₀(1) K-span − {s.split(':', 1)[1][:44]}: {_fmt(v, e)} → "
                         + _verdict(v, e, "K-span better ground state", "K=1 net better",
                                    "same"))
        if res.has("full:cf1|E1[0]"):
            v, e = res.diff(f"{n}|E1[0]", "full:cf1|E1[0]")
            put("RS4f", "Ê₀(1) K-span − full(cf1)", Kn, v, e)
            lines.append(f"  RS4 Ê₀(1) K-span − full (cross-fitted, k=1): {_fmt(v, e)} → "
                         + _verdict(v, e, "better than `full`", "behind `full`", "level"))
        for lvl in [x for x in range(1, Kn) if labels[x] == "non-A1"][:1]:
            if res.has(f"{n}|fE[{lvl}]") and res.has("E1:cf|E1[0]"):
                for arm in ("E1:cf", "E2:cf"):
                    v, e = res.diff(f"{n}|E1[{lvl}]", f"{arm}|E1[0]")
                    put("RH3", f"E Ê(1) learned − {arm}", Kn, v, e)
                    lines.append(f"  RH3 Ê(1) learned non-A1 level {lvl} (f_E "
                                 f"{res.get(f'{n}|fE[{lvl}]')[0]:.2f}) − classical {arm[:2]} "
                                 f"level 0, cross-fitted: {_fmt(v, e)} → "
                                 + _verdict(v, e, "learned BETTER on the 2++ (E)",
                                            "classical better", "matches"))
    if st["tag"] == "hs":
        for s in singles + [n + "@val" for n in trained if res.has(n + "@val|E1[0]")]:
            v, e = res.diff(f"{s}|E1[0]", "full:cf1|E1[0]")
            v2, e2 = res.diff(f"{s}|E2[0]", "full:cf1|E2[0]")
            put("RH1", s, 1, v, e)
            lines.append(f"RH1 {s.split(':', 1)[1][:56]}: Ê₀(1) − full(cf1) = {_fmt(v, e)}  "
                         f"[Ê₀(2) {_fmt(v2, e2)}] → "
                         + _verdict(v, e, "better than `full`", "behind `full`", "level"))
    if ts is not None:
        for s in [n for n in fams if n.startswith(("single:", "learned:"))]:
            if not ts.has(f"{s}|A0"):
                continue
            v, e = ts.diff(f"{s}|A0", f"full:cf1|A0#{s}")
            edge = ts.get(f"{s}|ts_edge")[0]
            m1 = ts.get(f"{s}|ts_m1")
            lines.append(f"two-state ΔA₀ {s.split(':', 1)[1][:48]} − full(cf1): {_fmt(v, e)}  "
                         f"(m₁ {m1[0]:.3f} ± {m1[1]:.3f}, at the grid edge in the central fit: "
                         f"{'yes' if edge else 'no'})")
    return lines, vals


def main():
    paths = POSITIONAL or sorted(glob.glob("dumps/*_k[0-9]*_test_obars.pt"))
    sets = test_split_sets(paths) if paths else []
    if HS:
        sets.append(hs_set(HS))
    if not sets:
        raise SystemExit("nothing to read: give K-state dumps and/or --hs=")
    buf, saved, pooled = io.StringIO(), {}, {}
    for st in sets:
        k_cl = int(K_CLASSICAL or st["K"])
        res = analyse(st, k_cl)
        ts = two_state(st) if (TWO_STATE and st["tag"] == "hs") else None
        with redirect_stdout(buf):
            report(st, res, k_cl)
            lines, vals = readings(st, res, ts)
            print("\n  readings (notes/learned_spectrum.md §5):")
            for line in lines:
                print("   " + line)
        for key, ve in vals.items():
            pooled.setdefault(key, []).append((st["tag"], *ve))
        saved[st["dump"]] = {"keys": res.keys, "central": res.c, "samples": res.v,
                             "block": st["block"], "n_blocks": len(st["starts"])}
    with redirect_stdout(buf):
        multi = {k: v for k, v in pooled.items() if len({t for t, _, _ in v}) > 1}
        if multi:
            print("\n=== combined over ensembles (inverse variance)")
            for (rid, what, K), entries in sorted(multi.items()):
                w = [1 / e**2 for _, _, e in entries if e > 0]
                m = sum(wi * v for wi, (_, v, e) in zip(w, entries)) / sum(w)
                err = 1 / math.sqrt(sum(w))
                per = ", ".join(f"{t} {v:+.4f}({1e4 * e:.0f})" for t, v, e in entries)
                print(f"  {rid} K={K} {what}: {_fmt(m, err)}   [{per}]")
    text = buf.getvalue()
    print(text)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    torch.save({"samples": saved, "readings": pooled}, OUT + ".pt")
    with open(OUT + ".txt", "w") as f:
        f.write(text)
    print(f"wrote {OUT}.pt and {OUT}.txt")


if __name__ == "__main__":
    main()
