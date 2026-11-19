"""Where the headroom of the 0⁺⁺ operator is — and where it is not. Offline, ~1 minute.

Four measurements on the kept dumps, none of which needs a fit window, a fit
model or the A₀ estimator (which is saturated: a one-state fit on [2, 7] reads
0.93–1.0 where the true overlap is ≈ 0.88, table 1). Every difference is formed
inside every delete-block jackknife sample, on shared configurations.

  1. PURITY (high statistics, 40k configurations). cosh effective mass of the
     trained net and the single-operator purity deficit
         η(a) = C(a)² / (C(0) C(2a)),   η(a → ∞) = the TRUE A₀.
     η is the a = "one operator" case of table 4.

  2. OWN LOSS. The training loss, L = −mean[C(1)/C(0), C(2)/C(0)], of the trained
     net against the classical arms projected with the GEVP at (t0, td) = (0, 1),
     the classical arm CROSS-FITTED (v from the even blocks, scored on the odd
     ones, swapped, averaged) so it is out of sample like the net. This is the
     quantity the net was trained to minimise: losing here is not saturation.

  3. UNION. L(arm ∪ net) − L(arm), cross-fitted the same way: what the net adds
     to a classical span on the loss. Zero means the net is, for this objective,
     a point inside the span.

  4. EUCLIDEAN COMPLETENESS of a span S with probe G:
         η_S(a) = c(a)ᵀ C_S(0)⁻¹ c(a) / C_GG(2a),   c_i(a) = ⟨O_i(t) G(t ± a)⟩.
     In transfer-matrix language η_S(a) = ⟨G|T^a P_S T^a|G⟩ / ⟨G|T^{2a}|G⟩, with
     P_S the projector on S|0⟩: 1 − η_S(a) is the share of the Euclidean-evolved
     state T^a|G⟩ that lies OUTSIDE the span, and η_S(a → ∞) is the largest true
     ground-state overlap any operator of S can have. It is ≤ 1 by Cauchy–Schwarz
     (noise can exceed it) and needs no fit. On a periodic lattice C(2a) picks up
     the backward-propagating term sooner than C(a); the "wrap" column applies
     the ground-state correction [1 + e^{−m(Nt−4a)}] / [1 + e^{−m(Nt−2a)}]² with
     m = --m-ref (default 0.313, the high-statistics cosh plateau), exact once the
     ground state dominates. Read a ≤ 3 without it, a = 4, 5 only with it.

Table 1 also carries the control for the obvious alternative ("train at a longer
lag"): in a linear span, moving the GEVP from (0, 1) to longer or later pairs
changes the signal-to-noise at the plateau distance by a few per cent at most.

The regression of O(t ± a) on the present slice is what table 4 measures in a
linear span; scripts/train_distill.py is the same regression with a network.

Usage:  python scripts/variational_headroom.py [--m-ref=0.313] [--eps=1e-4]
                                               [--hs=<dump>] [--out=<stem>]
"""

import math
import os
import sys

import torch

_known = ("--m-ref=", "--eps=", "--hs=", "--out=")
_bad = [a for a in sys.argv[1:] if not a.startswith(_known)]
if _bad:
    raise SystemExit(f"unknown argument(s) {_bad}; known: {list(_known)}")


def _flag(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")),
                default)


M_REF = float(_flag("m-ref", "0.313"))
EPS = float(_flag("eps", "1e-4"))
HS = _flag("hs", "dumps/best_glueball_gelt_sm0-2-4-6_ens2_hs_test_obars.pt")
OUT = _flag("out", "results/glueball/variational_headroom")
BLOCK = 10  # the dumps' jackknife block on the 400-configuration test splits
HS_BLOCKS = 200
ENS = {"run5": "", "ens1": "_ens1", "ens2": "_ens2"}
NETS = {"4lv": "sm0-2-4-6", "7lv": "sm0-2-4-6-8-12-16"}
ARMS = ("published", "deep", "shapes_sm", "full")


# ── correlators ───────────────────────────────────────────────────────────────
def corr1(o, kmax):
    """Connected C(0..kmax) of one operator ``(n, Nt)``, every time origin."""
    d = o - o.mean()
    return torch.stack([(d * d.roll(-k, 1)).mean() for k in range(kmax + 1)])


def corr_mat(O, kmax):
    """``(K, n, Nt)`` → symmetrised connected ``(kmax + 1, K, K)``."""
    d = O - O.mean(dim=(1, 2), keepdim=True)
    n = O.shape[1] * O.shape[2]
    C = torch.stack([torch.einsum("ant,bnt->ab", d, d.roll(-k, 2)) / n
                     for k in range(kmax + 1)])
    return 0.5 * (C + C.transpose(1, 2))


def unit(O):
    """Each operator to unit variance: no Ritz value moves, the cut becomes relative."""
    return O / O.std(dim=(1, 2), keepdim=True)


def whiten(C0, eps):
    w, U = torch.linalg.eigh(C0)
    keep = w > eps * w[-1]
    return U[:, keep] / w[keep].sqrt()


def ground_vector(C, eps):
    """GEVP ground vector at (t0, td) = (0, 1), truncated whitening."""
    W = whiten(C[0], eps)
    M = W.T @ C[1] @ W
    _, Q = torch.linalg.eigh(0.5 * (M + M.T))
    return W @ Q[:, -1]


def crossfit(B, eps, kmax=3):
    """C(0..kmax) of the (0, 1)-projected operator of basis ``B``, out of sample.

    v is fitted on the even jackknife blocks and scored on the odd ones, then
    swapped; the two connected correlators are averaged.
    """
    n = B.shape[1]
    blk = (torch.arange(n) // BLOCK) % 2
    out = 0
    for fit, ev in ((blk == 0, blk == 1), (blk == 1, blk == 0)):
        v = ground_vector(corr_mat(B[:, fit], 1), eps)
        out = out + corr1(torch.einsum("k,knt->nt", v, B[:, ev]), kmax) / 2
    return out


def loss_of(C):
    """The training loss of train_glueball.rayleigh_loss, without its scale pin."""
    return -(C[1] / C[0] + C[2] / C[0]) / 2


def cosh_mass(C, t, Nt):
    """m solving C(t)/C(t+1) = cosh ratio (bisection); nan where the ratio ≤ 1."""
    r = (C[t] / C[t + 1]).item()
    if not r > 1:
        return float("nan")
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        f = (math.exp(-m * t) + math.exp(-m * (Nt - t))) / (
            math.exp(-m * (t + 1)) + math.exp(-m * (Nt - t - 1)))
        lo, hi = (m, hi) if f < r else (lo, m)
    return 0.5 * (lo + hi)


def wrap_factor(a, Nt, m):
    return (1 + math.exp(-m * (Nt - 4 * a))) / (1 + math.exp(-m * (Nt - 2 * a))) ** 2


def eta(S, G, amax, eps):
    """η_S(a), a = 1..amax, for span ``S (K, n, Nt)`` and probe ``G (n, Nt)``."""
    S = unit(S)
    dS = S - S.mean(dim=(1, 2), keepdim=True)
    dG = G - G.mean()
    n = G.numel()
    C0 = torch.einsum("ant,bnt->ab", dS, dS) / n
    W = whiten(0.5 * (C0 + C0.T), eps)
    out = []
    for a in range(1, amax + 1):
        c = 0.5 * (torch.einsum("ant,nt->a", dS, dG.roll(-a, 1))
                   + torch.einsum("ant,nt->a", dS, dG.roll(a, 1))) / n
        out.append(((W.T @ c) ** 2).sum() / (dG * dG.roll(-2 * a, 1)).mean())
    return torch.stack(out)


# ── jackknife ─────────────────────────────────────────────────────────────────
def jackknife(fn, n, block):
    """Delete-block jackknife of ``fn(keep_mask) -> 1-d tensor``: (full, err)."""
    nb = n // block
    full = fn(torch.ones(n, dtype=torch.bool))
    s = []
    for i in range(nb):
        m = torch.ones(n, dtype=torch.bool)
        m[i * block:(i + 1) * block] = False
        s.append(fn(m))
    s = torch.stack(s)
    return full, ((nb - 1) / nb * ((s - s.mean(0)) ** 2).sum(0)).sqrt()


def combine(rows):
    """Inverse-variance combination of [(value, err), ...] tensors → (mean, err)."""
    f = torch.stack([r[0] for r in rows])
    e = torch.stack([r[1] for r in rows])
    w = 1 / e ** 2
    return (w * f).sum(0) / w.sum(0), 1 / w.sum(0).sqrt()


def pm(x, e, d=4):
    return f"{x:+.{d}f} ± {e:.{d}f}"


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    result = {"m_ref": M_REF, "eps": EPS}

    # ── 1. purity at high statistics ──────────────────────────────────────────
    if os.path.exists(HS):
        d = torch.load(HS, weights_only=False)
        G, B = d["gelt_obar"].double(), d["Obar_basis"].double()
        n, Nt = G.shape
        blk = n // HS_BLOCKS
        say(f"1. PURITY at high statistics  ({HS}, {n} configurations, block {blk})")

        def f1(m):
            C = corr1(G[m], 11)
            return torch.tensor([cosh_mass(C, t, Nt) for t in range(8)]
                                + [(C[a] ** 2 / (C[0] * C[2 * a])).item() for a in range(1, 6)])

        v, e = jackknife(f1, n, blk)
        say("   cosh m_eff(Δ), Δ = 0..7 : " + "  ".join(f"{v[t]:.4f}({e[t] * 1e4:.0f})"
                                                       for t in range(8)))
        say("   η(a) = C(a)²/(C(0)C(2a)), a = 1..5")
        say("      raw  : " + "  ".join(f"{v[7 + a]:.3f}({e[7 + a] * 1e3:.0f})"
                                        for a in range(1, 6)))
        say("      wrap : " + "  ".join(
            f"{v[7 + a] * wrap_factor(a, Nt, M_REF):.3f}({e[7 + a] * 1e3:.0f})"
            for a in range(1, 6)) + f"   (m_ref = {M_REF}) → the true A₀")
        result["hs_meff"], result["hs_eta_single"] = (v[:8], e[:8]), (v[8:], e[8:])

        say("   Euclidean completeness, probe = the net:")
        for lbl, S in (("published (4)", B), ("published ∪ net", torch.cat([B, G[None]]))):
            v, e = jackknife(lambda m, S=S: eta(S[:, m], G[m], 5, 1e-10), n, blk)
            say(f"      span {lbl:16s} raw: " + "  ".join(
                f"{v[a]:.3f}({e[a] * 1e3:.0f})" for a in range(5)))
            result[f"hs_eta_{lbl}"] = (v, e)

        # The lag of the objective, in a linear span. The net's loss maximises
        # C(1)/C(0), C(2)/C(0); the Parisi–Lepage signal-to-noise of a correlator
        # at separation Δ is C(Δ)/C(0), so a GEVP at (0, a) maximises the S/N at
        # the distance a and one at t0 > 0 ignores what T^{t0} removes anyway.
        # v is fitted on the first half of the chain at each (t0, td) and scored
        # on the second: C(5)/C(0) relative to the (0, 1) vector's.
        say("   the objective's lag, linear span: S/N at Δ = 5, C(5)/C(0), relative to (0,1)")
        pairs = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3)]
        half = n // 2
        for lbl, S in (("published (4)", B), ("published ∪ net", torch.cat([B, G[None]]))):
            Su = unit(S)
            CA = corr_mat(Su[:, :half], 4)
            vecs = []
            for t0, td in pairs:
                W = whiten(CA[t0], 1e-10)
                M = W.T @ CA[td] @ W
                vecs.append(W @ torch.linalg.eigh(0.5 * (M + M.T))[1][:, -1])
            ops = torch.einsum("pk,knt->pnt", torch.stack(vecs), Su[:, half:])

            def f5(m, ops=ops):
                r = torch.stack([(lambda C: C[5] / C[0])(corr1(o[m], 5)) for o in ops])
                return r / r[0]

            v, e = jackknife(f5, n - half, blk)
            say(f"      span {lbl:16s} " + "  ".join(
                f"({t0},{td}) {v[i]:.3f}({e[i] * 1e3:.0f})" for i, (t0, td) in enumerate(pairs)))
            result[f"hs_lag_{lbl}"] = (v, e)
    else:
        say(f"1. (no high-statistics dump at {HS}; skipped)")

    # ── 2–4 on the 400-configuration test splits ──────────────────────────────
    own, union, comp = {}, {}, {}
    for lv, stem in NETS.items():
        for ens, tag in ENS.items():
            dump = f"dumps/best_glueball_gelt_{stem}{tag}_test_obars.pt"
            cache = f"dumps/su2_fair_fight_obars_{ens}.pt"
            if not (os.path.exists(dump) and os.path.exists(cache)):
                continue
            g = torch.load(dump, weights_only=False)["gelt_obar"].double()
            F = torch.load(cache, weights_only=False)["bases"]
            n = g.shape[0]
            for arm in ARMS:
                if arm not in F:
                    continue
                Bs = unit(F[arm].double())

                def f2(m, Bs=Bs):
                    Cg = corr1(g[m], 3)
                    Cc = crossfit(Bs[:, m], EPS)
                    Cu = crossfit(torch.cat([Bs[:, m], unit(g[None, m])]), EPS)
                    return torch.stack([
                        loss_of(Cg), loss_of(Cc), loss_of(Cg) - loss_of(Cc),
                        torch.log(Cg[1] / Cg[2]) - torch.log(Cc[1] / Cc[2]),
                        loss_of(Cu) - loss_of(Cc),
                        torch.log(Cu[1] / Cu[2]) - torch.log(Cc[1] / Cc[2])])

                v, e = jackknife(f2, n, BLOCK)
                own.setdefault((lv, arm), []).append((ens, v, e))
                for un in (False, True):
                    S = torch.cat([F[arm].double(), g[None]]) if un else F[arm].double()
                    ve = jackknife(lambda m, S=S: eta(S[:, m], g[m], 3, EPS), n, BLOCK)
                    comp.setdefault((lv, arm + (" ∪ net" if un else "")), []).append(ve)

    say("\n2. OWN LOSS  L = −mean[C(1)/C(0), C(2)/C(0)]; classical arm at (0,1), cross-fitted."
        "\n   ΔL = L(net) − L(classical): POSITIVE = the net is behind on its own objective.")
    for (lv, arm), rows in own.items():
        per = "   ".join(f"{ens} {pm(v[2], e[2])}" for ens, v, e in rows)
        m, s = combine([(v[2:4], e[2:4]) for _, v, e in rows])
        say(f"   {lv} net − {arm:9s}: ΔL {pm(m[0], s[0])} ({m[0] / s[0]:+.1f}σ)"
            f"   Δm_eff(1) {pm(m[1], s[1])}   [{per}]")
        result[f"own_{lv}_{arm}"] = (m, s)

    say("\n3. UNION  L(arm ∪ net) − L(arm), cross-fitted: what the net adds to the span.")
    for (lv, arm), rows in own.items():
        if arm in ("shapes_sm", "full"):
            m, s = combine([(v[4:6], e[4:6]) for _, v, e in rows])
            say(f"   {lv} net added to {arm:9s}: ΔL {pm(m[0], s[0])} ({m[0] / s[0]:+.1f}σ)"
                f"   Δm_eff(1) {pm(m[1], s[1])}")
            result[f"union_{lv}_{arm}"] = (m, s)

    say("\n4. EUCLIDEAN COMPLETENESS η_S(a), a = 1..3, probe = the ensemble's net (raw)."
        "\n   1 − η = the share of T^a|net⟩ outside the span.")
    for (lv, span), rows in comp.items():
        m, s = combine(rows)
        say(f"   {lv}  span {span:17s}: " + "  ".join(
            f"a={a + 1}: {m[a]:.3f}({s[a] * 1e3:.0f})" for a in range(3)))
        result[f"eta_{lv}_{span}"] = (m, s)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(result, OUT + ".pt")
    with open(OUT + ".txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}.pt, {OUT}.txt")


if __name__ == "__main__":
    main()
