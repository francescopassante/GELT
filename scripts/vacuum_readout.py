"""The vacuum study's readings, fixed before the numbers exist. Offline, seconds.

Reads results/vacuum/{replicas,classical,net_*,rw_*}<ens>.pt and applies:

  V0  gate   the conditional autocorrelation plateaus (first − last late point
             within 2σ) and Stein's identity holds on the measured slices
             (|z| < 4). If not, nothing below is read.
  V1         the predictable fraction ‖E[F|U]‖²/⟨‖F‖²⟩ — reported.
  V2         the classical ladder ε(dr) … ε(full) — reported; 1 − ε(dr) is the
             share of the vacuum score a rescaled 3D Wilson action explains.
  V3  the headline, per network residual over an arm:
        R-V1  ε(arm) − ε(net + arm) > 0 at ≥ 3σ → the network finds vacuum
              structure outside the classical ansatz; "substantial" if the gain
              is ≥ 25% of ε(arm).
        R-V0  |gain| < 2σ → at this resolution the classical arm is the vacuum
              as far as this network can tell.
        R-V−  gain < 0 at ≥ 3σ → the selection on validation did not carry to
              the replicas: understand before quoting anything.
  V4  M1 on a target where theory predicts it: the Greensite–Olejník kernel
      depends on the field (through the covariant Laplacian), so a GELT with
      input-dependent offset weights should beat the frozen-α ablation.
      softmax vs frozen, same residual and seed, paired: ≥ 3σ → pays; else tie.
  V5  ground truth, per ladder that passes G-RW: χ²/dof and max|Δ|/σ of every
      model against the exact relative weights. A model is *validated* on a
      family if χ²/dof < 2 and max|Δ|/σ < 4. The network *beats* the classical
      ansatz if its χ² is below half of `full`'s on both `noise` and `plane`.
      `smear` is off the training distribution: reported, not read.

    python scripts/vacuum_readout.py [--vac-ensemble-seed=0]
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vacuum_common as vc  # noqa: E402

vc.validate_argv()

import torch  # noqa: E402


def load(p):
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def main():
    rep, cl = load(vc.replica_path()), load(vc.classical_path())
    if rep is None:
        raise SystemExit(f"no {vc.replica_path()}")
    curve = torch.tensor(rep["curve"], dtype=torch.float64)
    F0F0 = torch.tensor([float((f.double() ** 2).sum()) for f in rep["F0"]])
    sw, late = rep["meta"]["sweeps"], rep["meta"]["late"]
    lt = [j for j, s in enumerate(sw) if s >= late]
    d, de = vc.jackknife_diff(curve[:, lt[0]], curve[:, lt[-1]], F0F0, vc.JACK_BLOCK)
    st = torch.tensor(rep["stein"], dtype=torch.float64)
    z = float(st.mean() / (st.std() / len(st) ** 0.5))
    plateau, stein_ok = abs(d) < 2 * de + 1e-12, abs(z) < 4
    print(f"V0: plateau {d:+.4f} ± {de:.4f} ({'ok' if plateau else 'FAIL'}), "
          f"Stein z = {z:+.2f} ({'ok' if stein_ok else 'FAIL'})")
    if not (plateau and stein_ok):
        raise SystemExit("V0 failed — nothing below is read.")
    S = torch.tensor([float((a.double() * b.double()).sum())
                      for a, b in zip(rep["F0"], rep["Flate"])])
    f, fe = vc.jackknife_ratio(S, F0F0, vc.JACK_BLOCK)
    print(f"V1: predictable fraction of the temporal force {f:.4f} ± {fe:.4f}")

    if cl is None:
        raise SystemExit(f"no {vc.classical_path()}")
    print("V2: classical ladder (ε = 0 exact vacuum score, 1 = Φ ≡ 0):")
    for arm in vc.ARM_ORDER:
        e, ee = cl["arms"][arm]["eps"]
        print(f"   {arm:10s} ε = {e:.4f} ± {ee:.4f}")
    print(f"   → a rescaled 3D Wilson action explains {1 - cl['arms']['dr']['eps'][0]:.1%} "
          "of the non-trivial vacuum score")

    nets = {}
    for p in sorted(glob.glob(f"{vc.OUT}/net_*{vc.ENS}*.pt")):
        d = load(p)
        if d is None or d.get("eval") is None:
            continue
        if d["eval"]["rep_idx"] != list(rep["idx"]):
            print(f"   ** {os.path.basename(p)}: other replicas — skipped")
            continue
        nets[os.path.basename(p)[:-3]] = d
    print("V3: networks, residual over a classical arm:")
    for name, d in nets.items():
        arm = d["config"]["residual"]
        e_net = d["eval"]["e"]
        eps = d["eval"]["eps"]
        if arm == "none":
            print(f"   {name}: ε = {eps[0]:.4f} ± {eps[1]:.4f} (no residual)")
            continue
        g, ge = vc.jackknife_diff(cl["arms"][arm]["e"], e_net, cl["S"], vc.JACK_BLOCK)
        base = cl["arms"][arm]["eps"][0]
        if g > 3 * ge:
            verdict = "R-V1" + (" (substantial)" if g >= 0.25 * base else "")
        elif abs(g) < 2 * ge:
            verdict = "R-V0"
        elif g < -3 * ge:
            verdict = "R-V−"
        else:
            verdict = "between 2σ and 3σ — no reading"
        print(f"   {name}: ε = {eps[0]:.4f} ± {eps[1]:.4f}; gain over {arm} "
              f"{g:+.4f} ± {ge:.4f} ({g / max(ge, 1e-12):+.1f}σ, {g / max(base, 1e-12):.0%} "
              f"of ε({arm})) → {verdict}")

    print("V4: softmax vs frozen α (same residual, seed, inputs):")
    found = False
    for name, d in nets.items():
        c = d["config"]
        if c["arch"] != "gelt" or c["alpha_mode"] != "softmax":
            continue
        twin = name.replace("net_gelt_", "net_gelt_frozen_")
        if twin in nets:
            found = True
            g, ge = vc.jackknife_diff(nets[twin]["eval"]["e"], d["eval"]["e"], cl["S"],
                                      vc.JACK_BLOCK)
            print(f"   ε(frozen) − ε(softmax) = {g:+.4f} ± {ge:.4f} → "
                  + ("input-dependent weighting pays" if g > 3 * ge else "tie"))
    if not found:
        print("   (no softmax/frozen pair yet)")

    print("V5: exact relative weights:")
    for p in sorted(glob.glob(f"{vc.OUT}/rw_*{vc.ENS}.pt")):
        r = load(p)
        tag = f"{r['family']} (config {r['config']}, s_max {r['s'][-1]:.3g}, " \
              f"ΔΦ_true {float(r['true'][-1]):+.2f} ± {float(r['true_err'][-1]):.2f})"
        if not r["gate"]:
            print(f"   {tag}: G-RW failed ({r['bad_rungs']} bad rungs) — not read")
            continue
        print(f"   {tag}{'  [off-distribution: reported only]' if r['family'] == 'smear' else ''}")
        for m, v in r["models"].items():
            ok = v["chi2"] < 2 and v["maxdev"] < 4
            print(f"      {m:38s} χ²/dof {v['chi2']:8.2f}  max|Δ|/σ {v['maxdev']:6.1f}  "
                  f"{'validated' if ok else ''}")


if __name__ == "__main__":
    main()
