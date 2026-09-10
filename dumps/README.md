# Test-split Ō dumps, tracked on purpose

`results/` is gitignored, and on the V100 it is root-owned (the jobs run in a
container), so neither `git pull` nor `scp` can put a file there as the user.
These are 386 KB each and are inputs to an offline audit rather than run
artifacts, so they travel with the repo instead:

- `best_glueball_gelt_sm0-2-4-6_test_obars.pt`       — Run 5
- `best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt`  — the seed-1 replication
- `best_glueball_gelt_sm0-2-4-6-8-12-16_test_obars.pt`      — the 7-level net, run5
- `best_glueball_gelt_sm0-2-4-6-8-12-16_ens1_test_obars.pt` — the 7-level net, ens1

The two 7-level nets are **d_model 24**, although neither the name nor the
dump's `meta` says so (they predate the width tag). Their `Obar_basis` is still
the 4-level `published` basis, so the fair fight's slice gate runs on them
unchanged.

The trained operator's per-configuration Ō(t) on the held-out split, dumped by
`train_glueball.py`. Consumed by `su2_fair_fight.py` and
`operator_decomposition.py`; the ensemble each pairs with is derived from the
`_ens1` tag in the filename, so do not rename them.

    SFF_DUMPS=dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt,dumps/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt \
      python scripts/su2_fair_fight.py
