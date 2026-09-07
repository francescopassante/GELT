# Test-split Ō dumps, tracked on purpose

`results/` is gitignored, and on the V100 it is root-owned (the jobs run in a
container), so neither `git pull` nor `scp` can put a file there as the user.
These two are 386 KB each and are inputs to an offline audit rather than run
artifacts, so they travel with the repo instead:

- `best_glueball_gelt_sm0-2-4-6_test_obars.pt`       — Run 5
- `best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt`  — the seed-1 replication

The trained operator's per-configuration Ō(t) on the held-out split, dumped by
`train_glueball.py`. Consumed by `su2_fair_fight.py` and
`operator_decomposition.py`; the ensemble each pairs with is derived from the
`_ens1` tag in the filename, so do not rename them.

    SFF_DUMPS=dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt,dumps/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt \
      python scripts/su2_fair_fight.py
