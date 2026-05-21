"""
Script to measure the execution time of a function.
"""

import statistics
import time

if __name__ == "__main__":
    from functools import partial

    from gelt import (
        SU,
        build_plaquette_datasets,
        haar_ensemble,
        random_links,
    )
    from gelt.lattice import action

    SU3 = SU(3)
    warmup = 1
    repeats = 3

    def func():
        beta = 1.0
        train, val, test = build_plaquette_datasets(
            N=100,
            D=3,
            L=5,
            gaugegroup=SU(3),
            beta=beta,
            target=partial(action, beta=beta),
            structured=True,
            sampler=haar_ensemble,
            R=3,
        )

    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    mean = statistics.fmean(times)
    median = statistics.median(times)
    best = min(times)
    print(
        f"Execution time over {repeats} runs: "
        f"mean={mean:.4g}s, median={median:.4g}s, min={best:.4g}s"
    )
