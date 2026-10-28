#! /gpfs/work/nimtz/miniforge3/envs/tridiag/bin/python -u
# SBATCH --job-name=benchmarks
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=48

import gc
import os
import pickle
import sys
from functools import partial
from time import perf_counter

import torch
from marinade.gotm.tridiagonal import random_tridiag_dd, tridiagonal_solve


def time_memory_func(f, f_before=None, f_after=None, use_cuda=False, n_runs=20, warm_runs=1):
    """Measure time and memory costs of a function."""
    t = []
    m = []
    for i in range(n_runs + warm_runs):
        if use_cuda:
            assert torch.cuda.is_available()
            # Create CUDA events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            if f_before:
                x = f_before()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if f_before:
                start_event.record()
                f(x)
                end_event.record()
                del x
            else:
                start_event.record()
                f()
                end_event.record()
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            if f_after:
                f_after()
            next_t = start_event.elapsed_time(end_event) / 1000  # ms -> s
        else:
            gc.collect()
            start_time = perf_counter()
            f()
            next_t = perf_counter() - start_time
        gc.collect()
        if i >= warm_runs:
            t.append(next_t)
            if use_cuda:
                m.append(peak_alloc)
            else:
                m.append(0.0)

    return torch.tensor(t, dtype=torch.float64).mean().item(), torch.tensor(m, dtype=torch.float64).mean().item()


def print_cuda_memory(appendix="", use_cuda=False):
    """Print CUDA allocated and reserved memory."""
    if use_cuda:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            "allocated:",
            torch.cuda.memory_allocated() / 1e6,
            "MB",
            "reserved:",
            torch.cuda.memory_reserved() / 1e6,
            "MB",
            appendix,
        )


def benchmark_tridiag_single(
    f_solver,
    method: str,
    implicit_backward: bool,
    requires_grad: bool,
    n_rows_all=None,
    n_batch_all=None,
    use_cuda=False,
    cuda_device_idx=0,
    dtype=None,
    n_runs=100,
    warm_runs=10,
):
    """Measure time and memory costs of a solver for various system dimensions and batch sizes."""
    device = torch.device(f"cuda:{cuda_device_idx}") if use_cuda else "cpu"

    dtype = (torch.float32 if use_cuda else torch.float64) if dtype is None else dtype
    print_cuda_memory("start", use_cuda=use_cuda)
    s = (len(n_rows_all), len(n_batch_all))
    # initialize arrays to store timing results
    t = torch.full(s, torch.nan, dtype=torch.float64)
    m = torch.full(s, torch.nan, dtype=torch.float64)
    dd = partial(random_tridiag_dd, device=device, dtype=dtype)
    print_cuda_memory("before loop", use_cuda=use_cuda)

    for j, n_batch in enumerate(n_batch_all):
        for i, n_rows in enumerate(n_rows_all):
            print(n_rows, n_batch)
            torch.manual_seed(42)

            try:
                a, b, c, y = dd(requires_grad=requires_grad, n_rows=n_rows, n_batch=n_batch)  # generate data
                if f_solver == "tridiagonal_solve":
                    f_before = None
                    f = partial(tridiagonal_solve, a, b, c, y, method=method, implicit_backward=implicit_backward)
                    f_after = None
                elif f_solver == "tridiagonal_sum_backward":
                    f_before = partial(
                        tridiagonal_solve, a, b, c, y, method=method, implicit_backward=implicit_backward
                    )
                    f = lambda x: x.sum().backward()

                    def f_after():
                        a.grad, b.grad, c.grad, y.grad = None, None, None, None

                else:
                    raise Exception("no valid function specified:", f_solver, "was given")
                t[i, j], m[i, j] = time_memory_func(
                    f, f_before, f_after, use_cuda=use_cuda, n_runs=n_runs, warm_runs=warm_runs
                )
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "memory" not in str(e).lower():
                    raise
                print("CUDA" if use_cuda else "system", "out of memory")
                print_cuda_memory(use_cuda=use_cuda)

            a, b, c, y, f, f_before, f_after = None, None, None, None, None, None, None
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    return t, m


if __name__ == "__main__":
    f_solver = "tridiagonal_solve"
    method = "thomas"
    implicit_backward = False
    requires_grad = False
    use_cuda = True

    f_solver = sys.argv[1]
    method = sys.argv[2]
    implicit_backward = eval(sys.argv[3].capitalize())
    requires_grad = eval(sys.argv[4].capitalize())
    use_cuda = eval(sys.argv[5].capitalize())

    print("running on node", os.environ.get("SLURMD_NODENAME"))
    print(
        f'{f_solver} {method} {"implicit" if implicit_backward else "explicit"} {"grad" if requires_grad else "nograd"} {"cuda" if use_cuda else "cpu"}'
    )

    save_path = "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/"

    n_rows_all = [16, 32, 64, 128, 256]
    n_batch_all = [2**i for i in range(0, 28, 1)]

    t, m = benchmark_tridiag_single(
        f_solver,
        method,
        implicit_backward=implicit_backward,
        requires_grad=requires_grad,
        n_rows_all=n_rows_all,
        n_batch_all=n_batch_all,
        use_cuda=use_cuda,
    )

    with open(
        save_path
        + f'{f_solver}_{method}_{"implicit" if implicit_backward else "explicit"}_{"grad" if requires_grad else "nograd"}_{"cuda" if use_cuda else "cpu"}.pkl',
        "wb",
    ) as file:
        pickle.dump(t, file)
        pickle.dump(m, file)
