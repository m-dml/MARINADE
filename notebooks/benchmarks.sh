#! /usr/bin/bash

declare -a method=("thomas" "paracr" "linalg")
declare -a implicit_backward=(true false)
declare -a use_cuda=(true)

for m in ${method[@]}; do
    for ib in ${implicit_backward[@]}; do
        for c in ${use_cuda[@]}; do
            if $c; then
                sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
                    --partition=p2GPU40 benchmarks.py tridiagonal_solve $m $ib False $c
                sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
                    --partition=p2GPU40 benchmarks.py tridiagonal_solve $m $ib True $c
                sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
                    --partition=p2GPU40 benchmarks.py tridiagonal_sum_backward $m $ib True $c
            # else
            #     sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
            #         --partition=pNode benchmarks.py tridiagonal_solve $m $ib False $c
            #     sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
            #         --partition=pNode benchmarks.py tridiagonal_solve $m $ib True $c
            #     sbatch -o "/gpfs/work/nimtz/experiments/benchmark_tridiagonal/slurm-%j.out" \
            #         --partition=pNode benchmarks.py tridiagonal_sum_backward $m $ib True $c
            fi
        done
    done
done
