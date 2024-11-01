import argparse
from typing import Any, Dict, Tuple
from bench import Benchmark, add_benchmark_args
from cloud.benchmark.util import BenchmarkResults
import multiprocessing as mp
import sys


def run_benchmark_process(process_args: Tuple[int, int, dict]) -> str:
    """Run a single benchmark process
    Args:
        process_args: Tuple of (process_id, query_id, results, bench_kwargs)
    """
    process_id, query_id, bench_kwargs = process_args
    try:
        # Modify prefix for this process group
        bench_kwargs = bench_kwargs.copy()
        bench_kwargs['prefix'] = f"{bench_kwargs['prefix']}-{process_id}"

        benchmark = Benchmark(**bench_kwargs)
        result = benchmark.run()
        return result.to_json()

    except Exception as e:
        print(f"Process {process_id}, query {query_id} failed: {e}", file=sys.stderr)
        return None

def run_multi_benchmark(
    num_processes: int,
    query_process: int,
    dataset: str,
    num_tables: int,
    batch_size: int,
    num_queries: int,
    ingest: bool,
    index: bool,
    prefix: str,
    reset: bool,
) -> BenchmarkResults:
    total_processes = num_processes * (query_process if not ingest and not index else 1)
    print(f"Starting {total_processes} benchmark processes...")

    bench_kwargs = {
        'dataset': dataset,
        'num_tables': num_tables,
        'batch_size': batch_size,
        'num_queries': num_queries,
        'ingest': ingest,
        'index': index,
        'prefix': prefix,  # Base prefix, will be modified per process
        'reset': reset
    }

    process_args = []

    if ingest or index:
        for i in range(0, num_processes):
            process_kwargs = bench_kwargs.copy()
            process_kwargs['prefix'] = f"{prefix}-{i}"
            process_args.append((i, 0, process_kwargs))
    else:
        for i in range(0, num_processes):
            base_prefix = f"{prefix}-{i}"
            for j in range(0, query_process):
                process_kwargs = bench_kwargs.copy()
                process_kwargs['prefix'] = base_prefix
                process_args.append((i, j, process_kwargs))

    with mp.Pool(processes=total_processes) as pool:
        process_results = pool.map(run_benchmark_process, process_args)

        successful_results = [BenchmarkResults.from_json(r) for r in process_results if r is not None]

        if not successful_results:
            raise RuntimeError("All benchmark processes failed - check logs for details")

        return BenchmarkResults.combine(successful_results)

def validate_args(args: argparse.Namespace):
    if args.query_process > 1:
        if args.ingest or args.index:
            raise ValueError(
                "Multiple query processes per table (query_process > 1) is only allowed "
                "with --no-ingest and --no-index flags"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument
    parser.add_argument(
        "-n",
        "--num_processes",
        type=int,
        required=False,
        default=1,
        help="Number of total benchmark process. This number should be the same for data ingestion and data querying."
    )
    parser.add_argument(
        "-qn",
        "--query_process",
        type=int,
        required=False,
        default=1,
        help="Number of the query process per table. When this number is not 1, total process because num_processes * query_process."
    )
    add_benchmark_args(parser)
    args = parser.parse_args()
    validate_args(args)
    print(args)

    result = run_multi_benchmark(
        args.num_processes,
        args.query_process,
        args.dataset,
        args.tables,
        args.batch,
        args.queries,
        args.ingest,
        args.index,
        args.prefix,
        args.reset,
    )

    result.print()


if __name__ == "__main__":
    main()