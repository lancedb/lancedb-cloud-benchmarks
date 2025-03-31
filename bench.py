import argparse
import concurrent
import os
import sys
import time
from concurrent.futures import wait
import traceback
from typing import Iterable, List, Tuple, Optional
import multiprocessing as mp

from lancedb.remote.errors import LanceDBClientError
from lancedb.remote.table import RemoteTable

import lancedb
import numpy as np
import pyarrow as pa
from datasets import load_dataset, DownloadConfig

from cloud.benchmark.util import await_indices, BenchmarkResults
from cloud.benchmark.query import QueryType, VectorQuery, FTSQuery, HybridQuery


def add_benchmark_args(parser: argparse.ArgumentParser):
    """Add benchmark arguments to an existing parser"""
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="KShivendu/dbpedia-entities-openai-1M",
        help="huggingface dataset name",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=int,
        default=4,
        help="number of concurrent tables per process",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=20000,
        help="max batch size for ingestion",
    )
    parser.add_argument(
        "-q",
        "--queries",
        type=int,
        default=1000,
        help="number of queries to run against each table",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        choices=[qt.value for qt in QueryType],
        default=QueryType.VECTOR.value,
        help="type of query to run",
    )
    parser.add_argument(
        "--ingest",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="run ingestion before queries",
    )
    parser.add_argument(
        "--index",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="create indices",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="ldb-cloud-benchmarks",
        help="table name prefix",
    )
    parser.add_argument(
        "-r",
        "--reset",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="drop tables before starting",
    )


class Benchmark:
    def __init__(
        self,
        dataset: str,
        num_tables: int,
        batch_size: int,
        num_queries: int,
        query_type: str,
        ingest: bool,
        index: bool,
        prefix: str,
        reset: bool,
    ):
        self.dataset = dataset
        self.num_tables = num_tables
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.ingest = ingest
        self.index = index
        self.prefix = prefix
        self.reset = reset

        azure_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        storage_options = None
        if azure_account_name:
            storage_options = {"azure_storage_account_name": azure_account_name}
        self.db = lancedb.connect(
            uri=os.environ["LANCEDB_DB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
            region=os.getenv("LANCEDB_REGION", "us-east-1"),
            storage_options=storage_options,
        )

        if query_type == QueryType.VECTOR.value:
            self.query_obj = VectorQuery()
        elif query_type == QueryType.VECTOR_WITH_FILTER.value:
            self.query_obj = VectorQuery(filter=True)
        elif query_type == QueryType.FTS.value:
            self.query_obj = FTSQuery()
        elif query_type == QueryType.HYBRID.value:
            self.query_obj = HybridQuery()

        self.tables: List[RemoteTable] = []
        self.results = BenchmarkResults()
        self.results.tables = num_tables

    def run(self) -> BenchmarkResults:
        if self.reset:
            self._drop_tables()

        if self.ingest:
            self.tables = list(self._create_tables())
            self._ingest()
        else:
            self.tables = list(self._open_tables())

        if self.index:
            self._create_indices()

        if self.num_queries > 0:
            self._query_tables()
        return self.results

    def _create_tables(self) -> Iterable[RemoteTable]:
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("title", pa.string()),
                pa.field("text", pa.string()),
                pa.field("openai", pa.list_(pa.float32(), 1536)),
            ]
        )
        for i in range(self.num_tables):
            table_name = f"{self.prefix}-{i}"
            try:
                table = self.db.create_table(
                    table_name,
                    schema=schema,
                )
            except Exception as e:
                if "already exists" in str(e):
                    print(f"Reusing existing table {table_name}. Use --reset to reset table data")
                    table = self.db.open_table(table_name)
                else:
                    raise

            yield table

    def _open_tables(self) -> Iterable[RemoteTable]:
        for i in range(self.num_tables):
            table_name = f"{self.prefix}-{i}"
            yield self.db.open_table(table_name)

    def _drop_tables(self):
        try:
            tables = list(self._open_tables())
        except Exception:
            return

        for t in tables:
            print(f"dropping table {t.name}")
            try:
                self.db.drop_table(t.name)
            except Exception:
                return

    def _ingest(self):
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.tables)
        ) as executor:
            futures = []

            for table in self.tables:
                futures.append(executor.submit(self._ingest_table, table))
            results = [future.result() for future in futures]

            total_s = time.time() - start
            total_rows = sum(results)

            self.results.ingest_duration_second = total_s
            self.results.ingest_rows = total_rows
            self.results.ingest_rows_per_second = (
                self.results.ingest_rows / self.results.ingest_duration_second
            )

            print(
                f"ingested {total_rows} rows in {len(self.tables)} tables in {total_s:.1f}s. average: {total_rows / total_s:.1f}rows/s"
            )

    def _ingest_table(self, table: RemoteTable) -> int:
        # todo: support batch size > 1000
        add_times = []
        begin = time.time()
        total_rows = 0
        for batch in self._convert_dataset(table.schema):
            for slice in self._split_record_batch(batch, self.batch_size):
                start_time = time.time()
                self._add_batch(table, slice)
                total_rows += len(slice)
                elapsed = int((time.time() - start_time) * 1000)
                add_times.append(elapsed)
                print(
                    f"{table.name}: added batch with size {len(slice)} in {elapsed}ms. rows in table: {table.count_rows()}"
                )

        total_s = int((time.time() - begin))
        print(
            f"{table.name}: ingested {total_rows} rows in {total_s}s. average: {total_rows / total_s:.1f}rows/s"
        )
        self._add_percentiles("ingest", add_times)
        return total_rows

    def _add_batch(self, table, batch):
        try:
            table.add(batch)
        except Exception as e:
            print(f"{table.name}: error during add: {e}")

    def _split_record_batch(self, record_batch, batch_size):
        num_rows = record_batch.num_rows
        for i in range(0, num_rows, batch_size):
            yield record_batch.slice(i, min(batch_size, num_rows - i))

    def _query_tables(self):
        num_tables = len(self.tables)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
            futures = []

            for table in self.tables:
                futures.append(executor.submit(self._query_table, table))
            results = [future.result() for future in futures]

            total_queries = self.num_queries * num_tables
            total_qps = sum(results)

            self.results.total_queries = total_queries
            self.results.queries_per_second = total_qps
            print(
                f"completed {total_queries} queries on {num_tables} tables. average: {total_qps:.1f}QPS"
            )

    def _await_index(self, table: RemoteTable, index_type: str, start_time):
        await_indices(table, 1, [index_type])
        print(
            f"{table.name}: {index_type} indexing completed in {int(time.time() - start_time)}s."
        )

    def _create_indices(self):
        # create the indices - these will be created async
        table_indices = {}
        for t in self.tables:
            t.create_index(
                metric="cosine", vector_column_name="openai", index_type="IVF_PQ"
            )
            t.create_scalar_index("id", index_type="BTREE")
            t.create_fts_index("title")
            table_indices[t] = ["IVF_PQ", "FTS", "BTREE"]

        print("waiting for index completion...")
        start = time.time()

        # poll for index completion in parallel to gather accurate indexing time
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.tables) * 3
        ) as executor:
            futures = []
            for table, indices in table_indices.items():
                for index in indices:
                    futures.append(
                        executor.submit(self._await_index, table, index, start)
                    )
            try:
                wait(futures)
            except Exception as e:
                print(f"Error during index creation: {e}")
            total_s = time.time() - start
            self.results.index_duration_second = total_s
            print(f"found all indices for {len(self.tables)} tables in {total_s:.1f}s.")

    def _to_fixed_size_array(self, array, dim):
        return pa.FixedSizeListArray.from_arrays(array.values, dim)

    def _convert_dataset(self, schema) -> Iterable[pa.RecordBatch]:
        batch_iterator = load_dataset(
            self.dataset,
            cache_dir="/tmp/datasets/cache",
            download_config=DownloadConfig(resume_download=True, disable_tqdm=True),
            split="train",
        ).data.to_batches()

        buffer = []
        buffer_rows = 0
        for batch in batch_iterator:
            rb = pa.RecordBatch.from_arrays(
                [
                    batch["_id"],
                    batch["title"],
                    batch["text"],
                    self._to_fixed_size_array(batch["openai"], 1536),
                ],
                schema=schema,
            )

            if buffer_rows >= self.batch_size:
                table = pa.Table.from_batches(buffer)
                combined = table.combine_chunks().to_batches(
                    max_chunksize=self.batch_size
                )[0]
                buffer.clear()
                buffer_rows = 0
                yield combined
            else:
                buffer.append(rb)
                buffer_rows += len(rb)

        for b in buffer:
            yield b

    def _query_table(self, table: RemoteTable, warmup_queries=100):
        # log a warning if data is not fully indexed
        total_rows = table.count_rows()
        list_resp = table.list_indices()
        not_indexed = len(list_resp) != 3
        for idx in list_resp:
            stats = table.index_stats(idx["index_name"])
            if total_rows == stats["num_indexed_rows"]:
                not_indexed = False
        if not_indexed:
            print(
                f"{table.name}: warning: indexing is not complete, query performance may be degraded. "
                f"total rows: {total_rows}. indices: {list_resp}"
            )

        print(
            f"{table.name}: starting query test. {self.num_queries=} {warmup_queries=} {total_rows=}"
        )
        for _ in range(warmup_queries):
            self._query(table)

        diffs = []
        begin = time.time()
        for _ in range(self.num_queries):
            start_time = time.time()
            self._query(table)
            elapsed = int((time.time() - start_time) * 1000)
            diffs.append(elapsed)
        total_s = max(int(time.time() - begin), 1)
        qps = self.num_queries / total_s
        print(f"{table.name}: query count: {self.num_queries} average: {qps :.1f}QPS")
        self._add_percentiles("query", diffs)
        return qps

    def _query(self, table: RemoteTable):
        try:
            self.query_obj.query(table)
        except Exception as e:
            print(f"{table.name}: error during query: {e}")

    def _add_percentiles(self, type, diffs, percentiles=[50, 90, 95, 99, 100]):
        percentile_values = {p: np.percentile(diffs, p) for p in percentiles}

        for p, percentile_value in percentile_values.items():
            print(f"p{p}: {percentile_value:.2f}ms")

        # Extend the latency lists instead of overwriting
        if type == "query":
            self.results.query_latencies.extend(diffs)
        elif type == "ingest":
            self.results.ingest_latencies.extend(diffs)


def run_benchmark_process(process_args: Tuple[int, int, dict]) -> Optional[str]:
    """Run a single benchmark process
    Args:
        process_args: Tuple of (process_id, query_id, results, bench_kwargs)
    """
    process_id, query_id, bench_kwargs = process_args
    try:
        # Modify prefix for this process group
        bench_kwargs = bench_kwargs.copy()
        bench_kwargs["prefix"] = f"{bench_kwargs['prefix']}-{process_id}"

        benchmark = Benchmark(**bench_kwargs)
        result = benchmark.run()
        return result.to_json()

    except Exception as e:
        print(f"Process {process_id}, query {query_id} failed: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None


def run_multi_benchmark(
    num_processes: int,
    query_processes: int,
    dataset: str,
    num_tables: int,
    batch_size: int,
    num_queries: int,
    query_type: str,
    ingest: bool,
    index: bool,
    prefix: str,
    reset: bool,
) -> BenchmarkResults:
    total_processes = num_processes * (
        query_processes if not ingest and not index else 1
    )
    print(f"Starting {total_processes} benchmark processes...")

    bench_kwargs = {
        "dataset": dataset,
        "num_tables": num_tables,
        "batch_size": batch_size,
        "num_queries": num_queries,
        "query_type": query_type,
        "ingest": ingest,
        "index": index,
        "prefix": prefix,  # Base prefix, will be modified per process
        "reset": reset,
    }

    process_args = []

    if ingest or index:
        for i in range(0, num_processes):
            process_kwargs = bench_kwargs.copy()
            process_args.append((i, 0, process_kwargs))
    else:
        for i in range(0, num_processes):
            for j in range(0, query_processes):
                process_kwargs = bench_kwargs.copy()
                process_args.append((i, j, process_kwargs))

    if total_processes > 1:
        with mp.Pool(processes=total_processes) as pool:
            process_results = pool.map(run_benchmark_process, process_args)
    else:
        process_results = [run_benchmark_process(process_args[0])]

    successful_results = [
        BenchmarkResults.from_json(r) for r in process_results if r is not None
    ]
    if not successful_results:
        raise RuntimeError(
            "All benchmark processes failed - check logs for details"
        )

    return BenchmarkResults.combine(successful_results)


def validate_args(args: argparse.Namespace):
    if args.query_processes > 1:
        if args.ingest or args.index:
            raise ValueError(
                "Multiple query processes per table (--query-processes > 1) is only allowed "
                "with --no-ingest and --no-index flags"
            )
        if args.num_processes > 1:
            raise ValueError(
                "Multiple query processes per table (--query-processes > 1) is only allowed "
                "with --num-processes 1"
            )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--num-processes",
        type=int,
        required=False,
        default=1,
        help="Total number of benchmark processes. Each process will ingest and query independent tables in parallel.",
    )
    group.add_argument(
        "-qn",
        "--query-processes",
        type=int,
        required=False,
        default=1,
        help="Number of concurrent processes to execute queries against the same set of created tables.",
    )
    add_benchmark_args(parser)
    args = parser.parse_args()
    validate_args(args)
    print(args)

    result = run_multi_benchmark(
        args.num_processes,
        args.query_processes,
        args.dataset,
        args.tables,
        args.batch,
        args.queries,
        args.query_type,
        args.ingest,
        args.index,
        args.prefix,
        args.reset,
    )

    result.print()


if __name__ == "__main__":
    main()
