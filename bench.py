import argparse
import concurrent
import os
import time
from concurrent.futures import wait
from typing import Iterable, List

from lancedb.remote.errors import LanceDBClientError
from lancedb.remote.table import RemoteTable

import lancedb
import numpy as np
import pyarrow as pa
from datasets import load_dataset, DownloadConfig

from cloud.benchmark.util import await_indices, BenchmarkResults


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
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="number of rows for the table",
    )


class Benchmark:
    def __init__(
        self,
        dataset: str,
        num_tables: int,
        batch_size: int,
        num_queries: int,
        ingest: bool,
        index: bool,
        prefix: str,
        reset: bool,
        rows: int,
    ):
        self.dataset = dataset
        self.num_tables = num_tables
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.ingest = ingest
        self.index = index
        self.prefix = prefix
        self.reset = reset
        self.rows = rows

        self.db = lancedb.connect(
            uri=os.environ["LANCEDB_DB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
            region=os.getenv("LANCEDB_REGION", "us-east-1"),
        )

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

        # TODO(lu) add self.query
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
            except LanceDBClientError as e:
                if "already exists" in str(e):
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
                if total_rows >= self.rows:
                    break
            if total_rows >= self.rows:
                break

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
        try:
            total_rows = table.count_rows()
            for idx in table.list_indices()["indexes"]:
                stats = table.index_stats(idx["index_name"])
                if total_rows != stats["num_indexed_rows"]:
                    print(
                        f"{table.name}: warning: indexing is not complete, query performance may be degraded. "
                        f"total rows: {total_rows} index: {stats}"
                    )
        except Exception as e:
            print(f"{table.name}: failed to check index status: {e}")

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

    def _query(self, table: RemoteTable, nprobes=1):
        try:
            table.search(np.random.standard_normal(1536)).metric("cosine").nprobes(
                nprobes
            ).select(["openai", "title"]).to_list()
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


def run_benchmark(
    dataset: str,
    num_tables: int,
    batch_size: int,
    num_queries: int,
    ingest: bool,
    index: bool,
    prefix: str,
    reset: bool,
    rows: int,
) -> BenchmarkResults:
    benchmark = Benchmark(
        dataset, num_tables, batch_size, num_queries, ingest, index, prefix, reset, rows
    )
    return benchmark.run()


def main():
    parser = argparse.ArgumentParser()
    add_benchmark_args(parser)
    args = parser.parse_args()
    print(args)

    result: BenchmarkResults = run_benchmark(
        args.dataset,
        args.tables,
        args.batch,
        args.queries,
        args.ingest,
        args.index,
        args.prefix,
        args.reset,
        args.rows,
    )
    result.print(True)


if __name__ == "__main__":
    main()
