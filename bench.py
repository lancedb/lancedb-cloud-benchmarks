import argparse
import concurrent
import os
import time
import traceback
from concurrent.futures import wait
from multiprocessing import Pool
from pprint import pprint
from typing import Iterable

from lancedb.remote.errors import LanceDBClientError
from lancedb.remote.table import RemoteTable

import lancedb
import numpy as np
import pyarrow as pa
from datasets import load_dataset, DownloadConfig

from cloud.benchmark.report import (
    set_result,
    save_results,
    get_report_dir,
    generate_report,
    aggregate_results,
)
from src.cloud.benchmark.util import print_percentiles, await_indices, get_percentiles


def run_benchmark(
    dataset: str,
    num_tables: int,
    batch_size: int,
    num_queries: int,
    ingest: bool,
    index: bool,
    prefix: str,
    reset: bool,
    test_run_id: str,
):
    try:
        print(f"starting test run {test_run_id} with prefix {prefix}")
        db = lancedb.connect(
            uri=os.environ["LANCEDB_DB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
            region=os.getenv("LANCEDB_REGION", "us-east-1"),
        )
        results = {
            "params": {
                "test_run_id": test_run_id,
                "dataset": dataset,
                "batch_size": batch_size,
                "num_queries": num_queries,
                "prefix": prefix,
            },
            "tables": {},
        }

        if reset:
            _drop_tables(db, num_tables, prefix)

        if ingest:
            tables = list(_create_tables(db, num_tables, prefix))
            _ingest(tables, dataset, batch_size, results)
        else:
            tables = list(_open_tables(db, num_tables, prefix))

        if index:
            _create_indices(tables, results)

        _query_tables(tables, num_queries, results)

        # pprint(results)
        save_results(test_run_id, prefix, results)

    except Exception as e:
        print(f"benchmark failed with error: {e}")
        print(traceback.format_exc())
        raise


def _create_tables(
    db: lancedb.LanceDBConnection, num_tables: int, prefix: str
) -> Iterable[RemoteTable]:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("openai", pa.list_(pa.float32(), 1536)),
        ]
    )
    for i in range(num_tables):
        table_name = f"{prefix}-{i}"
        try:
            table = db.create_table(
                table_name,
                schema=schema,
            )
        except LanceDBClientError as e:
            if "already exists" in str(e):
                table = db.open_table(table_name)
            else:
                raise

        yield table


def _open_tables(
    db: lancedb.LanceDBConnection, num_tables: int, prefix: str
) -> Iterable[RemoteTable]:
    for i in range(num_tables):
        table_name = f"{prefix}-{i}"
        yield db.open_table(table_name)


def _drop_tables(db, num_tables, prefix):
    tables = list(_open_tables(db, num_tables, prefix))
    for t in tables:
        print(f"dropping table {t.name}")
        db.drop_table(t.name)


def _ingest(tables: list[RemoteTable], dataset: str, batch_size: int, results: dict):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tables)) as executor:
        futures = []

        for table in tables:
            futures.append(
                executor.submit(_ingest_table, dataset, table, batch_size, results)
            )
        r = [future.result() for future in futures]

        total_s = time.time() - start
        total_rows = sum(r)
        rows_s = total_rows / total_s
        print(
            f"ingested {total_rows} rows in {len(tables)} tables in {total_s:.1f}s. average: {rows_s :.1f}rows/s"
        )

        set_result(results, "ingest_time_all_tables", total_s)
        set_result(results, table.name, "ingest_total_rows_all_tables", total_rows)
        set_result(results, table.name, "ingest_avg_rows_s_all_tables", rows_s)


def _ingest_table(
    dataset: str, table: RemoteTable, batch_size: int, results: dict
) -> int:
    add_times = []
    begin = time.time()
    total_rows = 0
    for batch in _convert_dataset(table.schema, dataset, batch_size):
        for slice in _split_record_batch(batch, batch_size):
            start_time = time.time()
            _add_batch(table, batch)
            total_rows += len(batch)
            elapsed = int((time.time() - start_time) * 1000)
            add_times.append(elapsed)
            print(
                f"{table.name}: added batch with size {len(slice)} in {elapsed}ms. rows in table: {table.count_rows()}"
            )

    total_s = int((time.time() - begin))
    rows_s = total_rows / total_s
    print(
        f"{table.name}: ingested {total_rows} rows in {total_s}s. average: {rows_s :.1f}rows/s"
    )
    add_percentiles = get_percentiles(add_times)
    print_percentiles(add_percentiles)

    set_result(results, table.name, "ingest_time_s", total_s)
    set_result(results, table.name, "rows_ingested", total_rows)
    set_result(results, table.name, "ingest_avg_rows_s", rows_s)
    set_result(results, table.name, "ingest_percentiles_ms", add_percentiles)

    return total_rows


def _add_batch(table, batch):
    try:
        table.add(batch)
    except Exception as e:
        print(f"{table.name}: error during add: {e}")


def _split_record_batch(record_batch, batch_size):
    num_rows = record_batch.num_rows
    for i in range(0, num_rows, batch_size):
        yield record_batch.slice(i, min(batch_size, num_rows - i))


def _query_tables(tables: list[RemoteTable], num_queries: int, results: dict):
    num_tables = len(tables)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = []

        for table in tables:
            futures.append(executor.submit(_query_table, table, num_queries, results))
        r = [future.result() for future in futures]

        total_queries = num_queries * num_tables
        total_qps = sum(r)
        print(
            f"completed {total_queries} queries on {num_tables} tables. average: {total_qps:.1f}QPS"
        )
        set_result(results, None, "total_queries", total_queries)
        set_result(results, None, "total_qps", total_qps)


def _await_index(table: RemoteTable, index_type: str, start_time):
    await_indices(table, 1, [index_type])
    print(
        f"{table.name}: {index_type} indexing completed in {int(time.time() - start_time)}s."
    )


def _create_indices(tables: list[RemoteTable], results):
    # create the indices - these will be created async
    table_indices = {}
    for t in tables:
        t.create_index(
            metric="cosine", vector_column_name="openai", index_type="IVF_PQ"
        )
        t.create_scalar_index("id", index_type="BTREE")
        t.create_fts_index("title")
        table_indices[t] = ["IVF_PQ", "FTS", "BTREE"]

    print("waiting for index completion...")
    start = time.time()

    # poll for index completion in parallel to gather accurate indexing time
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tables) * 3) as executor:
        futures = []
        for table, indices in table_indices.items():
            for index in indices:
                futures.append(executor.submit(_await_index(table, index, start)))
        wait(futures)
        total_s = time.time() - start
        print(f"found all indices for {len(tables)} tables in {total_s:.1f}s.")


def _to_fixed_size_array(array, dim):
    return pa.FixedSizeListArray.from_arrays(array.values, dim)


def _convert_dataset(schema, dataset: str, batch_size: int) -> Iterable[pa.RecordBatch]:
    batch_iterator = load_dataset(
        dataset,
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
                _to_fixed_size_array(batch["openai"], 1536),
            ],
            schema=schema,
        )

        if buffer_rows >= batch_size:
            table = pa.Table.from_batches(buffer)
            combined = table.combine_chunks().to_batches(max_chunksize=batch_size)[0]
            buffer.clear()
            buffer_rows = 0
            yield combined
        else:
            buffer.append(rb)
            buffer_rows += len(rb)

    for b in buffer:
        yield b


def _query_table(table: RemoteTable, num_queries: int, results: dict, warmup_queries=1):
    # log a warning if data is not fully indexed
    total_rows = table.count_rows()
    for idx in table.list_indices()["indexes"]:
        stats = table.index_stats(idx["index_name"])
        if total_rows != stats["num_indexed_rows"]:
            print(
                f"{table.name}: warning: indexing is not complete, query performance may be degraded. "
                f"total rows: {total_rows} index: {stats}"
            )

    print(
        f"{table.name}: starting query test. {num_queries=} {warmup_queries=} {total_rows=}"
    )
    for _ in range(warmup_queries):
        _query(table)

    diffs = []
    begin = time.time()
    for _ in range(num_queries):
        start_time = time.time()
        _query(table)
        elapsed = int((time.time() - start_time) * 1000)
        diffs.append(elapsed)
    total_s = int(time.time() - begin)
    qps = num_queries / total_s
    print(f"{table.name}: query count: {num_queries} average: {qps :.1f}QPS")
    percentiles = get_percentiles(diffs)
    print_percentiles(percentiles)

    set_result(results, table.name, "query_total_time_s", total_s)
    set_result(results, table.name, "num_queries", num_queries)
    set_result(results, table.name, "query_avg_qps", qps)
    set_result(results, table.name, "query_percentiles_ms", percentiles)

    return qps


def _query(table: RemoteTable, nprobes=1):
    try:
        table.search(np.random.standard_normal(1536)).metric("cosine").nprobes(
            nprobes
        ).select(["openai", "title"]).to_list()
    except Exception as e:
        print(f"{table.name}: error during query: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="KShivendu/dbpedia-entities-openai-1M",
        help="huggingface dataset name. Only KShivendu/dbpedia-entities-openai-1M is supported currently",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=int,
        default=4,
        help="number of concurrent tables",
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
        help="run ingestion before running queries",
    )
    parser.add_argument(
        "--index",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="run index creation and verification",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="ldb-cloud-benchmarks",
        help="table name prefix",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="number of parallel processes to launch",
    )
    parser.add_argument(
        "-r",
        "--reset",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="drop tables before ingesting",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        default=None,
        help="drop tables before ingesting",
    )
    args = parser.parse_args()
    print(args)

    test_run_id = args.id
    if not test_run_id:
        test_run_id = str(int(time.time()))

    # launch child processes based on configured concurrency
    # each child process will operate on unique tables based on table name prefix
    with Pool(processes=args.concurrency) as pool:
        p_args = [
            (
                args.dataset,
                args.tables,
                args.batch,
                args.queries,
                args.ingest,
                args.index,
                f"{args.prefix}-{i+1}" if args.concurrency > 1 else args.prefix,
                args.reset,
                test_run_id,
            )
            for i in range(args.concurrency)
        ]
        pool.starmap(run_benchmark, p_args)

    # aggregate results from concurrent benchmarks
    aggregate_results(test_run_id)

    # generate the report
    report_file = generate_report(get_report_dir(test_run_id))
    print(f"finished test run {test_run_id}. ")
    print(f"report saved to {report_file}")


if __name__ == "__main__":
    main()
