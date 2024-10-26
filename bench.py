import argparse
import concurrent
import os
import time
from typing import Iterable

from lancedb.remote.errors import LanceDBClientError
from lancedb.remote.table import RemoteTable

import lancedb
import numpy as np
import pyarrow as pa
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

from cloud.benchmark.util import print_percentiles, await_indices


def run_benchmark(
    dataset: str,
    num_tables: int,
    batch_size: int,
    num_queries: int,
    ingest: bool,
    index: bool,
    prefix: str,
):
    db = lancedb.connect(
        uri=os.environ["LANCEDB_DB_URI"],
        api_key=os.environ["LANCEDB_API_KEY"],
        host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
        region=os.getenv("LANCEDB_REGION", "us-east-1"),
    )
    if ingest:
        tables = list(_create_tables(db, num_tables, prefix))
        _ingest(tables, dataset, batch_size)
    else:
        tables = list(_open_tables(db, num_tables, prefix))

    if index:
        _create_vector_index(tables, "openai")

    _query_tables(tables, num_queries)


def _create_tables(
    db: lancedb.LanceDBConnection, num_tables: int, prefix: str
) -> Iterable[RemoteTable]:
    schema = pa.schema(
        [
            pa.field("_id", pa.string()),
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


def _ingest(tables: list[RemoteTable], dataset: str, batch_size: int):
    # TODO: ingest the datasets in parallel in separate processes?
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tables)) as executor:
        futures = []

        for table in tables:
            futures.append(executor.submit(_ingest_table, dataset, table, batch_size))
        [future.result() for future in futures]


def _ingest_table(dataset: str, table: RemoteTable, batch_size: int):
    # todo: support batch size > 1000
    add_times = []
    begin = time.time()
    total_rows = 0
    for batch in tqdm(_convert_dataset(table.schema, dataset)):
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
    print(
        f"{table.name}: ingested {total_rows} rows in {total_s}s. average: {total_rows / total_s:.1f}rows/s"
    )
    print_percentiles(add_times)


def _add_batch(table, batch):
    table.add(batch)


def _split_record_batch(record_batch, batch_size):
    num_rows = record_batch.num_rows
    for i in range(0, num_rows, batch_size):
        yield record_batch.slice(i, min(batch_size, num_rows - i))


def _query_tables(tables: list[RemoteTable], num_queries: int):
    for table in tables:
        _query_table(table, num_queries)


def _create_vector_index(tables: list[RemoteTable], column_name: str):
    for t in tables:
        _create_vector_index_for_table(t, column_name)

    for t in tables:
        start = time.time()
        await_indices(t, 1, ["IVF_PQ"])
        print(f"{t.name}: indexing completed in {int(time.time() - start)}s.")


def _create_vector_index_for_table(table: RemoteTable, column_name: str):
    table.create_index(
        metric="cosine", vector_column_name=column_name, index_type="IVF_PQ"
    )


def _to_fixed_size_array(array, dim):
    return pa.FixedSizeListArray.from_arrays(array.values, dim)


def _convert_dataset(schema, dataset: str) -> Iterable[pa.RecordBatch]:
    for batch in load_dataset(
        dataset,
        download_config=DownloadConfig(num_proc=8, resume_download=True),
        split="train",
    ).data.to_batches():
        yield pa.RecordBatch.from_arrays(
            [
                batch["_id"],
                batch["title"],
                batch["text"],
                _to_fixed_size_array(batch["openai"], 1536),
            ],
            schema=schema,
        )


def _query_table(table, num_queries: int, warmup_queries=100):
    # log a warning if data is not fully indexed
    total_rows = table.count_rows()
    for idx in table.list_indices()["indexes"]:
        stats = table.index_stats(idx["index_name"])
        if total_rows != stats["num_indexed_rows"]:
            print(
                f"{table.name}: warning: indexing is not complete, query performance may be degraded. "
                f"total rows: {total_rows} index: {stats}"
            )

    print(f"{table.name}: starting query test. {num_queries=} {warmup_queries=} {total_rows=}")
    for _ in range(warmup_queries):
        _query(table)

    diffs = []
    begin = time.time()
    for _ in tqdm(range(num_queries)):
        start_time = time.time()
        _query(table)
        elapsed = int((time.time() - start_time) * 1000)
        diffs.append(elapsed)
    total_s = int(time.time() - begin)
    print(f"{table.name}: query average: {num_queries / total_s:.1f}QPS")
    print_percentiles(diffs)


def _query(table: RemoteTable, nprobes=1):
    result = (
        table.search(np.random.standard_normal(1536))
        .metric("cosine")
        .nprobes(nprobes)
        .select(["openai", "title"])
        .to_list()
    )
    return result


def main():
    parser = argparse.ArgumentParser()
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
        default=10,
        help="number of concurrent tables",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1000,
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
    args = parser.parse_args()
    print(args)
    run_benchmark(
        args.dataset,
        args.tables,
        args.batch,
        args.queries,
        args.ingest,
        args.index,
        args.prefix,
    )


if __name__ == "__main__":
    main()
