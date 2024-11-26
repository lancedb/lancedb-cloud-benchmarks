from typing import Optional
import lancedb
import numpy as np
import time
import sys
import backoff
from lancedb.remote.table import RemoteTable
import os

# Constants
VECTOR_DIM = 1024
TABLE_ROWS = 1000
UPDATE_ROWS = 3000
SEARCH_TIMES = 1000


def create_random_vector(dim):
    return np.random.rand(dim).tolist()


def perform_search(table, vector_column, dim):
    search_vector = create_random_vector(dim)
    start_time = time.time()
    results = (
        table.search(search_vector, vector_column_name=vector_column)
        .limit(5)
        .to_pandas()
    )
    end_time = time.time()
    search_time = end_time - start_time
    return results, search_time


@backoff.on_exception(
    backoff.constant, ValueError, max_time=600, interval=10, logger=None
)
def await_indices(
    table: RemoteTable, count: int = 1, index_types: Optional[list[str]] = []
) -> list[dict]:
    """poll for all indices to be created on the table"""
    indices = table.list_indices()
    # print(f"current indices for table {table}: {indices}")
    result_indices = []
    for index in indices["indexes"]:
        if not index["index_name"]:
            raise ValueError("still waiting for index creation")
        result_indices.append(index)

    if not result_indices:
        raise ValueError("still waiting for index creation")

    if len(result_indices) < count:
        raise ValueError(
            f"still waiting for more indices "
            f"(current: {len(result_indices)}, desired: {count})"
        )

    index_names = [n["index_name"] for n in result_indices]
    stats = [table.index_stats(n) for n in index_names]
    if index_types:
        types = [s["index_type"] for s in stats]
        for t in index_types:
            if t not in types:
                raise ValueError(
                    f"still waiting for correct index type "
                    f"(current: {types}, desired: {index_types})"
                )

    unindexed_rows = [s["num_unindexed_rows"] for s in stats]
    for u in unindexed_rows:
        if u != 0:
            raise ValueError(f"still waiting for unindexed rows to be 0 (current: {u})")

    return result_indices


def main(table_name):
    # Connect to the database
    db = lancedb.connect(
        uri=os.environ["LANCEDB_DB_URI"],
        api_key=os.environ["LANCEDB_API_KEY"],
        host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
        region=os.getenv("LANCEDB_REGION", "us-east-1"),
    )

    if table_name in db.table_names():
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'")
    else:
        # Create table with two vector columns
        data = [
            {"id": i, "vector": create_random_vector(VECTOR_DIM)}
            for i in range(TABLE_ROWS)
        ]
        table = db.create_table(table_name, data)
        print(
            f"Created new table '{table_name}' with {TABLE_ROWS} rows and vector columns"
        )

    table.create_index()
    await_indices(table, 1, ["IVF_PQ"])

    # Update table
    update_data = [
        {"id": i, "vector": create_random_vector(VECTOR_DIM)}
        for i in range(TABLE_ROWS, TABLE_ROWS + UPDATE_ROWS)
    ]
    table.add(update_data)
    print(f"Added {UPDATE_ROWS} new rows")

    # Perform multiple searches on both vector columns
    vector_configs = [
        ("vector", VECTOR_DIM),
    ]

    for vector_column, dim in vector_configs:
        print(
            f"\nPerforming {SEARCH_TIMES} searches on {vector_column} (dimension: {dim}):"
        )
        total_search_time = 0

        for i in range(SEARCH_TIMES):
            results, search_time = perform_search(table, vector_column, dim)
            total_search_time += search_time
            print(f"\nSearch {i+1} results ({vector_column}):")
            print(f"Search time: {search_time:.4f} seconds")

        average_search_time = total_search_time / SEARCH_TIMES
        print(
            f"\nAverage search time for {vector_column}: {average_search_time:.4f} seconds"
        )

    db.drop_table(table_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <table_name>")
        sys.exit(1)

    table_name = sys.argv[1]
    main(table_name)
