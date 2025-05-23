import lancedb
import numpy as np
import time
import os

# Constants
VECTOR_DIM = 1024
TABLE_ROWS = 1000
UPDATE_ROWS = 3000
SEARCH_TIMES = 30


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


def main(table_name):
    # Connect to the database
    azure_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    storage_options = None
    if azure_account_name:
        storage_options = {"azure_storage_account_name": azure_account_name}

    db = lancedb.connect(
        uri=os.environ["LANCEDB_DB_URI"],
        api_key=os.environ["LANCEDB_API_KEY"],
        host_override=os.getenv("LANCEDB_HOST_OVERRIDE"),
        region=os.getenv("LANCEDB_REGION", "us-east-1"),
        storage_options=storage_options,
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
    print("Creating indices...")
    table.wait_for_index(index_names=["vector_idx"])
    print("Indices created")

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
            print(f"\nSearch {i + 1} results ({vector_column}):")
            print(f"Search time: {search_time:.4f} seconds")

        average_search_time = total_search_time / SEARCH_TIMES
        print(
            f"\nAverage search time for {vector_column}: {average_search_time:.4f} seconds"
        )
    db.drop_table(table_name)


if __name__ == "__main__":
    table_name = f"sanity2-test-{int(time.time())}"
    main(table_name)
