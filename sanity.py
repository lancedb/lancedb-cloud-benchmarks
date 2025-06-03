from bench import Benchmark, QueryType, add_benchmark_args
import argparse
import time


def get_default_args():
    """Get default arguments from the benchmark argument parser"""
    parser = argparse.ArgumentParser()
    add_benchmark_args(parser)
    default_args = parser.parse_args([])
    args_dict = vars(default_args)

    # Map parser argument names to Benchmark parameter names
    return {
        "dataset": args_dict["dataset"],
        "num_tables": args_dict["tables"],
        "batch_size": args_dict["batch"],
        "num_queries": args_dict["queries"],
        "query_type": args_dict["query_type"],
        "ingest": args_dict["ingest"],
        "index": args_dict["index"],
        "prefix": args_dict["prefix"],
        "reset": args_dict["reset"],
        "size": args_dict["size"],
    }


def run_benchmark(benchmark_args: dict) -> None:
    print(f"\nRunning sanity test with args: {benchmark_args}")
    benchmark = Benchmark(**benchmark_args)
    benchmark.run()
    print("Sanity test passed")


def main():
    # Setting for sanity runs
    batch_size = 100
    dataset_size = 2000
    num_queries = 3  # Number of queries per run

    base_args = get_default_args()

    # Override only the parameters we want to change
    base_args.update(
        {
            "num_tables": 1,
            "batch_size": batch_size,
            "size": dataset_size,
            "prefix": f"sanity-test-{int(time.time())}",
        }
    )

    print("=== Starting Sanity Test ===")
    print(f"Using prefix: {base_args['prefix']}")

    try:
        # Step 1: Ingest and index data
        print("\n=== Step 1: Data Ingestion and Indexing ===")
        ingest_args = base_args.copy()
        ingest_args.update(
            {
                "num_queries": 0,  # No queries during ingestion
                "ingest": True,
                "index": True,
                "reset": True,
            }
        )
        run_benchmark(ingest_args)

        # Step 2: Run queries for each query type
        print("\n=== Step 2: Running Queries ===")

        query_args = base_args.copy()
        query_args.update(
            {
                "num_queries": num_queries,
                "ingest": False,  # Skip ingestion
                "index": False,  # Skip indexing
                "reset": False,
            }
        )

        for query_type in QueryType:
            print(f"\n--- Testing {query_type.value} queries ---")

            run_args = query_args.copy()
            run_args["query_type"] = query_type.value
            run_benchmark(run_args)

        print("\n=== Sanity Test Completed ===")
    except Exception as e:
        print("\n=== Sanity Test Failed ===")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
