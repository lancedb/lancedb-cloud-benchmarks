import functools
import time
from typing import List, Optional

import backoff
import numpy as np
from lancedb.remote.table import RemoteTable
import json


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


def log_timer(func, log_func=print):
    """Logs function call with a duration timer"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__

        result = None
        exception = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e

        elapsed = int((time.time() - start_time) * 1000)
        log_args = {
            "function": func_name,
            "duration": elapsed,
            f"{func_name}_duration": elapsed,
        }
        if exception:
            log_args["exception"] = str(exception)

        log_func(
            f"{func_name} {'failed' if exception else 'completed'} "
            f"in {elapsed}ms: {log_args}"
        )

        if exception:
            raise exception

        return result

    return wrapper


def print_percentiles(diffs, percentiles=[50, 90, 99, 100]):
    # TODO(future) average the percentiles to be the result
    for p in percentiles:
        percentile_value = np.percentile(diffs, p)
        print(f"p{p}: {percentile_value:.2f}ms")


class BenchmarkResults:
    def __init__(self):
        self.tables = 0
        self.ingest_rows = 0
        self.ingest_duration_second = 0
        self.ingest_rows_per_second = 0
        self.index_duration_second = 0
        self.total_queries = 0
        self.queries_per_second = 0
        # P50, P90, P95, P99
        # TODO(future) self.query_lantency_percentile = {}

    @staticmethod
    def combine(results: List["BenchmarkResults"]) -> Optional["BenchmarkResults"]:
        """
        Combine multiple benchmark results into a single result.
        Handles parallel execution by taking max durations and summing counts.
        """
        if not results:
            return None

        combined = BenchmarkResults()

        combined.tables = sum(r.tables for r in results)

        combined.ingest_rows = sum(r.ingest_rows for r in results)
        combined.ingest_duration_second = max(
            (r.ingest_duration_second for r in results), default=0
        )

        if combined.ingest_duration_second > 0:
            combined.ingest_rows_per_second = (
                combined.ingest_rows / combined.ingest_duration_second
            )

        combined.index_duration_second = max(
            (r.index_duration_second for r in results), default=0
        )

        combined.total_queries = sum(r.total_queries for r in results)
        combined.queries_per_second = sum(r.queries_per_second for r in results)

        return combined

    def to_json(self) -> str:
        """Convert to JSON string for safe multiprocessing transfer"""
        return json.dumps(
            {
                "tables": self.tables,
                "ingest_rows": self.ingest_rows,
                "ingest_duration_second": self.ingest_duration_second,
                "ingest_rows_per_second": self.ingest_rows_per_second,
                "index_duration_second": self.index_duration_second,
                "total_queries": self.total_queries,
                "queries_per_second": self.queries_per_second,
            }
        )

    @classmethod
    def from_json(cls, json_str: Optional[str]) -> Optional["BenchmarkResults"]:
        """Create from JSON string after multiprocessing transfer"""
        if json_str is None:
            return None
        try:
            data = json.loads(json_str)
            result = cls()
            result.tables = data["tables"]
            result.ingest_rows = data["ingest_rows"]
            result.ingest_duration_second = data["ingest_duration_second"]
            result.ingest_rows_per_second = data["ingest_rows_per_second"]
            result.index_duration_second = data["index_duration_second"]
            result.total_queries = data["total_queries"]
            result.queries_per_second = data["queries_per_second"]
            return result
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON results: {e}")
            return None
        except KeyError as e:
            print(f"Missing key in results data: {e}")
            return None

    def print(self):
        print("\n=== Benchmark Results ===")
        print(f"Number of tables: {self.tables}")
        print("\nIngestion:")
        print(f"  Total rows: {self.ingest_rows:,}")
        print(f"  Duration: {self.ingest_duration_second:.1f}s")
        print(f"  Rows/second: {self.ingest_rows_per_second:.1f}")
        print("\nIndexing:")
        print(f"  Duration: {self.index_duration_second:.1f}s")
        print("\nQueries:")
        print(f"  Total queries: {self.total_queries}")
        print(f"  Queries/second: {self.queries_per_second:.1f}")
