import functools
import time
from typing import List, Optional

import numpy as np

import backoff
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


class BenchmarkResults:
    def __init__(self):
        self.tables = 0
        self.ingest_rows = 0
        self.ingest_duration_second = 0
        self.ingest_rows_per_second = 0
        self.index_duration_second = 0
        self.total_queries = 0
        self.queries_per_second = 0
        self.query_latencies = []
        self.ingest_latencies = []
        # Keep these for final results
        self.ingest_latency_p50 = 0
        self.ingest_latency_p90 = 0
        self.ingest_latency_p95 = 0
        self.ingest_latency_p99 = 0
        self.query_latency_p50 = 0
        self.query_latency_p90 = 0
        self.query_latency_p95 = 0
        self.query_latency_p99 = 0

    @staticmethod
    def combine(results: List["BenchmarkResults"]) -> Optional["BenchmarkResults"]:
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

        # Combine all latency measurements
        for r in results:
            combined.query_latencies.extend(r.query_latencies)
            combined.ingest_latencies.extend(r.ingest_latencies)

        # Calculate percentiles from combined measurements
        combined._update_latencies(combined.query_latencies, "query")
        combined._update_latencies(combined.ingest_latencies, "ingest")

        return combined

    def _update_latencies(self, latencies: list, type: str):
        """
        Update percentile values based on latency measurements.
        Args:
            latencies: List of latency measurements
            type: Either "query" or "ingest"
        """
        if not latencies:
            return

        if type == "query":
            self.query_latency_p50 = np.percentile(latencies, 50)
            self.query_latency_p90 = np.percentile(latencies, 90)
            self.query_latency_p95 = np.percentile(latencies, 95)
            self.query_latency_p99 = np.percentile(latencies, 99)
        elif type == "ingest":
            self.ingest_latency_p50 = np.percentile(latencies, 50)
            self.ingest_latency_p90 = np.percentile(latencies, 90)
            self.ingest_latency_p95 = np.percentile(latencies, 95)
            self.ingest_latency_p99 = np.percentile(latencies, 99)

    def to_json(self) -> str:
        return json.dumps(
            {
                "tables": self.tables,
                "ingest_rows": self.ingest_rows,
                "ingest_duration_second": self.ingest_duration_second,
                "ingest_rows_per_second": self.ingest_rows_per_second,
                "index_duration_second": self.index_duration_second,
                "total_queries": self.total_queries,
                "queries_per_second": self.queries_per_second,
                "query_latencies": self.query_latencies,
                "ingest_latencies": self.ingest_latencies,
                "ingest_latency_p50": self.ingest_latency_p50,
                "ingest_latency_p90": self.ingest_latency_p90,
                "ingest_latency_p95": self.ingest_latency_p95,
                "ingest_latency_p99": self.ingest_latency_p99,
                "query_latency_p50": self.query_latency_p50,
                "query_latency_p90": self.query_latency_p90,
                "query_latency_p95": self.query_latency_p95,
                "query_latency_p99": self.query_latency_p99,
            }
        )

    @classmethod
    def from_json(cls, json_str: Optional[str]) -> Optional["BenchmarkResults"]:
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
            result.query_latencies = data["query_latencies"]
            result.ingest_latencies = data["ingest_latencies"]
            result.ingest_latency_p50 = data["ingest_latency_p50"]
            result.ingest_latency_p90 = data["ingest_latency_p90"]
            result.ingest_latency_p95 = data["ingest_latency_p95"]
            result.ingest_latency_p99 = data["ingest_latency_p99"]
            result.query_latency_p50 = data["query_latency_p50"]
            result.query_latency_p90 = data["query_latency_p90"]
            result.query_latency_p95 = data["query_latency_p95"]
            result.query_latency_p99 = data["query_latency_p99"]
            return result
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON results: {e}")
            return None
        except KeyError as e:
            print(f"Missing key in results data: {e}")
            return None

    def print(self, update_latencies: bool = False):
        print("\n=== Benchmark Results ===")
        print(f"Number of tables: {self.tables}")
        print("\nIngestion:")
        print(f"  Total rows: {self.ingest_rows:,}")
        print(f"  Duration: {self.ingest_duration_second:.1f}s")
        print(f"  Rows/second: {self.ingest_rows_per_second:.1f}")
        print("  Latencies per batch:")
        print(f"    p50: {self.ingest_latency_p50:.2f}ms")
        print(f"    p90: {self.ingest_latency_p90:.2f}ms")
        print(f"    p95: {self.ingest_latency_p95:.2f}ms")
        print(f"    p99: {self.ingest_latency_p99:.2f}ms")
        print("\nIndexing:")
        print(f"  Duration: {self.index_duration_second:.1f}s")
        print("\nQueries:")
        print(f"  Total queries: {self.total_queries}")
        print(f"  Queries/second: {self.queries_per_second:.1f}")
        print("  Latencies per query:")
        print(f"    p50: {self.query_latency_p50:.2f}ms")
        print(f"    p90: {self.query_latency_p90:.2f}ms")
        print(f"    p95: {self.query_latency_p95:.2f}ms")
        print(f"    p99: {self.query_latency_p99:.2f}ms")
        print("\n=== Benchmark Results End ===")
