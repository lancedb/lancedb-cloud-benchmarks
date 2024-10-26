import functools
import time
from typing import Optional

import backoff
import numpy as np
from lancedb.remote.table import RemoteTable


@backoff.on_exception(backoff.constant, ValueError, max_time=600, interval=10)
def await_indices(
    table: RemoteTable, count: int = 1, index_types: Optional[list[str]] = []
) -> list[dict]:
    """poll for all indices to be created on the table"""
    indices = table.list_indices()
    print(f"current indices for table {table}: {indices}")
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
            raise ValueError(
                f"still waiting for unindexed rows to be 0 (current: {u})"
            )

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
    for p in percentiles:
        percentile_value = np.percentile(diffs, p)
        print(f"p{p}: {percentile_value:.2f}ms")
