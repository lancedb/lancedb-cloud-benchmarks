from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Any, List

from lancedb.remote.table import RemoteTable
import numpy as np


class QueryType(Enum):
    VECTOR = "vector"
    VECTOR_WITH_FILTER = "vector_with_filter"
    FTS = "fts"
    HYBRID = "hybrid"


class Query(ABC):
    """Abstract base class for different query types"""

    @abstractmethod
    def query(self, table: RemoteTable, **kwargs) -> Any:
        """Execute the query on the given table"""
        pass


class VectorQuery(Query):
    def __init__(
        self,
        dim: int = 1536,
        metric: str = "cosine",
        nprobes: int = 1,
        selected_columns: List[str] = None,
        limit: int = 1,
        filter: bool = False,
    ):
        self.dim = dim
        self.metric = metric
        self.nprobes = nprobes
        self.selected_columns = selected_columns or ["openai", "title"]
        self.limit = limit
        self.filter = filter

    def query(self, table: RemoteTable, **kwargs) -> Any:
        query = (
            table.search(np.random.standard_normal(self.dim))
            .metric(self.metric)
            .nprobes(self.nprobes)
        )

        if self.filter and "total_rows" in kwargs:
            total_rows = kwargs["total_rows"]
            random_value = random.randint(0, total_rows)

            if random_value < total_rows // 2:
                query = query.where(f"_id > {random_value}")
            else:
                query = query.where(f"_id < {random_value}")

        return query.select(self.selected_columns).limit(self.limit).to_arrow()


class FTSQuery(Query):
    """Simple full-text search implementation"""

    def __init__(
        self,
        words: List[str] = None,
        column: str = "title",
        selected_columns: List[str] = None,
        limit: int = 1,
    ):
        self.words = words or [
            # Common nouns
            "University",
            "Institute",
            "School",
            "Museum",
            "Library",
            "History",
            "Science",
            "Art",
            "Literature",
            "Philosophy",
            # Locations
            "America",
            "Europe",
            "Asia",
            "China",
            "India",
            "Japan",
            "Russia",
            "Germany",
            "France",
            "England",
            # Organizations
            "Company",
            "Corporation",
            "Association",
            "Society",
            "Foundation",
            # Fields
            "Technology",
            "Engineering",
            "Medicine",
            "Economics",
            "Politics",
        ]
        self.column = column
        self.selected_columns = selected_columns or ["title"]
        self.limit = limit

    def query(self, table: RemoteTable, **kwargs) -> Any:
        query_text = random.choice(self.words)
        return (
            table.search(query_text, query_type="fts")
            .select(self.selected_columns)
            .limit(self.limit)
            .to_arrow()
        )
