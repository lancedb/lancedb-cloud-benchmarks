from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Any, List

from lancedb.remote.table import RemoteTable
from lancedb.rerankers import RRFReranker
import numpy as np

QUERY_WORDS = [
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
        words: List[str] = None,
        dim: int = 1536,
        metric: str = "cosine",
        nprobes: int = 1,
        selected_columns: List[str] = None,
        limit: int = 1,
        filter: bool = False,
    ):
        self.words = words or QUERY_WORDS
        self.dim = dim
        self.metric = metric
        self.nprobes = nprobes
        self.selected_columns = selected_columns or ["openai", "title"]
        self.limit = limit
        self.filter = filter

    def query(self, table: RemoteTable, **kwargs) -> Any:
        query = table.search(np.random.standard_normal(self.dim))

        if self.filter:
            filter_text = random.choice(self.words)
            query = query.where(f"text LIKE '%{filter_text}%'", prefilter=True)

        return (
            query.metric(self.metric)
            .nprobes(self.nprobes)
            .select(self.selected_columns)
            .limit(self.limit)
            .to_arrow()
        )


class FTSQuery(Query):
    """Simple full-text search implementation"""

    def __init__(
        self,
        words: List[str] = None,
        column: str = "title",
        selected_columns: List[str] = None,
        limit: int = 1,
    ):
        self.words = words or QUERY_WORDS
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


class HybridQuery(Query):
    """Simple hybrid search implementation"""

    def __init__(
        self,
        text_query: List[str] = None,
        vector_query: List[float] = None,
        metric: str = "cosine",
        column: str = "text",
        selected_columns: List[str] = None,
        limit: int = 1,
    ):
        self.text_query = text_query
        self.vector_query = vector_query
        self.metric = metric
        self.column = column
        self.selected_columns = selected_columns or ["openai", "title"]
        self.limit = limit

    def query(self, table: RemoteTable, **kwargs) -> Any:
        text_query = random.choice(self.words)
        vector_query = np.random.standard_normal(self.dim)
        reranker = RRFReranker()
        return (
            table.search(query_type="hybrid")
            .vector(vector_query)
            .text(text_query)
            .rerank(reranker)
            .select(self.selected_columns)
            .limit(self.limit)
            .to_arrow()
        )
