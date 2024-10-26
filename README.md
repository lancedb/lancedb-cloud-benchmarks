# lancedb-cloud-benchmarks

Benchmarking tools for LanceDB Cloud and LanceDB Enterprise.

### Background 

This benchmark script will download the [dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset,
ingest and index it in LanceDB Cloud/Enterprise and run vector searches on the table.

It will report on ingestion time, indexing completion time, and query performance percentiles. 

Further metrics can be gathered by the [LanceDB team](mailto:contact@lancedb.com) upon request.


### Running benchmarks

1. Install uv

`curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Configure environment
```
export LANCEDB_API_KEY=<your api key>`
export LANCEDB_DB_URI=<your db uri from lancedb cloud console, i.e. "db://mydb-d5ac3e">`
export LANCEDB_HOST_OVERRIDE=<optional uri if using lancedb enterprise>`
```

3. Run the benchmark

`uv run bench.py`

### Examples

Ingest the dataset into 4 tables and run 10k queries with a custom table prefix: 

`uv run bench.py -t 4 -q 10000 -p mytable`

Run query benchmark only against existing tables:

`uv run bench.py -t 4 -q 10000 --no-ingest --no-index`


### Help
Run `uv run bench.py --help`