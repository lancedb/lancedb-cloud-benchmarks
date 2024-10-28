# lancedb-cloud-benchmarks

Benchmarking tools for LanceDB Cloud and LanceDB Enterprise.

### Background 

This benchmark script will download the [dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset,
ingest and index it into N tables in LanceDB Cloud/Enterprise and run vector searches on the tables.

It will report on ingestion time, indexing completion time, and query performance percentiles. 

Further metrics can be gathered by the [LanceDB team](mailto:contact@lancedb.com) upon request.


### Running benchmarks

1. Install uv

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Note: on some systems, you may need to install clang (i.e. `sudo yum install clang`)

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

### Deployment recommendations

To get representative results, it is recommended to run the benchmarking script in a production-like environment with adequate network bandwidth and deployed in a region/datacenter close to the LanceDB endpoint.

For LanceDB Cloud, it is recommended to deploy the benchmarking script in **AWS us-east-1** region.

### Scaling out

At high traffic levels, ingestion and query performance may be limited in a single Python client process. It is possible to scale out
to larger aggregate numbers by using multiple processes or even distributing across multiple VMs. In this case, the result metrics will need to be aggregated
to get the total QPS and throughput.

i.e. Run 4 query benchmarks in separate processes with 4 tables each with table name prefix 'my-prefix' (bash):
```bash
./multi.sh 4 my-prefix "uv run bench.py -t 4 -q 10000 --no-ingest --no-index"
```

This technique can also be used for high-throughput ingestion to multiple tables in parallel using a table prefix per process:
i.e. Ingest to 50 tables in parallel across 10 processes with prefix 'high-throughput' (bash):
```bash
./multi.sh 10 high-throughput "uv run bench.py -t 5"
```
