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

#### Example 1: Each Process Creates a Table and Queries It
To run multiple query benchmarks in parallel processes, you can initiate several benchmarking processes,
each creating and querying its own tables.
The following command demonstrates how to start 4 benchmarking processes,
each querying 4 tables with the table name prefix "my-prefix":
```
uv run bench.py -n 4 -t 4 -p my-prefix -q 10000 -r
```
Parameters:
`-n 4`: Number of benchmarking processes to run
`-t 4`: Number of tables to query per process
`-p my-prefix`: Prefix for the table names
`-q 10000`: Number of queries to run against each table
`-r`: recreate the table and indices if exist

After the initial setup, you can rerun the query performance tests without recreating tables or indexes by using the following command:
```
uv run bench.py -n 4 -t 4 -p my-prefix -q 10000 --no-ingest --no-index
```
Additional Flags:
`--no-ingest`: Skips the table creation step.
`--no-index`: Skips the index creation step.

#### Example 2: Multiple Processes Querying the Same Table
In this scenario, you first create a table with a specified name prefix and the necessary indexes.
Use the following command to set this up:
```
uv run bench.py -t 1 -p my-prefix -q 0 -r
```
Parameters:
`-t 1`: Create one table (the same table will be queried by multiple processes).
`-p my-prefix`: Prefix for the table name.
`-q 0`: No queries are run during the table creation.
`-r`: recreate the table and indices if exist

Once the table is created, you can launch multiple processes to query against the same table.
Each process will run 10,000 queries. Use the following command:
```
uv run bench.py -t 1 -p my-prefix --no-ingest --no-index --query-process 5 -q 10000
```
Parameters:
`--query-process 5`: Specifies that 5 processes will query the same table concurrently.
`--no-ingest`: Skips the table creation step.
`--no-index`: Skips the index creation step.
`-q 10000`: Number of queries each process will run against each table