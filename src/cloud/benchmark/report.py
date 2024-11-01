import json
import os
from pathlib import Path
from typing import Optional

from jinja2 import Template


def generate_report(test_run_id: str) -> str:
    report_dir = get_report_dir(test_run_id)

    # copy template to report dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template = os.path.join(current_dir, "report/index.html")

    new_path = report_dir / "index.html"

    # load aggregated data from report dir
    with open(report_dir / "aggregated.json", "r") as f:
        aggregated = json.load(f)

    with open(template, "r+") as t:
        template = Template(t.read())
        html = template.render(aggregated)
        with open(new_path, "w+") as o:
            o.write(html)

    return str(new_path)


def set_result(results: dict, table_name: Optional[str], key: str, val):
    if not table_name:
        results[key] = val
        return
    if table_name not in results:
        results["tables"][table_name] = {}
    results["tables"][table_name][key] = val


def save_results(test_run_id: str, prefix: str, results: dict):
    result_dir = get_report_dir(test_run_id)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{prefix}.json"
    with open(result_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"results saved to {result_file}")


def get_report_dir(
    test_run_id: str, base_path: str = "/tmp/lancedb-cloud-benchmarks/results"
) -> Path:
    result_dir = Path(base_path) / test_run_id
    return result_dir


def aggregate_results(test_run_id: str, out_file_name: str = "aggregated.json") -> str:
    """Aggregate the results from all processes and store it in the output file"""
    report_dir = get_report_dir(test_run_id)
    out_file = os.path.join(report_dir, out_file_name)
    merged = {
        "processes": {},
    }

    total_qps = 0
    total_queries = 0
    total_rows = 0
    total_rows_s = 0
    params = None
    for f in os.listdir(report_dir):
        if f.endswith(".json"):
            file_path = os.path.join(report_dir, f)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    merged["processes"][data["params"]["prefix"]] = data
                    params = data["params"]

                    # aggregate the results for all processes and tables
                    total_qps += data.get("total_qps", 0)
                    total_queries += data.get("total_queries", 0)
                    total_rows += data.get("ingest_total_rows_all_tables", 0)
                    total_rows_s += data.get("ingest_avg_rows_s_all_tables", 0)

                except json.JSONDecodeError as e:
                    print(f"error reading {f}: {e}")

    merged["params"] = params
    merged["aggregated"] = {
        "qps": total_qps,
        "queries": total_queries,
        "ingestion_rows_s": total_rows_s,
        "ingested_rows": total_rows,
    }

    with open(out_file, "w") as outfile:
        json.dump(merged, outfile, indent=4)

    print(f"aggregated results saved to {out_file}")
    return str(out_file)
