"""Generate report-ready tables and plots from existing experiment CSVs.

These are generated assets for the later report, NOT the report itself.
Missing inputs never crash the generation — a note file is written instead.
"""

from pathlib import Path

from src.experiments.plot_utils import (
    ensure_parent_dir,
    save_bar_chart,
    save_text_note,
)
from src.experiments.summary import (
    ACKLEY_SUMMARY_FIELDNAMES,
    CVRP_SUMMARY_FIELDNAMES,
    read_csv_rows,
    safe_float,
    summarize_ackley_rows,
    summarize_cvrp_rows,
    write_summary_csv,
)

CVRP_TABLE_COLUMNS = [
    "instance", "algorithm", "runs", "feasible_runs", "best_cost",
    "mean_cost", "std_cost", "best_gap_percent", "mean_elapsed_time",
]

ACKLEY_TABLE_COLUMNS = [
    "algorithm", "runs", "best_value", "mean_best_value", "std_best_value",
    "mean_distance_from_origin", "mean_elapsed_time",
]

GP_GEP_TABLE_COLUMNS = [
    "algorithm", "seed", "train_fitness", "eval_fitness", "eval_solved_count",
    "eval_puzzle_count", "eval_total_expanded_nodes", "eval_total_cost",
    "best_expression",
]


# ---------- small helpers ----------

def format_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def rows_to_markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown_table(path, rows, columns, title=None) -> Path:
    path = ensure_parent_dir(path)
    text = rows_to_markdown_table(rows, columns)
    if title:
        text = f"# {title}\n\n" + text
    path.write_text(text + "\n")
    return path


def load_rows_if_exists(path) -> list[dict]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    return read_csv_rows(path)


def _bar_chart_if_numeric(rows, label_func, field, title, ylabel, output_path):
    """Bar chart over the rows whose field is numeric, or None if none are."""
    pairs = [(label_func(row), safe_float(row.get(field))) for row in rows]
    pairs = [(label, value) for label, value in pairs if value is not None]
    if not pairs:
        return None
    labels = [label for label, _ in pairs]
    values = [value for _, value in pairs]
    return save_bar_chart(labels, values, title, ylabel, output_path)


# ---------- CVRP assets ----------

def make_cvrp_report_assets(raw_csv, summary_csv, output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    created = []
    raw_rows = load_rows_if_exists(raw_csv)
    summary_rows = load_rows_if_exists(summary_csv)

    if not raw_rows and not summary_rows:
        created.append(save_text_note(
            "CVRP data was not found, so no CVRP assets were generated.\n"
            f"raw csv: {raw_csv}\nsummary csv: {summary_csv}",
            output_dir / "cvrp_data_missing.txt"))
        return created

    if not summary_rows:
        summary_rows = summarize_cvrp_rows(raw_rows)
        generated = output_dir / "cvrp_summary_generated.csv"
        write_summary_csv(generated, summary_rows, CVRP_SUMMARY_FIELDNAMES)
        created.append(generated)

    created.append(write_markdown_table(
        output_dir / "cvrp_summary_table.md", summary_rows, CVRP_TABLE_COLUMNS,
        title="CVRP summary (generated asset, not the final report)"))

    def label(row):
        return f"{row.get('instance', '')} / {row.get('algorithm', '')}"

    for field, title, ylabel, filename in [
        ("best_cost", "CVRP best cost", "best cost", "cvrp_best_cost.png"),
        ("best_gap_percent", "CVRP gap to BKS", "gap percent", "cvrp_gap_percent.png"),
        ("mean_elapsed_time", "CVRP mean runtime", "seconds", "cvrp_runtime.png"),
    ]:
        chart = _bar_chart_if_numeric(summary_rows, label, field, title, ylabel,
                                      output_dir / filename)
        if chart is not None:
            created.append(chart)
    return created


# ---------- Ackley assets ----------

def make_ackley_report_assets(raw_csv, summary_csv, output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    created = []
    raw_rows = load_rows_if_exists(raw_csv)
    summary_rows = load_rows_if_exists(summary_csv)

    if not raw_rows and not summary_rows:
        created.append(save_text_note(
            "Ackley data was not found, so no Ackley assets were generated.\n"
            f"raw csv: {raw_csv}\nsummary csv: {summary_csv}",
            output_dir / "ackley_data_missing.txt"))
        return created

    if not summary_rows:
        summary_rows = summarize_ackley_rows(raw_rows)
        generated = output_dir / "ackley_summary_generated.csv"
        write_summary_csv(generated, summary_rows, ACKLEY_SUMMARY_FIELDNAMES)
        created.append(generated)

    created.append(write_markdown_table(
        output_dir / "ackley_summary_table.md", summary_rows, ACKLEY_TABLE_COLUMNS,
        title="Ackley summary (generated asset, not the final report)"))

    def label(row):
        return row.get("algorithm", "")

    for field, title, ylabel, filename in [
        ("best_value", "Ackley best value", "best value", "ackley_best_value.png"),
        ("mean_distance_from_origin", "Ackley distance from origin",
         "mean distance", "ackley_distance.png"),
        ("mean_elapsed_time", "Ackley mean runtime", "seconds", "ackley_runtime.png"),
    ]:
        chart = _bar_chart_if_numeric(summary_rows, label, field, title, ylabel,
                                      output_dir / filename)
        if chart is not None:
            created.append(chart)
    return created


# ---------- GP/GEP assets ----------

def make_gp_gep_report_assets(comparison_csv, comparison_summary_txt,
                              output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    created = []
    rows = load_rows_if_exists(comparison_csv)

    if not rows:
        created.append(save_text_note(
            "GP/GEP comparison data was not found, so no GP/GEP assets were "
            f"generated.\ncomparison csv: {comparison_csv}",
            output_dir / "gp_gep_data_missing.txt"))
        return created

    created.append(write_markdown_table(
        output_dir / "gp_gep_comparison_table.md", rows, GP_GEP_TABLE_COLUMNS,
        title="GP vs GEP comparison (generated asset, not the final report)"))

    def label(row):
        return f"{row.get('algorithm', '')} / seed {row.get('seed', '')}"

    for field, title, ylabel, filename in [
        ("eval_fitness", "GP vs GEP eval fitness", "fitness", "gp_gep_eval_fitness.png"),
        ("eval_total_expanded_nodes", "GP vs GEP expanded nodes",
         "expanded nodes", "gp_gep_expanded_nodes.png"),
    ]:
        chart = _bar_chart_if_numeric(rows, label, field, title, ylabel,
                                      output_dir / filename)
        if chart is not None:
            created.append(chart)

    if comparison_summary_txt and Path(comparison_summary_txt).exists():
        created.append(save_text_note(Path(comparison_summary_txt).read_text(),
                                      output_dir / "gp_gep_summary_note.txt"))
    else:
        created.append(save_text_note(
            f"GP/GEP summary txt was not found: {comparison_summary_txt}",
            output_dir / "gp_gep_summary_note.txt"))
    return created


# ---------- one entry point ----------

def generate_report_assets(cvrp_raw=None, cvrp_summary=None, ackley_raw=None,
                           ackley_summary=None, gp_gep_csv=None,
                           gp_gep_summary=None,
                           output_dir="results/report_assets") -> list[Path]:
    created = []
    if cvrp_raw or cvrp_summary:
        created.extend(make_cvrp_report_assets(cvrp_raw, cvrp_summary, output_dir))
    if ackley_raw or ackley_summary:
        created.extend(make_ackley_report_assets(ackley_raw, ackley_summary, output_dir))
    if gp_gep_csv or gp_gep_summary:
        created.extend(make_gp_gep_report_assets(gp_gep_csv, gp_gep_summary, output_dir))
    return created
