"""Small matplotlib helpers for generated report assets.

One simple figure per call, default style, saved as PNG. When data is
missing, save_text_note can leave a plain-text explanation instead.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files without needing a display
import matplotlib.pyplot as plt


def ensure_parent_dir(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_bar_chart(labels, values, title, ylabel, output_path) -> Path:
    output_path = ensure_parent_dir(output_path)
    figure = plt.figure(figsize=(max(6.0, 0.8 * len(labels)), 4.5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_grouped_bar_chart(group_labels, series_by_name, title, ylabel,
                           output_path) -> Path:
    """series_by_name maps a series name to one value per group."""
    output_path = ensure_parent_dir(output_path)
    figure = plt.figure(figsize=(max(6.0, 0.9 * len(group_labels)), 4.5))
    names = list(series_by_name)
    width = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        positions = [g + i * width for g in range(len(group_labels))]
        plt.bar(positions, series_by_name[name], width=width, label=name)
    centers = [g + 0.4 - width / 2 for g in range(len(group_labels))]
    plt.xticks(centers, group_labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_line_chart(x_values, y_values, title, xlabel, ylabel, output_path) -> Path:
    output_path = ensure_parent_dir(output_path)
    figure = plt.figure(figsize=(7.0, 4.5))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_text_note(text, output_path) -> Path:
    """Plain text note, used when a plot cannot be generated."""
    output_path = ensure_parent_dir(output_path)
    output_path.write_text(text if text.endswith("\n") else text + "\n")
    return output_path
