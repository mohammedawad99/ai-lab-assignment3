"""Stage 12-A: the unified executable entry point and run wrappers."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_a3(*args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "a3.py"), *args],
                          capture_output=True, text=True, cwd=REPO_ROOT)


def test_entrypoint_exists():
    assert (REPO_ROOT / "a3.py").exists()


def test_help_exits_zero_and_lists_commands():
    result = run_a3("--help")
    assert result.returncode == 0, result.stderr
    for command in ("sanity", "audit", "test", "report-info",
                    "cvrp-example", "rushhour-hard-summary"):
        assert command in result.stdout


def test_report_info_exits_zero():
    result = run_a3("report-info")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "report PDF" in result.stdout
    assert "report/evidence" in result.stdout


def test_sanity_runs_the_tiny_example():
    result = run_a3("sanity")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "80.64" in result.stdout
    assert "feasible: yes" in result.stdout


def test_rushhour_hard_summary_prints_committed_evidence():
    result = run_a3("rushhour-hard-summary")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "blocker_depth" in result.stdout


def test_wrappers_and_build_scripts_exist():
    assert (REPO_ROOT / "run_a3.sh").exists()
    assert (REPO_ROOT / "run_a3.bat").exists()
    assert (REPO_ROOT / "scripts" / "build_exe_unix.sh").exists()
    assert (REPO_ROOT / "scripts" / "build_exe_windows.bat").exists()


def test_readme_documents_the_entrypoint():
    text = (REPO_ROOT / "README.md").read_text()
    assert "a3.py" in text
    assert "Executable / run commands" in text


def test_no_generated_build_artifacts_committed():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, cwd=REPO_ROOT).stdout.splitlines()
    assert not any(p.startswith(("dist/", "build/")) or p.endswith(".spec")
                   or p.endswith(".exe") for p in tracked)
