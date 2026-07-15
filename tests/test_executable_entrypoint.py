"""Stage 12-A: the unified executable entry point and run wrappers."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_a3():
    spec = importlib.util.spec_from_file_location("a3_module", REPO_ROOT / "a3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "a3.py" in text
    assert "Executable / run commands" in text


# ---- Stage 12-D2: frozen (PyInstaller) mode compatibility ----

def test_project_root_in_frozen_dist_layout(monkeypatch, tmp_path):
    # dist/a3.exe must resolve the project root to the parent of dist/
    a3 = load_a3()
    fake_root = tmp_path / "ai-lab-assignment3"
    (fake_root / "dist").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_root / "dist" / "a3.exe"))
    assert a3.project_root() == fake_root


def test_project_root_unfrozen_is_repo_root():
    a3 = load_a3()
    assert a3.project_root() == REPO_ROOT


def test_frozen_runner_never_returns_the_exe(monkeypatch, tmp_path):
    a3 = load_a3()
    fake_root = tmp_path / "proj"
    (fake_root / "dist").mkdir(parents=True)
    exe = fake_root / "dist" / "a3.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    runner = a3.find_python_runner(fake_root)  # no .venv there -> fallback
    assert str(exe) not in runner
    assert runner  # some python command is always returned


def test_runner_prefers_project_venv():
    a3 = load_a3()
    runner = a3.find_python_runner(REPO_ROOT)
    assert runner == [str(REPO_ROOT / ".venv" / "bin" / "python")] or \
        runner == [sys.executable]


def test_report_info_paths_use_project_root():
    # the module-level REPO_ROOT (used by report-info) must be the repo root
    a3 = load_a3()
    assert a3.REPO_ROOT == REPO_ROOT
    assert (a3.REPO_ROOT / "report" / "assignment3_report.pdf").exists()


def test_no_generated_build_artifacts_committed():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, cwd=REPO_ROOT).stdout.splitlines()
    assert not any(p.startswith(("dist/", "build/")) or p.endswith(".spec")
                   or p.endswith(".exe") for p in tracked)
