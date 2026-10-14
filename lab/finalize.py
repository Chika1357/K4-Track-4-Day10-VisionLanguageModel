"""Build submission figures/report after all mandatory experiments have run."""
import json
import argparse
from pathlib import Path

from lab.analyze import analyze_e0, diagnose_coordinates
from lab.plots import plot_results
from lab.summarize import generate_report, load_evidence
from lab.visualize import visualize


def finalize(build_only=False):
    # Guard first: a partial experiment must never become a fabricated final report.
    load_evidence()
    selection = json.loads(Path("report/runs/e4_selection.json").read_text())
    analyze_e0()
    diagnose_coordinates(selection["base_setup"])
    visualize(f"work/{selection['base_setup'].lower()}.jsonl", selection["base_setup"])
    plot_results()
    if Path("report/evidence/e5.json").exists():
        from lab.bonus_analysis import analyze_bonus
        analyze_bonus()
    generate_report()
    if not build_only:
        from scripts.check_submission import run_checks
        checks = run_checks(Path.cwd())
        for status, name, detail in checks:
            print(f"[{status}] {name}: {detail}")
        if any(status != "PASS" for status, _, _ in checks):
            raise RuntimeError("Submission checker has FAIL/WARN; inspect before committing the final checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true",
                        help="Build on server, then sync to the Git repo and run its unchanged checker")
    finalize(parser.parse_args().build_only)
