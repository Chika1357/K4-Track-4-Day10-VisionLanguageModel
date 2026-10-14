"""Build submission figures/report after all mandatory experiments have run."""
import json
import subprocess
import sys
from pathlib import Path

from lab.analyze import analyze_e0, diagnose_coordinates
from lab.plots import plot_results
from lab.summarize import generate_report, load_evidence
from lab.visualize import visualize


def finalize():
    # Guard first: a partial experiment must never become a fabricated final report.
    load_evidence()
    selection = json.loads(Path("report/runs/e4_selection.json").read_text())
    analyze_e0()
    diagnose_coordinates(selection["base_setup"])
    visualize(f"work/{selection['base_setup'].lower()}.jsonl", selection["base_setup"])
    plot_results()
    generate_report()
    result = subprocess.run([sys.executable, "scripts/check_submission.py"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Submission checker failed; inspect output before committing the final checkpoint")


if __name__ == "__main__":
    finalize()
