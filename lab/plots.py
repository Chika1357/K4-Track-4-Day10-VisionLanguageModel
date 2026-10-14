"""Generate exportable charts solely from actual evidence and training logs."""
import json
from pathlib import Path


def plot_results(report_root="report", work_root="work"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, work = Path(report_root), Path(work_root)
    paths = [report/"evidence"/f"{name}.json"
             for name in ("e0", "e1", "e2", "e3", "e4_rank_4", "e4_rank_64")]
    evidence = [json.loads(path.read_text()) for path in paths]
    labels = [item["name"] for item in evidence]
    means = [item["micro"]["f1"] for item in evidence]
    # Percentile CI can occasionally exclude the point estimate; draw absolute bounds.
    figure, axis = plt.subplots(figsize=(10, 4.5))
    positions = list(range(len(evidence)))
    axis.scatter(positions, means, s=50, color="#2456a6", zorder=3)
    for x, item in zip(positions, evidence):
        low, high = item["f1_ci95"]
        axis.vlines(x, low, high, color="#2456a6", linewidth=2)
        axis.hlines([low, high], x-.08, x+.08, color="#2456a6")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Micro F1 @ IoU 0.5")
    axis.set_ylim(0, 1)
    axis.set_title("Test results with assignment bootstrap CI95 (311 images)")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    (report/"figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(report/"figures/f1_comparison.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name in ("e1", "e2", "e3", "e4_rank_4", "e4_rank_64"):
        rows = [json.loads(line) for line in (work/name/"train.jsonl").read_text().splitlines()]
        axis.plot([r["step"] for r in rows], [r["loss"] for r in rows], label=name, alpha=.75)
    axis.set_xlabel("Successful optimizer updates")
    axis.set_ylabel("Answer-only cross-entropy loss")
    axis.set_title("Actual training logs; no smoothing")
    axis.legend()
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(report/"figures/training_loss.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    plot_results()
