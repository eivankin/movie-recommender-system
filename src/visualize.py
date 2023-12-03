import matplotlib.pyplot as plt

from src.evaluator import MetricReport


def plot_metric(
    metric_name: str, train_metric_values: list[float], test_metric_values: list[float]
) -> None:
    epochs_range = range(1, len(train_metric_values) + 1)
    plt.plot(epochs_range, train_metric_values, label=f"Train")
    plt.plot(epochs_range, test_metric_values, label=f"Test")
    plt.title(f"{metric_name} Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)


def plot_metrics(reports: list[MetricReport]) -> None:
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plot_metric(
        f"Precision@{reports[0].top_k}",
        [report.precision for report in reports if not report.is_on_test],
        [report.precision for report in reports if report.is_on_test],
    )

    plt.subplot(2, 2, 2)
    plot_metric(
        f"Recall@{reports[0].top_k}",
        [report.recall for report in reports if not report.is_on_test],
        [report.recall for report in reports if report.is_on_test],
    )

    plt.subplot(2, 2, 3)
    plot_metric(
        f"F1@{reports[0].top_k}",
        [report.f1 for report in reports if not report.is_on_test],
        [report.f1 for report in reports if report.is_on_test],
    )

    plt.subplot(2, 2, 4)
    plot_metric(
        "AUC",
        [report.auc for report in reports if not report.is_on_test],
        [report.auc for report in reports if report.is_on_test],
    )
    plt.tight_layout()
    plt.show()
