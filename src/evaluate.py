import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_classification_model(model_name, y_true, y_pred):
    """
    Evaluate a multi-class classification model.

    Returns accuracy, macro precision, macro recall, macro F1,
    and weighted F1.
    """

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def evaluate_rating_model(model_name, y_true, y_pred):
    """
    Evaluate a 1-to-5 star rating prediction model.

    Mean Absolute Error is included because star ratings are ordered.
    """

    results = evaluate_classification_model(model_name, y_true, y_pred)
    results["mean_absolute_error"] = mean_absolute_error(y_true, y_pred)

    return results


def save_classification_report(y_true, y_pred, output_path):
    """
    Save sklearn classification report as a CSV file.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path)

    return report_df


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """
    Create and save a confusion matrix plot.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return matrix