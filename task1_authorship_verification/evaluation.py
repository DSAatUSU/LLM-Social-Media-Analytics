import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_results(json_file_path):
    """
    Loads classification results from a JSON file and evaluates metrics.

    Args:
        json_file_path (str): Path to the JSON file containing results.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1 score, and confusion matrix.
    """
    with open(json_file_path, "r") as file:
        results = json.load(file)

    all_ground_truth = []
    all_predictions = []

    invalid_predictions = 0  # Counter for invalid responses

    for user_result in results:
        for entry in user_result["results"]:
            ground_truth = int(entry["ground_truth_label"])
            model_prediction = entry["model_prediction"]

            # Check if prediction is valid (not None and numeric)
            if model_prediction is not None and model_prediction in ["0", "1"]:
                all_ground_truth.append(ground_truth)
                all_predictions.append(int(model_prediction))  # Convert to integer
            else:
                invalid_predictions += 1
                print(f"Warning: Invalid prediction '{model_prediction}' for tweet: {entry['tweet']}")

    # Handle case where all predictions are invalid
    if not all_predictions:
        print("Error: No valid predictions found. Returning empty metrics.")
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "classification_report": "No valid predictions found.",
            "confusion_matrix": None,
            "invalid_predictions": invalid_predictions
        }

    # Compute classification metrics
    metrics = {
        "accuracy": accuracy_score(all_ground_truth, all_predictions) * 100,
        "precision": precision_score(all_ground_truth, all_predictions, zero_division=0) * 100,
        "recall": recall_score(all_ground_truth, all_predictions, zero_division=0) * 100,
        "f1_score": f1_score(all_ground_truth, all_predictions, zero_division=0) * 100,
        "classification_report": classification_report(all_ground_truth, all_predictions, zero_division=0),
    }

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_ground_truth, all_predictions, labels=[0, 1])
    metrics["confusion_matrix"] = conf_matrix.tolist()  # Convert to JSON serializable format

    # Add counts for each class
    metrics["class_counts"] = {
        "class_0": {"ground_truth": conf_matrix[0].sum(), "predicted": conf_matrix[:, 0].sum()},
        "class_1": {"ground_truth": conf_matrix[1].sum(), "predicted": conf_matrix[:, 1].sum()},
    }

    # Include invalid prediction count for debugging
    metrics["invalid_predictions"] = invalid_predictions

    return metrics
