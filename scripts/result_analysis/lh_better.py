import pyiqa
import torch

def main():
    """
    Prints whether each IQA metric is 'lower better' or 'higher better'.
    """
    # List all available models
    metric_list = pyiqa.list_models()

    # Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Iterate through all metrics
    for metric_name in metric_list:
        try:
            # Create the metric
            iqa_metric = pyiqa.create_metric(metric_name, device=device)

            # Print the result
            better = 'Lower Better' if iqa_metric.lower_better else 'Higher Better'
            print(f"{metric_name}: {better}")
        except Exception as e:
            print(f"Failed to create metric {metric_name}: {e}")

if __name__ == "__main__":
    main()
