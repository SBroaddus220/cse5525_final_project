# cse5525_final_project

## Description

Please see the [final project report](docs/Final_Project_Report_.pdf) for an overview.

## Getting Started

### Prerequisites

1. Python 3.12+
```bash
python --version  # Windows
python3 --version  # Linux
```

### Installation
1. Clone the repository and navigate to the project root

```bash
git clone https://github.com/SBroaddus220/cse5525_final_project
cd cse5525_final_project
```

2. Install requirements with Poetry or any package manager of your choice.
    1. **Installation**
    Please refer to the [official Poetry website](https://python-poetry.org/docs/#installation) for installation instructions.

    2. **Verify Installation**
    ```bash
    poetry --version
    ```

3. Install project dependencies
```bash
poetry install
```

4. Activate Poetry shell
```bash
poetry shell
```

5. Ensure PyTorch is installed with CUDA if using GPU
Check if PyTorch is able to recognize your GPU:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be >=1
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")  # Should display desired GPU
```
If any of the following are not what they should be, then properly install the CUDA-enabled version of PyTorch.
Following instructions at the [official PyTorch Website](https://pytorch.org/get-started/locally/)

6. Download Spacy's "en_core_web_sm" model
```bash
python -m spacy download en_core_web_sm
```

### Adding Files
The logic for finding files is defined in `./main.py`. 
UIDs are used to reference files in the Metric classes and so each file must have a UID that is capable of being generated with just the file path as input.
Currently, the application is tied to the [DiffusionDB](https://poloclub.github.io/diffusiondb/), so UIDs and file paths are based on its structure.
If additional databases are desired, new logic to relate the file paths & UIDs is necessary and generation must be changed in `main.py`.

#### `DiffusionDB` Data Download
The Poloclub downloader script is provided in the `./scripts/` directory. 
An example use case is provided below:
```bash
python ./scripts/download.py -i 23 --output "./data/images"
```
This will place the downloaded zip file in the correct location. Please extract it at the same location, and then the application should automatically recognize the data.
The data used includes indices 23-25. Download each and unzip them in the same location. Everything should be automatically detected.

### Configuration File
Modifying the configuration file is unneeded, but if you want some customization (e.g. modifying logging config), then feel free to edit the config file at `./cse5525_final_project/config.py`

### Running the Application
There are two main steps to the pipeline:
1. Compute metrics for any images detected in `images/` and store in persistent sqlite db.

The following should automatically compute all metrics for all detected images.
```bash
python ./compute_metrics.py
```

2. Train prediction models. Note that this will take a while depending on the number of samples.
This will populate `results/` with a timestamped directory containing the results.
Note that this only supports the IQA metrics due to time constraints.

```bash
python run_metric.py "iqa_arniqa_metrics"  # Run for each IQA metric.
```

The script is designed around multiple scripts running at once through a resource cluster. Please refer to `run_metric.slurm` and `submit_all_metrics.sh` for examples of how to run multiple scripts.

### Compiling Results
Result compilation scans through the `results/` directory and handles all found data appropriately. This was handled separately from the main program so view `./scripts/result_analysis/requirements.txt` for the requirements (tested on Linux).

There are four steps to compiling results:

1. Create an initial compilation of the results.
```bash
python ./scripts/result_analysis/compile_results.py ./results ./compiled_results
```

2. Scan the results for any missing data (Optional). This will create a `scan.json` file which lists any missing models / files for any metric.
```bash
python ./scripts/result_analysis/scan_results.py ./compiled_results ./scan.json
```

3. Create initial visualizations for the metrics.
```bash
python ./scripts/result_analysis/visualize_results.py ./compiled_results
```

4. Aggregate SHAP values (important!) and create intuitive visualizations for each.
```bash
python ./scripts/result_analysis/shap_aggregation.py ./compiled_results
```

This should create a bunch of images and aggregated `.csv` files detailing the results.

### Running Scripts
There are several scripts in the `./scripts/` directory. These can be run as any python script.
Please refer to the docstrings of any script for any specific instructions.
```bash
# Example format
python ./scripts/script.py
```

## Browsing the Database
To browse the SQLITE3 database, a recommended pre-built solution is [DB Browser for SQLITE](https://sqlitebrowser.org/).

## Creating New Metrics
In this application, metrics for the files are systematically generated by using modular logic in an abstract class. New metrics can easily be added by providing logic for the SQLITE table and the metric calculation. 

The system's metrics can be found at `./cse5525_final_project/metrics.py` for reference.

A template for automatic generation of a Metric class through an LLM is provided at `./llm_metric_generation_template.md`.
To use this template, please replace the existing comments with specific details regarding the metric, and then insert the generated class into the metrics file.

## Authors and Acknowledgement
Steven Broaddus (https://stevenbroaddus.com/#contact)
Tommy Tong (https://github.com/puhthetic)

## Contributing
Due to this being a course project, no contributions are allowed probably.
