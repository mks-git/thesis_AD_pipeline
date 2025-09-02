# thesis_AD_pipeline

A modular pipeline for explainable anomaly detection (AD) and explanation discovery (ED) on high-dimensional time series data, based on the Exathlon benchmark.

## Overview

This project provides a complete framework for:
- Preprocessing and feature extraction from raw time series data
- Training and evaluating anomaly detection models (e.g., Autoencoder, TS2Vec)
- Scoring and thresholding for outlier detection
- Explanation discovery using methods such as EXstream, MacroBase, and LIME
- Reporting and visualization of results

The pipeline is designed for reproducibility and extensibility, supporting experiments on the Exathlon dataset and beyond.

## Project Structure

```
thesis_AD_pipeline/
├── exathlon/                # Extended exathlon pipeline
│   ├── apps/                # Spark application source code
│   ├── data/                # Raw and processed data
│   ├── img/                 # Images and figures
│   ├── notebooks/           # Jupyter notebooks for experiments and reproducibility
│   ├── outputs/             # Pipeline outputs (models, results, reports)
│   ├── src/                 # Main pipeline source code
│   ├── extract_data.sh      # Script to extract raw data
│   ├── requirements.txt     # original Exathlon Python dependencies
│   └── README.md            # Exathlon-specific documentation
├── ts2vec/                  # TS2Vec embedding model implementation
│   ├── datasets/
│   ├── models/
│   ├── scripts/
│   ├── tasks/
│   ├── ts2vec.py            # Holds interfaces for the TS2Vec model
│   ├── requirements.txt     # original TS2Vec Python dependencies
│   ├── README.md            # TS2Vec-specific documentation
├── requirements.txt     # Top-level requirements
└── .gitignore
```

## Setup

1. **Install dependencies**  
   Use the provided `requirements.txt` files (top-level and in submodules) to install all necessary Python packages.  
   Example using conda:
   ```bash
   conda create -n unified-py38 python=3.8
   conda activate unified-py38
   pip install -r requirements.txt
   ```

2. **Extract Data**  
   Download and extract the Exathlon dataset as described in `exathlon/README.md`:
   ```bash
   cd exathlon
   ./extract_data.sh
   ```

3. **Configure Environment**  
   Create a `.env` file in the exathlon project root with the following entries:
   ```
   USED_DATA=SPARK
   DATA_ROOT=path/to/extracted/data/raw
   OUTPUTS_ROOT=path/to/pipeline/outputs
   ```

## Usage

### Running the Pipeline

From the `exathlon/src` directory, you can run the main pipeline script with various options.  
Example (anomaly detection and explanation discovery with TS2Vec and EXstream):

```bash
python run_pipeline.py --pipeline-type ad.ed --app-id 0 --model-type ts2vec --scoring-method knn --explanation-method exstream
```

See the notebooks in `exathlon/notebooks/` for detailed experiment instructions and reproducibility.

### Reporting

Generate performance tables and figures using the reporting scripts or notebooks, e.g.:

```bash
python ../src/reporting/report_results.py --evaluation-step scoring --scoring-set-name test --scoring-metrics auprc --scoring-granularity global
```

## Main Components

- **Data Processing:** Feature extraction, transformation, and period segmentation.
- **Modeling:** Training and inference for AD models (AE, TS2Vec, BiGAN, etc.).
- **Scoring:** Assigning outlier scores to time series using reconstruction error or embedding-based kNN.
- **Explanation Discovery:** Methods for explaining detected anomalies (EXstream, MacroBase, LIME).
- **Evaluation & Reporting:** Metrics computation, aggregation, and visualization.

## Notebooks

- `exathlon/notebooks/thesis_reproducibility.ipynb`: End-to-end pipeline runs and result analysis.

## References

- [Exathlon Benchmark Paper](http://vldb.org/pvldb/vol14/p2613-tatbul.pdf)
- [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466)

## License

- Exathlon: [Apache 2.0](exathlon/LICENSE)
- TS2Vec: [MIT](ts2vec/LICENSE)
- Dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

For further details, see the documentation in each submodule and the provided Jupyter notebooks.
