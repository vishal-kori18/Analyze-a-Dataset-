# DataLens — AI & Machine Learning Dataset Analyzer

A beginner-friendly Streamlit dashboard for exploring CSV datasets. It includes a bundled Wine recognition sample, data-quality checks and cleaning, interactive Plotly charts, automatic descriptive insights, an optional scikit-learn model, and downloadable CSV/Markdown exports.

## Features

- Upload a CSV or start with the included sample dataset.
- Preview records, inspect column types, missingness, unique values, and descriptive statistics.
- Find duplicate rows and numeric outliers using the IQR rule; choose to fill/drop missing data, remove duplicates, cap outliers, or remove outlier rows.
- Create interactive histograms, bar charts, scatter plots, box plots, and correlation heatmaps.
- Read data-grounded summaries of missingness, variability, correlations, and (where recognizable date/time columns exist) changes over time.
- Train a basic Random Forest classifier or regressor with a holdout evaluation and feature-importance view.
- Download the cleaned working dataset and a Markdown analysis report.
- Responsive layout, sidebar navigation, and helpful handling of common input/model errors.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- Internet access for the initial package installation

## Install and run locally

Open a terminal in this project folder, then run the commands for your operating system.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser. To stop the app, press `Ctrl+C` in the terminal.

## How to use

1. Keep **Built-in sample** selected to explore `data/wine_classification.csv`, or choose **Upload a CSV** in the sidebar.
2. Review the dataset in **Overview**.
3. On **Data quality**, select cleaning options and click **Apply cleaning to working dataset**. The original in-memory upload is preserved for the session; the cleaned working copy is used by the other pages.
4. Choose charts in **Visualize** and review **Insights**.
5. For a basic supervised-learning demo, open **Machine learning**, select a target and prediction type, then click **Train Random Forest**. The app uses a reproducible 75/25 split; scores are educational baselines, not guarantees of future performance.
6. Download the cleaned CSV or Markdown report from **Export**.

### Sample data

The bundled sample is the Wine recognition dataset distributed by scikit-learn and sourced from the UCI Machine Learning Repository. It has 178 records, 13 chemical measurements, and a `wine_class` target. It is intended for demonstration and is not generated synthetic data.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks IQR outlier detection, missing-value/duplicate/outlier cleaning, insight/report generation, and basic ML pipeline construction.

## Project structure

```text
.
├── app.py
├── data/
│   └── wine_classification.csv
├── tests/
│   └── test_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Upload this project to GitHub

1. Create a free account or sign in at [github.com](https://github.com).
2. Select **New repository**, enter a name such as `datalens-dataset-analyzer`, choose Public or Private, and create it **without** adding a README, license, or `.gitignore` (this project already contains those files).
3. In a terminal, change to this project directory and run:

   ```bash
   git init
   git add app.py README.md requirements.txt .gitignore data tests
   git commit -m "Build DataLens dataset analyzer"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/datalens-dataset-analyzer.git
   git push -u origin main
   ```

4. Replace `YOUR-USERNAME` and the repository name with the values you used. If Git asks you to sign in, use GitHub's browser/device authentication or a configured Git credential manager (do not put a password or token in the remote URL).
5. The repository URL to share is `https://github.com/YOUR-USERNAME/datalens-dataset-analyzer`.

If you initialized a README or other files on GitHub already, clone the repository first and copy this project into that clone, or reconcile the histories before pushing.

## Notes and limitations

- CSV files are processed in the running Streamlit session; the app does not write uploaded data to disk or send it to an external service.
- This repository is a local Streamlit project. It does not create or publish a hosted website or GitHub repository automatically.
- Outliers are statistical flags, not necessarily errors. Review the domain context before removing or capping them.
- Automatic insights and correlations are descriptive and must not be interpreted as proof of causation.
- Model quality depends on the selected data, target, leakage risks, and suitability of the holdout split. Use domain-appropriate validation for real decisions.
