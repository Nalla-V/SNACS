# Landmark-based Shortest-Path Distance Estimation (SNACS 2025)

This repository contains the code used for the SNACS course project on **landmark-based shortest-path distance estimation** in large, unweighted graphs, including baseline landmark selection strategies and a **Brandes-inspired (structure-aware)** strategy.

The pipeline is designed to be run per dataset, producing:
- a preprocessed graph representation (`edges.parquet`, `node_map.json`)
- a landmark distance index (`distances.parquet`, `landmarks.json`)
- query sets + exact distances for evaluation
- plots and tables used in the report

## Project layout

Main scripts:

- `preprocess.py`  
  Converts an input edge list into a dense-ID graph (`edges.parquet`, `node_map.json`).

- `generate_queries.py`  
  Generates 500 stratified query pairs (short / medium / long hop distances).

- `true_distance.py`  
  Computes exact shortest-path distances for the configured queries and saves `true_distances.json`.

- `LM.py`  
  Baseline landmark selection + distance table construction:
  random, degree, sampled closeness, and hop-exclusion variants.

- `LM_brandes.py`  
  Structure-aware landmark selection using a Brandes-inspired participation signal.

- `evaluate.py`  
  Computes estimated distances using the landmark index and reports approximation error per query.

Analysis/plotting utilities:

- `stats.py`  
  Basic dataset statistics (|V|, |E|, average degree, sampled clustering coefficient).

- `distance_distribution.py`  
  Samples BFS runs and plots a rough hop-distance distribution.

- `approx_quality_plot.py`  
  Plots mean relative error vs. landmark budget.

- `timing_plot.py`  
  Plots offline preprocessing time vs. landmark budget from timing JSON files.

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All scripts read settings from `config.yaml`. Typical fields include:

- `input_tsv`: path to the raw edge list (for `preprocess.py`)
- `output_dir`: folder where outputs for the dataset are written
- `k`: number of landmarks
- `lm_sel`: strategy name (e.g., `random`, `degree`, `degree_h`,  `closeness`, `closeness_h`, `brandes`, `brandes_h`)
- `h_min`: hop-exclusion threshold (0 = none, 1 = exclude neighbors)
- `closeness_samples`: number of sampled BFS sources for closeness (default: 200)
- `queries`: list of query pairs (copy from generated_queries.yaml)


### Before you run the pipeline

Update `config.yaml` for the dataset/experiment you want to run:

- Set `input_tsv` to the dataset edge list file.
- Set `output_dir` to a dataset-specific folder (recommended, e.g., `Facebook/`, `Douban/`), to avoid overwriting results.
- Set `k`, `lm_sel`, and `h_min` to the landmark budget/strategy you want to evaluate.
- After running `generate_queries.py`, copy the generated queries into `config.yaml` under `queries:`.


## Recommended run order (per dataset)

1) Preprocess the edge list:

```bash
python preprocess.py
```

2) (Optional) Compute basic statistics and distance distribution plots:

```bash
python stats.py
python distance_distribution.py
```

3) Generate 500 unique queries:

```bash
python generate_queries.py
```

Copy the generated queries from `<output_dir>/generated_queries.yaml` into `config.yaml` under `queries:`.

4) Compute exact true distances for the 500 queries:

```bash
python true_distance.py
```

5) Build the landmark distance index (choose one):

Baselines:

```bash
python LM.py
```

Brandes-inspired strategy:

```bash
python LM_brandes.py
```

6) Evaluate approximation quality:

```bash
python evaluate.py
```

7) Plot results:

```bash
python approx_quality_plot.py
python timing_plot.py
```

## Outputs (per dataset)

Written under `output_dir`:
- `edges.parquet`, `node_map.json`
- `landmarks.json`, `distances.parquet`
- `generated_queries.yaml`
- `true_distances.json`
- `{k}_timing_{lm_sel}.json`
- `{k}_approx_quality_{lm_sel}.json`
- figures (`*_distance_distribution.png`, `*_approx_quality_comparison.png`, etc.)


