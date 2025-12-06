## Overview

## Repository Structure
- `approx_quality_plot.py` — Approximation error plotting.
- `config.yaml` — Input YAML file.
- `evaluate.py` — Approximation error evaluation for the input queries.
- `generate_queries.py` — Generate 500 queries for the evaluation.
- `LM_brandes.py` — Landamrk selection and precomputing distances using new startegy Brandes
- `LM.py` — Landamrk selection and precomputing distances using existing strategy - random, degree, degree_h, closeness, closeness_h
- `proprocess.py` — Preprocess the input file to generate the nodes and edges mapping.
- `true_distance.py` — True distance calculation for the input queries .

## Requirements
- Python 3.8+
- Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Make the changes in the Configuration file 
1. config.yaml 
   input_tsv: "large_twitch_edges.csv"   <---- Input File name
   output_dir: "output"           <---- Output folder
   k: 500                         <---- Landmark set - 2, 5, 10, 20, 50, 100, 500
   h_min: 1                       <---- Hop Distance - 0,1
   lm_sel: "brandes"              <---- Landmark Strategy - random, degree, degree_h, closeness, closeness_h, brandes, brandes_h
   queries:                       <---- Query list
   - [2524, 67465]
   - [45596, 67465]
   - [67465, 153763]


## Running Experiments
1. Preprocess the input file (one time execution): 
```bash
python preprocess.py
```
Output files - edges.parquet and node_map.json

2. Generate 500 queries (one time execution): 
```bash
python generate_queries.py
```
Output files - generated-queries.yaml
!! MANUAL ACTION REQUIRED !! - Copy the 500 queries from this yaml and paste it in config.yaml file

3. Compute True distance for the queries (one time execution): 
```bash
python true_distance.py
```
Output files - true_distance.json

*** Iterative process Starts ****

4. Run Brandes for Landmark selection strategy using Brandes: 
```bash
python LM_brandes.py
```
Output files - distances.parquet, landmarks.json, *_timing_*.json 

*** OR ***

4. Landmark selection strategy using random or degree or degree_h or closeness or closeness_h: 
```bash
python LM.py
```
Output files - distances.parquet, landmarks.json, *_timing_*.json 

5. Aprroximation error evaluation: 
```bash
python evaluate.py
```
Output files - *_approx_quality_*.json 

*** Iterative Process ends ***

6. Aprroximation error plots: 
```bash
python approx_quality_plot.py
```
Output files - approx_quality_comparison.png 






