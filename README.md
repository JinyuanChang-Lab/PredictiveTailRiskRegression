# Predictive Tail-Risk Regression for Heavy-Tailed Time Series

This repository contains the Python code and data used to reproduce the simulation studies and the real-data analyses for the paper **Predictive Tail-Risk Regression for Heavy-Tailed Time Series**.

The code is organized as a set of Jupyter notebooks together with a modified version of the `quantes` package and a helper module `utl.py`. Each notebook writes its figures and numerical results to a dedicated `res_*` folder, and the empirical datasets are stored in the `realdata/` folder. Run the notebooks from the repository root so that the `from utl import ...` calls and the relative output paths resolve correctly.

---

## Repository Structure

### `quantes/`

A modified version of the `quantes` Python package (Convolution Smoothed Quantile and Expected Shortfall Regression, by Wenxin Zhou et al.). It implements the proposed **LES-H** (linear) and **DES-H** (nonlinear) estimators, as well as the competing **FZ**, **HTZ**, **DES**, and **LLES** estimators considered in the paper. Install it locally before running the notebooks (see [How to Run](#how-to-run)); this modified version should be used instead of the publicly available one.

### `utl.py`

Helper module containing the data-generating processes used in the simulations (e.g. `gen_data_EXPAR_v1`, `gen_data_FAR_v1`, `gen_data_SIM_v1`, `generate_dgp_data_with_test_sn`) and auxiliary estimation and tuning functions (`Qt_Linear`, `QtES_LP`, `CV_Qt_LP`, etc.). All notebooks import from this module.

### Notebooks

- `Linear.ipynb` — Monte Carlo experiments for the linear models (Section 3.3 of the main paper).
- `Nonlinear.ipynb` — Monte Carlo experiments for the nonlinear models (Section 4.2 of the main paper).
- `Linear-sensitivity.ipynb` — sensitivity analysis for the linear model (Section B.1 of the supplementary material).
- `Nonlinear-sensitivity.ipynb` — sensitivity analysis for the nonlinear model (Section B.1 of the supplementary material).
- `Runtime-comparison.ipynb` — computational efficiency comparison (Section B.2 of the supplementary material).
- `Realdata.ipynb` — real-data analyses (Section A of the supplementary material).

#### Outputs in the main paper

| Notebook | Output file(s) | Reproduces |
|---|---|---|
| `Linear.ipynb` | `res_linear/boxplot_linear_normal_20.pdf`, `res_linear/boxplot_linear_t_20.pdf`, `res_linear/boxplot_linear_normal_30.pdf`, `res_linear/boxplot_linear_t_30.pdf` | Figure 1 |
| `Linear.ipynb` | `res_linear/res_inference/inference_vs_rho_coverage_normal.pdf`, `res_linear/res_inference/inference_vs_rho_coverage_t.pdf`, `res_linear/res_inference/inference_vs_rho_ci_width_normal.pdf`, `res_linear/res_inference/inference_vs_rho_ci_width_t.pdf` | Figure 2 |
| `Nonlinear.ipynb` | `res_nonlinear/fix_n_normal_EXPAR_v1_all.csv`, `res_nonlinear/fix_n_normal_FAR_v1_all.csv`, `res_nonlinear/fix_n_normal_SIM_v1_all.csv`, `res_nonlinear/fix_n_t_EXPAR_v1_all.csv`, `res_nonlinear/fix_n_t_FAR_v1_all.csv`, `res_nonlinear/fix_n_t_SIM_v1_all.csv` (mean and standard deviation summarized in the notebook) | Table 1 |
| `Nonlinear.ipynb` | `res_nonlinear/boxplot_expar_n.pdf`, `res_nonlinear/boxplot_expar_t.pdf`, `res_nonlinear/boxplot_far_n.pdf`, `res_nonlinear/boxplot_far_t.pdf`, `res_nonlinear/boxplot_sim_n.pdf`, `res_nonlinear/boxplot_sim_t.pdf` | Figure 3 |
| `Nonlinear.ipynb` | `res_nonlinear/plots_of_sequence_expar_n.pdf`, `res_nonlinear/plots_of_sequence_expar_t.pdf`, `res_nonlinear/plots_of_sequence_far_n.pdf`, `res_nonlinear/plots_of_sequence_far_t.pdf`, `res_nonlinear/plots_of_sequence_sim_n.pdf`, `res_nonlinear/plots_of_sequence_sim_t.pdf` | Figure 4 |

#### Outputs in the supplementary material

| Notebook | Output file(s) | Reproduces |
|---|---|---|
| `Realdata.ipynb` | LES-H estimates and 95% confidence intervals for the five Fama-French industry portfolios (displayed in the notebook) | Table A.1 |
| `Realdata.ipynb` | `res_realdata/famafench_rolling.pdf` | Figure A.1 |
| `Realdata.ipynb` | `res_realdata/plot_estimate_es_nn_oop.pdf` | Figure A.2 |
| `Linear-sensitivity.ipynb` | `res_linear/sens_linear_normal.pdf`, `res_linear/sens_linear_t.pdf` | Figure B.1 |
| `Nonlinear-sensitivity.ipynb` | `res_nonlinear/sens_nn_normal.pdf`, `res_nonlinear/sens_nn_t.pdf` | Figure B.2 |
| `Runtime-comparison.ipynb` | `res_runtime/runtime_linear_normal.pdf`, `res_runtime/runtime_linear_t.pdf` | Figure B.3 |
| `Runtime-comparison.ipynb` | `res_runtime/runtime_nn_normal.pdf`, `res_runtime/runtime_nn_t.pdf` | Figure B.4 |

### `realdata/`

Empirical datasets used by `Realdata.ipynb`:

- `5_Industry_Portfolios_Daily.csv` — daily returns on the five Fama-French industry portfolios (Consumer, Manufacturing, High Technology, Health, Other).
- `F-F_Research_Data_5_Factors_2x3_daily.CSV` — daily Fama-French five factors (MKT-RF, SMB, HML, RMW, CMA).
- `DataVulnerability.xls` — U.S. macroeconomic data (real GDP growth and the National Financial Conditions Index), 1973Q1 – 2016Q3, used for the out-of-sample ES forecasts of real GDP growth.

Public sources:

- Fama-French five factors and five industry portfolios: Kenneth French's Data Library (`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/`).
- U.S. macroeconomic data (GDP growth, NFCI): ICPSR (`https://doi.org/10.3886/E113169V1`).

### `res_linear/`, `res_nonlinear/`, `res_realdata/`

Output folders holding the figures and numerical results generated by the notebooks. They are pre-populated with the results used to produce the paper.

---

## How to Run

The code requires Python 3.11 and the following libraries: `numpy`, `matplotlib`, `joblib`, `scipy`, `tqdm`, `pandas`, `cvxopt`, `qpsolvers`, `torch`, and `jupyter`, plus the modified `quantes` package included in this repository.

1. Create and activate a dedicated Python environment:
   ```bash
   conda create -n ES python=3.11
   conda activate ES
   ```
2. Install the required Python libraries:
   ```bash
   pip install numpy matplotlib joblib scipy tqdm pandas cvxopt qpsolvers torch jupyter
   ```
3. Install the modified `quantes` package locally. From the repository root:
   ```bash
   cd quantes
   pip install -e .
   cd ..
   ```
   Use this modified version of `quantes` instead of the publicly available one.
4. Launch Jupyter and run all cells of the relevant notebook in sequence:
   ```bash
   jupyter notebook
   ```

The notebooks depend on `utl.py`, so run them from the repository root (where `utl.py` is located).

The simulation notebooks use multi-core parallelization via `joblib` (`Parallel`, `n_jobs`). The default settings use a large number of cores (up to 192) and many Monte Carlo replications; reduce `n_jobs` and the number of replications to match your machine. The full reproduction takes more than 8 hours on the hardware used by the authors.

---

