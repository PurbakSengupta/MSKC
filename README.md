![MSKC](/Results/Figures/MSKC.png?raw=true)

# Minimum-Sufficient Knowledge and Computation (MSKC) for Wind Energy Predictive Analysis

Code accompanying the IEEE BigData Special Session paper on MSKC. This repository tests one narrow, empirically tractable slice of the broader MSKC principle, adaptive knowledge acquisition, on real wind turbine ramp-event forecasting.

## What this project is actually testing

Machine learning models for real-time decision support often pull in extra knowledge (a neighboring sensor's reading, a physics-based correction, a richer model) unconditionally, on every input, whether or not that particular input actually needed it. This project asks a narrower, testable question instead: **do states exist where the cheaper, knowledge-free route is already good enough, and can a system reliably tell which states those are, without a measurable loss in decision quality?**

Concretely, the project builds three prediction routes for a five-class ordinal wind ramp-severity forecasting task (using real SCADA data from two wind farms):

- **r0 (base route):** a turbine's own recent readings only, no extra knowledge, no communication cost.
- **r1 (relational route):** adds the simultaneous reading of the nearest upwind neighbor turbine. Requires cross-turbine communication.
- **r2 (physics-informed route):** adds the residual between a turbine's observed power and a power curve fit for that turbine. No communication cost, a local lookup only.

A learned gate (referred to as KUS in the code) then decides, per input, whether the extra relational knowledge (r1) is worth invoking, compared against always injecting it. The central result the pipeline is built to produce is whether that gate can match always-on injection's decision quality while invoking the expensive route substantially less often.

**Note on scope:** the carbon/energy estimation scripts in this repository (`cagt_carbon_estimate.py`, `Fig_code/carbon_fig.py`) explore converting measured latency into an estimated energy or carbon figure. That conversion depends on unverified hardware assumptions and is **not** part of the current paper's reported claims; it's kept here as groundwork for a separate, dedicated follow-up paper, not as evidence backing this paper's results.

---

## Before running anything: download the data

The `data/` folder is empty in this repository. Download both datasets from Zenodo and place them there before running any script:

- **Kelmarsh wind farm data:** Plumley, C., & Takeuchi, R. (2025). Kelmarsh wind farm data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16807551
- **Penmanshiel wind farm data:** Plumley, C., & Takeuchi, R. (2025). Penmanshiel wind farm data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16807304

Both are ten-minute-resolution SCADA records, publicly released, covering the 2024 calendar year (six Kelmarsh turbines; five of Penmanshiel's fourteen turbines had usable telemetry in this release).

---

## Directory structure

```
IEEEBigData_SpecialSessionDM/
│
├── data/                          # PUT DOWNLOADED ZENODO DATA HERE (currently empty)
│
├── EDA/                           # Stage 1: raw data inventory, extraction, and cleaning
│   ├── 00_data_inventory.py
│   ├── 01_extract_core_signals.py
│   ├── 02_diagnose_duplicates.py
│   ├── 03_batch_extract.py
│   ├── 04_eda.py
│   ├── 05_check_extremes.py
│   └── 06_check_curtailment.py
│
├── extracted/                     # Output: cleaned per-turbine data after the EDA stage
│
├── eda_output/                    # Output: result CSVs produced during the EDA stage
│                                   # (label distributions, neighbor-availability tables, etc.)
│
├── define_ramp_labels.py          # Stage 2: builds the five-class ordinal ramp-severity labels
├── build_neighbour_feature.py     # Stage 2: builds the r1 relational (neighbor-turbine) feature
├── physics_residuals.py           # Stage 2: builds the r2 physics-informed residual feature
│
├── train_r0/r1.py                 # Stage 3: trains the r0 and r1 route classifiers
├── kus_gate_frontier.py           # Stage 3: trains the KUS gate; leak-free threshold (tau)
│                                   #          selection via a held-out validation slice, then a
│                                   #          single evaluation on the untouched test set
├── model_comparison.py            # Stage 3: model-weight probe (300 / 50 / 15 trees)
├── verify_lighter_models.py       # Stage 3: verification of the lightweight-model metrics
│
├── gating_baseline.py             # Stage 4: random-gating and hand-designed heuristic baselines
├── bootstrap_gate_knee.py         # Stage 4: bootstrap CIs, gate vs. always-on injection
├── bootstrap_learned_vs_heuristics.py  # Stage 4: bootstrap CIs, learned gate vs. heuristic
├── significance_subgroup.py       # Stage 4: subgroup / significance breakdowns
│
├── oob_reliability.py             # Stage 5 (robustness): out-of-bag reliability checks
├── cross_site_check.py            # Stage 5 (robustness): Kelmarsh-trained, Penmanshiel-tested
├── mlp_sanity_check.py            # Stage 5 (robustness): cross-architecture check (neural net)
│
├── cagt_carbon_estimate.py        # Exploratory: energy/carbon estimate (NOT used in this paper's
│                                   #             reported claims -- see "Scope" note above)
│
├── detailed_results.py            # Aggregates and prints the final, consolidated result set
│
├── Fig_code/                      # All figure-generation scripts
│   ├── figures.py
│   ├── figures_detailed.py
│   ├── figs_mlp.py
│   ├── label_check.py             # Generates the label-quality validation figure
│   └── carbon_fig.py              # Exploratory carbon figure (see "Scope" note above)
│
└── Results/                       # Main result CSVs (everything except the EDA-stage outputs)
    └── Figures/                   # Final figures used in the paper
```

---

## Suggested run order

1. Download the data into `data/` (see above).
2. Run the `EDA/` scripts in numeric order, `00` through `06`, to inventory, extract, and clean the raw SCADA data. Outputs land in `extracted/` and `eda_output/`.
3. Run `define_ramp_labels.py`, `build_neighbour_feature.py`, and `physics_residuals.py` to build the labels and the r1/r2 features.
4. Run `train_r0/r1.py`, then `kus_gate_frontier.py`, then `model_comparison.py` / `verify_lighter_models.py`.
5. Run `gating_baseline.py`, `bootstrap_gate_knee.py`, `bootstrap_learned_vs_heuristics.py`, and `significance_subgroup.py` for the comparison and significance results.
6. Run `oob_reliability.py`, `cross_site_check.py`, and `mlp_sanity_check.py` for the robustness checks.
7. Run `detailed_results.py` to consolidate everything, then the scripts in `Fig_code/` to generate the final figures into `Results/Figures/`.

---

## A note on the file descriptions above

The per-script descriptions in the tree above are reconstructed from filenames and the project's development history, not verified line-by-line against each script's current contents. If a description doesn't match what a script actually does, trust the script over this README and update the relevant line.
