# Do Bias Metrics Agree?

Benchmarking AIF360 and Fairlearn across fairness metrics and mitigation strategies using two classifiers and two datasets.

Victor Gabriel Dinu, Eddie Begovic · Summer Semester 2026

## What This Project Does

Different fairness toolkits measure bias differently and those differences matter. This project compares IBM's AIF360 and Microsoft's Fairlearn by running them against the same trained models and asking whether they agree on whether bias exists and whether a fix that satisfies one toolkit also satisfies the other.

Two classifiers are evaluated. Logistic Regression serves as an interpretable baseline and Random Forest as a non-linear alternative. This lets us check whether any disagreements between the toolkits show up regardless of the model used.

We look at three fairness concepts (demographic parity, equalized odds, and proportional parity) and apply two mitigation strategies. The first is AIF360's Reweighing, which is a pre-processing approach and the second is Fairlearn's ExponentiatedGradient, which works in-processing. Both are then re-scored with both toolkits to see whether a fix that satisfies one library holds up under the other.

## Datasets

**German Credit** (Hofmann, 1994)
1,000 records. Predicting good vs. bad credit risk. Protected attribute: sex.
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)

**COMPAS** (ProPublica, 2016)
6,167 records of criminal defendants in Broward County, Florida. Predicting two-year recidivism risk. Protected attribute: race (6 categories).
Source: [ProPublica Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) · [Raw data](https://github.com/propublica/compas-analysis)

## Architecture

The project runs as a single linear pipeline inside one notebook. Both datasets are loaded and wrapped into AIF360's `StandardDataset` format, which standardises how protected attributes and labels are represented across the rest of the pipeline. From there, two classifiers (Logistic Regression and Random Forest) are trained on each dataset, giving four baseline models in total.

Each baseline model is then scored by both AIF360 and Fairlearn on the same three fairness metrics, so any disagreement between the toolkits is visible before any intervention. Two mitigation strategies are applied next. AIF360 Reweighing adjusts the training sample weights before fitting, while Fairlearn ExponentiatedGradient retrains the classifier under an explicit fairness constraint. Both mitigated models are re-scored by both toolkits and the cross-toolkit scoring matrix in Section 7 puts all three versions (baseline, Reweighed, and EG) side by side. This makes it possible to see whether a fix that works under one toolkit also holds under the other.

## Notebook Structure

Everything runs in a single notebook: `DS_2_NOTEBOOK_v5.ipynb`

| Section | Contents |
|---|---|
| 0. Imports | All dependencies |
| 1. Data Loading | Load and wrap both datasets in AIF360's StandardDataset |
| 2. EDA | Outcome rates by group before any modelling |
| 3. Model Training | Train LR and RF on both datasets with accuracy and AUC summary |
| 4. Fairness Metric Comparison | Score all 4 models with both toolkits, verdict agreement and magnitude gaps |
| 5. Bias Mitigation: AIF360 Reweighing | Pre-processing mitigation, cross-toolkit re-scoring, verdict agreement and utility cost |
| 6. Bias Mitigation: Fairlearn ExponentiatedGradient | In-processing mitigation under demographic parity and equalized odds constraints, cross-toolkit re-scoring |
| 7. Cross-Toolkit Scoring Matrix | Baseline, Reweighed and EG side by side under both toolkits with verdict agreement and accuracy cost across all versions |
| 8. COMPAS Multi-Group Decomposition | Per-race selection rates for all six COMPAS groups across all three model versions (extra addition) |
| 9. Conclusion | Key findings on verdict agreement, magnitude gaps, and mitigation transfer |

## Getting Started

**1. Download the notebook**

Go to the [repository](https://github.com/16x29/Data-Science-2---Do-Bias-Metrics-Agree-), click on `DS_2_NOTEBOOK_v5.ipynb` and download it, then open it in Jupyter or VSCode.

**2. Install dependencies**

Run this in a notebook cell before anything else:
```python
!pip install aif360 fairlearn scikit-learn pandas numpy matplotlib
```

The ExponentiatedGradient mitigation in Section 6 uses fairlearn.reductions, which is included in the standard fairlearn install.

**3. Run the notebook**

Run all cells top to bottom. Both datasets load automatically at runtime so no manual download is needed. Each section has a markdown cell explaining what just happened and why.

## Requirements

- Python 3.9
- aif360, fairlearn, scikit-learn
- pandas, numpy, matplotlib
