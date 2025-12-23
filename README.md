# GATE: Group Average Treatment Effects

Standard implementation of Group Average Treatment Effects (GATE) using X-learner with propensity score weighting.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn scipy tqdm
   ```

2. **Configure your analysis:**
   - Open `GATE_Standard_Implementation.py`
   - Set `DATA_FILE` to your dataset path
   - Set `TREATMENT_COLS` to your treatment column names
   - Set `OUTCOME_COLS` to your outcome column names

3. **Run:**
   ```bash
   python GATE_Standard_Implementation.py
   ```

## Output

Results are saved to `gate_results_all.csv` with:
- **ATE_Overall**: Overall average treatment effect
- **GATE**: Treatment effect for each risk quartile (Q1-Q4)
- **95% Confidence Intervals** for each quartile
- **Baseline risk** estimates per quartile

## Requirements

- Python 3.7+
- numpy, pandas, scikit-learn, scipy, tqdm

