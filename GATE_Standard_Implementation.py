"""
Standard GATE (Group Average Treatment Effects) Implementation
Using X-learner with Propensity Score Weighting

This implementation uses the X-learner algorithm with equal weighting:
    τ(x) = 0.5 * τ₁(x) + 0.5 * τ₀(x)

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION FOR YOUR PROBLEM
# ============================================================================

# 1. DATA FILE PATH
DATA_FILE = "your_data.csv"  # Path to your dataset

# 2. TREATMENT COLUMNS (list of treatment column names in your data)
# Example for opioid overdose with 12 drugs:
TREATMENT_COLS = [
    'drug_1',      # Replace with your actual column names
    'drug_2',
    'drug_3',
    'drug_4',
    'drug_5',
    'drug_6',
    'drug_7',
    'drug_8',
    'drug_9',
    'drug_10',
    'drug_11',
    'drug_12'
]

# 3. OUTCOME COLUMNS (list of outcome column names in your data)
OUTCOME_COLS = [
    'overdose'     # Replace with your actual outcome column name(s)
]

# 4. OPTIONAL: ID COLUMNS TO EXCLUDE (add any ID columns in your data)
EXCLUDE_ID_COLS = ['person_id', 'Subject_ID', 'id', 'patient_id']

# 5. ANALYSIS PARAMETERS
OVERLAP_RANGE = [0.05, 0.95]  # Keep patients with PS in [0.05, 0.95]
BOOTSTRAP_REPS = 1000           # Number of bootstrap replications for CIs
RANDOM_STATE = 42
N_QUANTILES = 4                # Quartiles for risk stratification

# ============================================================================
# CORE GATE FUNCTIONS
# ============================================================================

def fit_or_constant(model, X, y):
    """Fit model or return constant predictor if insufficient variation."""
    try:
        if len(np.unique(y)) < 2:
            class ConstPredictor:
                def __init__(self, c): 
                    self.c = c
                def predict(self, X): 
                    return np.full(len(X), self.c)
            return ConstPredictor(y.mean())
        model.fit(X, y)
        return model
    except:
        class ConstPredictor:
            def __init__(self, c): 
                self.c = c
            def predict(self, X): 
                return np.full(len(X), self.c)
        return ConstPredictor(y.mean())


def x_learner_cate_standard(X, t, y, ps, base_learner=None):
    """
    X-learner for CATE estimation with equal weighting.
    
    Formula: τ(x) = 0.5 * τ₁(x) + 0.5 * τ₀(x)
    
    Steps:
    1. Fit m₁(x) on treated, m₀(x) on control
    2. Predict μ₁(x) and μ₀(x) for all patients
    3. Compute pseudo-outcomes: τ₁ = Y - μ₀(X) for treated, τ₀ = μ₁(X) - Y for control
    4. Fit h₁(x) and h₀(x) on pseudo-outcomes
    5. Combine: τ(x) = 0.5 * h₁(x) + 0.5 * h₀(x)
    """
    if base_learner is None:
        base_learner = HistGradientBoostingRegressor(max_iter=100)
    
    # Step 1: Fit outcome models
    m1 = fit_or_constant(
        base_learner.__class__(**base_learner.get_params()), 
        X[t==1], y[t==1]
    )
    m0 = fit_or_constant(
        base_learner.__class__(**base_learner.get_params()), 
        X[t==0], y[t==0]
    )
    
    # Step 2: Predict potential outcomes
    mu1_pred = np.clip(m1.predict(X), 0, 1)
    mu0_pred = np.clip(m0.predict(X), 0, 1)
    
    # Step 3: Construct pseudo-outcomes
    tau1 = y[t==1] - mu0_pred[t==1]  # Treated: Y - m₀(X)
    tau0 = mu1_pred[t==0] - y[t==0]  # Control: m₁(X) - Y
    
    # Step 4: Fit second-stage models
    h1 = fit_or_constant(
        base_learner.__class__(**base_learner.get_params()), 
        X[t==1], tau1
    )
    h0 = fit_or_constant(
        base_learner.__class__(**base_learner.get_params()), 
        X[t==0], tau0
    )
    
    # Step 5: Combine using equal weighting (0.5/0.5)
    cate = 0.5 * h1.predict(X) + 0.5 * h0.predict(X)
    
    return cate, mu0_pred


def compute_gate(cate, mu0, n_quantiles=4):
    """
    Compute Group Average Treatment Effects (GATEs) by baseline risk quartiles.
    """
    try:
        bins = pd.qcut(
            mu0, 
            q=n_quantiles, 
            labels=["Q1_Low", "Q2_Med-Low", "Q3_Med-High", "Q4_High"], 
            duplicates='drop'
        )
        
        gate_rows = []
        for bin_label in bins.unique():
            mask = (bins == bin_label)
            if mask.sum() == 0:
                continue
            
            gate_val = cate[mask].mean()
            gate_rows.append({
                'Quartile': bin_label,
                'GATE': gate_val,
                'n': mask.sum(),
                'mean_baseline_risk': mu0[mask].mean()
            })
        
        return pd.DataFrame(gate_rows)
    except Exception as e:
        print(f"      WARNING: Could not bin by risk: {e}")
        return pd.DataFrame()


def bootstrap_gate_ci(cate, mu0, n_quantiles=4, n_reps=200, random_state=42):
    """Bootstrap confidence intervals for GATEs."""
    rng = np.random.default_rng(random_state)
    n = len(cate)
    
    gate_boot = []
    quartile_map = {"Q1_Low": 0, "Q2_Med-Low": 1, "Q3_Med-High": 2, "Q4_High": 3}
    
    for _ in tqdm(range(n_reps), desc="      Bootstrap CIs", leave=False):
        idx = rng.integers(0, n, n)
        cate_boot = cate[idx]
        mu0_boot = mu0[idx]
        
        try:
            quantiles = pd.qcut(
                mu0_boot, 
                q=n_quantiles, 
                labels=["Q1_Low", "Q2_Med-Low", "Q3_Med-High", "Q4_High"],
                duplicates='drop'
            )
            for bin_label in quantiles.unique():
                mask = (quantiles == bin_label)
                if mask.sum() > 0:
                    q_num = quartile_map.get(bin_label, -1)
                    if q_num >= 0:
                        gate_boot.append({
                            'quartile': q_num, 
                            'gate': cate_boot[mask].mean()
                        })
        except:
            continue
    
    gate_boot_df = pd.DataFrame(gate_boot)
    
    # Compute CIs per quartile
    ci_dict = {}
    for q in range(n_quantiles):
        q_gates = gate_boot_df[gate_boot_df['quartile'] == q]['gate'].values
        if len(q_gates) > 0:
            ci_dict[q] = {
                'low': np.percentile(q_gates, 2.5),
                'high': np.percentile(q_gates, 97.5)
            }
    
    return ci_dict


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def run_gate_analysis(df, treatment_col, outcome_col):
    """
    Run GATE analysis on your dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Your dataset
    treatment_col : str
        Name of binary treatment column (0/1)
    outcome_col : str
        Name of binary outcome column (0/1)
        
    Returns:
    --------
    gate_df : DataFrame
        GATE results with quartiles, CIs, etc.
    """
    print("=" * 80)
    print("GATE Analysis: Group Average Treatment Effects")
    print("=" * 80)
    
    # Prepare data
    print(f"\n[1] Preparing data...")
    exclude_cols = {treatment_col, outcome_col}
    exclude_cols.update(EXCLUDE_ID_COLS)
    feature_cols = [c for c in df.columns 
                    if c not in exclude_cols 
                    and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].fillna(0.0).astype("float32").values
    t = df[treatment_col].astype(int).values
    y = df[outcome_col].astype(int).values
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Treatment: {treatment_col} (treated: {t.sum()}, control: {(t==0).sum()})")
    print(f"   Outcome: {outcome_col} (positive: {y.sum()}, negative: {(y==0).sum()})")
    
    # Propensity score estimation
    print(f"\n[2] Estimating propensity scores...")
    ps_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ps = ps_model.fit(X, t).predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    
    ps_auc = roc_auc_score(t, ps)
    print(f"   PS AUC: {ps_auc:.3f}")
    
    # Overlap trimming
    print(f"\n[3] Applying overlap trimming [{OVERLAP_RANGE[0]}, {OVERLAP_RANGE[1]}]...")
    keep = (ps >= OVERLAP_RANGE[0]) & (ps <= OVERLAP_RANGE[1])
    
    if keep.sum() == 0 or t[keep].sum() == 0 or t[keep].sum() == keep.sum():
        print("   WARNING: Insufficient overlap")
        return None
    
    X_overlap = X[keep]
    t_overlap = t[keep]
    y_overlap = y[keep]
    ps_overlap = ps[keep]
    
    print(f"   Kept: {keep.sum()}/{len(t)} ({keep.mean():.1%})")
    print(f"   Treated: {t_overlap.sum()}, Control: {(t_overlap==0).sum()}")
    
    # X-learner CATE
    print(f"\n[4] Running X-learner CATE estimation...")
    cate, mu0 = x_learner_cate_standard(X_overlap, t_overlap, y_overlap, ps_overlap)
    
    # Compute overall ATE
    ate_overall = cate.mean()
    print(f"   Overall ATE: {ate_overall:.4f}")
    
    # Compute GATEs
    print(f"\n[5] Computing GATEs by baseline risk quartiles...")
    gate_df = compute_gate(cate, mu0, n_quantiles=N_QUANTILES)
    
    if gate_df.empty:
        print("   WARNING: Could not compute GATEs")
        return None
    
    # Bootstrap CIs
    print(f"\n[6] Computing bootstrap confidence intervals...")
    ci_dict = bootstrap_gate_ci(cate, mu0, n_quantiles=N_QUANTILES, 
                                  n_reps=BOOTSTRAP_REPS, random_state=RANDOM_STATE)
    
    # Merge CIs
    quartile_map = {"Q1_Low": 0, "Q2_Med-Low": 1, "Q3_Med-High": 2, "Q4_High": 3}
    for idx, row in gate_df.iterrows():
        q_label = row['Quartile']
        q_num = quartile_map.get(q_label, -1)
        if q_num >= 0 and q_num in ci_dict:
            gate_df.loc[idx, 'CI_low'] = ci_dict[q_num]['low']
            gate_df.loc[idx, 'CI_high'] = ci_dict[q_num]['high']
    
    gate_df['Treatment'] = treatment_col
    gate_df['Outcome'] = outcome_col
    gate_df['N'] = len(X_overlap)
    gate_df['ATE_Overall'] = ate_overall  # Overall ATE for reference
    
    # Reorder
    col_order = ['Treatment', 'Outcome', 'N', 'ATE_Overall', 'Quartile', 'n', 
                 'mean_baseline_risk', 'GATE', 'CI_low', 'CI_high']
    gate_df = gate_df[[c for c in col_order if c in gate_df.columns]]
    
    print("\n" + "=" * 80)
    print("GATE Results")
    print("=" * 80)
    print(gate_df.to_string(index=False))
    
    return gate_df


def run_gate_multiple_treatments(df, treatment_list, outcome_list):
    """
    Run GATE analysis for multiple treatment-outcome pairs.
    
    Example for opioid overdose:
        treatment_list = ['drug_1', 'drug_2', ..., 'drug_12']
        outcome_list = ['overdose']
    
    Parameters:
    -----------
    df : DataFrame
        Your dataset
    treatment_list : list
        List of treatment column names
    outcome_list : list
        List of outcome column names
        
    Returns:
    --------
    results_df : DataFrame
        Combined GATE results for all (T, Y) pairs
    """
    all_results = []
    total_combinations = len(treatment_list) * len(outcome_list)
    
    pbar = tqdm(total=total_combinations, desc="Processing (T,Y) pairs")
    
    for T in treatment_list:
        for Y in outcome_list:
            print(f"\n   Processing: {T} -> {Y}")
            gate_df = run_gate_analysis(df, T, Y)
            
            if gate_df is not None:
                all_results.append(gate_df)
            
            pbar.update(1)
    
    pbar.close()
    
    # Combine results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        
        print("\n" + "=" * 80)
        print("Combined GATE Results")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        output_file = "gate_results_all.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Results saved to: {output_file}")
        
        return results_df
    else:
        print("\nWARNING: No results generated.")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    To use this script:
    1. Modify the CONFIGURATION section at the top of this file
    2. Set DATA_FILE to your data path
    3. Set TREATMENT_COLS to your treatment column names
    4. Set OUTCOME_COLS to your outcome column name(s)
    5. Run: python GATE_Standard_Implementation.py
    """
    
    import os
    
    print("="*80)
    print("GATE Standard Implementation")
    print("="*80)
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\n[ERROR] Data file not found: {DATA_FILE}")
        print("\nPlease update the CONFIGURATION section at the top of this file:")
        print("  1. Set DATA_FILE to your data path")
        print("  2. Set TREATMENT_COLS to your treatment column names")
        print("  3. Set OUTCOME_COLS to your outcome column name(s)")
        print("\nExample configuration for opioid overdose analysis:")
        print("  DATA_FILE = 'opioid_surgery_data.csv'")
        print("  TREATMENT_COLS = ['oxycodone', 'hydrocodone', 'morphine', ...]")
        print("  OUTCOME_COLS = ['overdose']")
        exit(1)
    
    # Load data
    print(f"\n[1] Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"    Dataset shape: {df.shape}")
    
    # Validate columns
    missing_treatments = [t for t in TREATMENT_COLS if t not in df.columns]
    missing_outcomes = [o for o in OUTCOME_COLS if o not in df.columns]
    
    if missing_treatments:
        print(f"\n[ERROR] Treatment columns not found in data: {missing_treatments}")
        print(f"Available columns: {list(df.columns)}")
        exit(1)
    
    if missing_outcomes:
        print(f"\n[ERROR] Outcome columns not found in data: {missing_outcomes}")
        print(f"Available columns: {list(df.columns)}")
        exit(1)
    
    print(f"    Treatment columns: {len(TREATMENT_COLS)}")
    print(f"    Outcome columns: {len(OUTCOME_COLS)}")
    
    # Run GATE analysis for all treatment-outcome combinations
    print(f"\n[2] Running GATE analysis for {len(TREATMENT_COLS)} x {len(OUTCOME_COLS)} = {len(TREATMENT_COLS) * len(OUTCOME_COLS)} combinations...")
    
    results = run_gate_multiple_treatments(df, TREATMENT_COLS, OUTCOME_COLS)
    
    if results is not None:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: gate_results_all.csv")
        print(f"Total treatment-outcome pairs analyzed: {len(results['Treatment'].unique()) * len(results['Outcome'].unique())}")
    else:
        print("\n[ERROR] No results generated. Check your data and configuration.")
