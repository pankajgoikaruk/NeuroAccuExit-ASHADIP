from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

def parse_pair(text: str, name: str) -> tuple[float, float]:
    values=[float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values)!=2 or values[0]>=values[1]:
        raise ValueError(f"{name} must contain two increasing values.")
    return values[0], values[1]

def objective_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([row[f"objective_{k}"] for k in ("compute","macro","micro","exact","hamming")], dtype=np.float64)

def quantile_margin_seed(*, y_true: np.ndarray, probabilities: list[np.ndarray], thresholds: list[np.ndarray], quantile: float) -> np.ndarray:
    final=(probabilities[-1]>=thresholds[-1].reshape(1,-1)).astype(np.int8)
    values:list[float]=[]
    for i in range(len(probabilities)-1):
        current=(probabilities[i]>=thresholds[i].reshape(1,-1)).astype(np.int8)
        corrected=(current!=y_true)&(final==y_true)
        margins=np.abs(probabilities[i]-thresholds[i].reshape(1,-1))
        per=np.zeros(y_true.shape[1])
        for j in range(y_true.shape[1]):
            observed=margins[corrected[:,j],j]
            if len(observed)>=3: per[j]=float(np.quantile(observed,quantile))
        values.extend([min(.97,.90-.05*i), 1.0 if i==0 else min(.40,.12+.08*i), min(1.0,.25+.15*i), *per.tolist()])
    return np.asarray(values,dtype=np.float64)

def select_buffered_pareto(feasible: pd.DataFrame, *, max_macro: float, max_micro: float, max_exact: float, max_hamming: float, safety_fraction: float) -> tuple[pd.Series,str]:
    frame=feasible.copy()
    frame["quality_utilisation"]=np.max(np.column_stack([frame["robust_macro_drop"]/max(max_macro,1e-9),frame["robust_micro_drop"]/max(max_micro,1e-9),frame["robust_exact_drop"]/max(max_exact,1e-9),frame["robust_hamming_increase"]/max(max_hamming,1e-9)]),axis=1)
    buffered=frame[frame["quality_utilisation"]<=float(safety_fraction)].copy()
    pool=buffered if not buffered.empty else frame
    compute=pool["estimated_flops_saved_pct"].to_numpy(float)
    norm=np.ones(len(pool)) if float(np.ptp(compute))<=1e-12 else (compute-compute.min())/np.ptp(compute)
    pool["buffered_knee_score"]=norm-.65*np.clip(pool["quality_utilisation"].to_numpy(float),0,2)
    selected=pool.sort_values(["buffered_knee_score","estimated_flops_saved_pct","parent_macro_f1"],ascending=[False,False,False]).iloc[0]
    return selected, "safety_buffered_pareto_knee" if not buffered.empty else "feasible_pareto_knee_no_buffered_candidate"
