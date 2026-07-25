from __future__ import annotations
import subprocess,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[3]

def evaluate_parent_lats(*,segment_csv:Path,labels_json:Path,lats_config_json:Path,out_dir:Path,parent_id_col:str,model_name:str)->dict[str,Any]:
    cmd=[sys.executable,str(PROJECT_ROOT/"scripts"/"v0.10"/"evaluate_frozen_lats_config_v010.py"),"--segment-pred-csv",str(segment_csv),"--labels-json",str(labels_json),"--config-json",str(lats_config_json),"--out-dir",str(out_dir),"--parent-id-col",parent_id_col,"--prob-prefix","dynamic_prob_","--model-name",model_name]
    subprocess.run(cmd,check=True); return pd.read_csv(out_dir/"v010_frozen_lats_eval.csv").iloc[0].to_dict()

def timing_stats(values:list[float])->dict[str,float]:
    a=np.asarray(values,float); return {"median_seconds":float(np.median(a)),"q1_seconds":float(np.quantile(a,.25)),"q3_seconds":float(np.quantile(a,.75)),"iqr_seconds":float(np.quantile(a,.75)-np.quantile(a,.25)),"mean_seconds":float(np.mean(a)),"std_seconds":float(np.std(a,ddof=1)) if len(a)>1 else 0.0,"repeats":len(a)}
