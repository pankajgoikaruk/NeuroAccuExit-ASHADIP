#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent; PROJECT_ROOT=Path(__file__).resolve().parents[3]
for p in (SCRIPT_DIR,PROJECT_ROOT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from tune_v017_problem import prepare_problem
from tune_v017_search import optimise
from tune_v017_artifacts import save_tuning

def parser():
    p=argparse.ArgumentParser(description="Tune sequential 3/5-exit policy")
    p.add_argument("--run_dir",required=True,type=Path); p.add_argument("--checkpoint",type=Path); p.add_argument("--manifest",type=Path); p.add_argument("--features_root",type=Path); p.add_argument("--labels_json",required=True,type=Path); p.add_argument("--lats_config_json",required=True,type=Path); p.add_argument("--parent_id_col",default="parent_clip_id")
    p.add_argument("--threshold_mode",choices=["tuned_per_exit","final_exit_tuned","fixed_0p5"],default="fixed_0p5"); p.add_argument("--fixed_threshold",type=float,default=.5); p.add_argument("--population_size",type=int,default=96); p.add_argument("--generations",type=int,default=60); p.add_argument("--cv_folds",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--crossover_probability",type=float,default=.90); p.add_argument("--mutation_probability",type=float,default=.18); p.add_argument("--mutation_scale",type=float,default=.06)
    p.add_argument("--confidence_bounds",default="0.50,0.99"); p.add_argument("--delta_bounds",default="0.00,1.00"); p.add_argument("--risk_bounds",default="0.00,1.00"); p.add_argument("--margin_bounds",default="0.00,0.50"); p.add_argument("--max_macro_f1_drop",type=float,default=.01); p.add_argument("--max_micro_f1_drop",type=float,default=.005); p.add_argument("--max_exact_match_drop",type=float,default=.01); p.add_argument("--max_hamming_increase",type=float,default=.002); p.add_argument("--min_total_early_fraction",type=float,default=.02); p.add_argument("--min_exit1_fraction",type=float,default=.005); p.add_argument("--safety_fraction",type=float,default=.75); p.add_argument("--batch_size",type=int); p.add_argument("--num_workers",type=int,default=0); p.add_argument("--device",default="cpu"); p.add_argument("--out_dir",required=True,type=Path); return p

def main():
    args=parser().parse_args(); problem=prepare_problem(args); save_tuning(problem,optimise(problem))
if __name__=="__main__": main()
