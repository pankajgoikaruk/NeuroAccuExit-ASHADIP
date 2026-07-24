#!/usr/bin/env python
from __future__ import annotations
import argparse,sys
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent; PROJECT_ROOT=Path(__file__).resolve().parents[3]
for p in (SCRIPT_DIR,PROJECT_ROOT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from eval_v017_problem import prepare
from eval_v017_execute import execute
from eval_v017_artifacts import save_results

def parser():
    p=argparse.ArgumentParser(description="Evaluate frozen sequential policy")
    p.add_argument("--run_dir",required=True,type=Path); p.add_argument("--checkpoint",type=Path); p.add_argument("--policy_json",required=True,type=Path); p.add_argument("--holdout_manifest",required=True,type=Path); p.add_argument("--features_root",required=True,type=Path); p.add_argument("--labels_json",required=True,type=Path); p.add_argument("--lats_config_json",required=True,type=Path); p.add_argument("--parent_id_col",default="parent_clip_id"); p.add_argument("--batch_size",type=int,default=128); p.add_argument("--timing_repeats",type=int,default=10); p.add_argument("--timing_seed",type=int,default=42); p.add_argument("--torch_threads",type=int,default=1); p.add_argument("--device",default="cpu"); p.add_argument("--out_dir",required=True,type=Path); return p

def main():
    problem=prepare(parser().parse_args()); save_results(problem,execute(problem))
if __name__=="__main__": main()
