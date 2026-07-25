from __future__ import annotations
from typing import Any
import numpy as np,pandas as pd
from common_v017 import jsonable,multilabel_metrics,save_json
from eval_v017_reporting import evaluate_parent_lats
from utils.profiling import estimate_flops_tiny_audiocnn

def save_results(p,e):
    a=p.args; baseline=e.timing["always_final"]["median_seconds"]; flops=estimate_flops_tiny_audiocnn(n_mels=p.n_mels,frames=int(p.tensors[0].shape[-1]),num_classes=len(p.labels),tap_blocks=p.taps); rows=[]; summaries={}
    for name,output in e.outputs.items():
        method_dir=p.out/name; method_dir.mkdir(parents=True,exist_ok=True); probs=output["selected_probabilities"]; selected=output["selected_exit"]; pred=np.zeros_like(p.y,np.int8); fractions={}
        for exit_no in range(1,p.exits+1):
            mask=selected==exit_no; fractions[exit_no]=float(np.mean(mask))
            if np.any(mask): pred[mask]=(probs[mask]>=p.thresholds[exit_no-1].reshape(1,-1)).astype(np.int8)
        segment=multilabel_metrics(p.y,pred); frame=p.frame[[a.parent_id_col,"feat_relpath",*p.labels]].copy()
        for i,label in enumerate(p.labels): frame[f"dynamic_prob_{label}"]=probs[:,i]; frame[f"dynamic_pred_{label}"]=pred[:,i]
        frame["selected_exit"]=selected; frame["continuation_reason"]=np.where(selected==p.exits,"reached_final_exit",np.asarray([f"stopped_at_exit{x}" for x in selected])); csv=method_dir/"segment_predictions.csv"; frame.to_csv(csv,index=False)
        parent=evaluate_parent_lats(segment_csv=csv,labels_json=a.labels_json.resolve(),lats_config_json=a.lats_config_json.resolve(),out_dir=method_dir/"parent_frozen_lats_v2",parent_id_col=a.parent_id_col,model_name=f"{p.exits}exit_{name}")
        avg=sum(fractions[x]*float(flops[f"exit{x}"]) for x in fractions); saved=100*(1-avg/max(float(flops[f"exit{p.exits}"]),1.)); timing=e.timing[name]
        row={"architecture":f"{p.exits}-exit","method":name,"validation_eligible":bool(name=="always_final" or p.methods[name]["validation_eligible"]),"total_early_fraction":1-fractions[p.exits],"average_exit_depth":float(np.mean(selected)),"estimated_flops_saved_pct":saved,"latency_median_per_segment_ms":1000*timing["median_seconds"]/len(p.frame),"latency_iqr_per_segment_ms":1000*timing["iqr_seconds"]/len(p.frame),"measured_speedup_vs_always_final":baseline/max(timing["median_seconds"],1e-12),**{f"segment_{k}":v for k,v in segment.items()},**{f"parent_{k}":v for k,v in parent.items() if isinstance(v,(int,float,np.integer,np.floating))}}
        for x in range(1,p.exits+1): row[f"exit{x}_fraction"]=fractions[x]
        rows.append(row); summary={**row,"timing":timing,"single_pass_timing":e.single.get(name),"policy":None if name=="always_final" else {"parameters":p.methods[name]["config"].to_dict(),"minimum_exit":p.methods[name]["minimum_exit"]},"genuine_skipping_statement":"Each sample stopped at its selected exit and did not execute later blocks."}; save_json(summary,method_dir/"runtime_summary.json"); summaries[name]=summary
    comparison=pd.DataFrame(rows); path=p.out/f"v017_{p.exits}exit_holdout_comparison.csv"; comparison.to_csv(path,index=False); ablation=p.out/f"v017_{p.exits}exit_ablation_table.csv"; comparison[comparison["method"]!="always_final"].to_csv(ablation,index=False); save_json({"experiment":"v0.17_EE_sequential_active_budget_anytime_exit","architecture":f"{p.exits}-exit","comparison":[jsonable(r) for r in rows],"methods":summaries,"important_note":"Holdout used one frozen policy; ablations are not retuned."},p.out/f"v017_{p.exits}exit_holdout_comparison.json")
    cols=["architecture","method","validation_eligible",*[f"exit{x}_fraction" for x in range(1,p.exits+1)],"estimated_flops_saved_pct","measured_speedup_vs_always_final","parent_macro_f1","parent_micro_f1","parent_exact_match","parent_hamming_loss"]; print(f"\nV0.17 sequential {p.exits}-exit holdout comparison complete"); print("-"*170); print(comparison[cols].to_string(index=False)); print(f"Saved comparison: {path}\nSaved ablation: {ablation}")
