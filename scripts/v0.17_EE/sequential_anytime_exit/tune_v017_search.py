from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
from common_v017 import evaluate_sequential_candidate, objective_matrix
from policies.sequential_anytime_exit import decode_sequential_genes, environmental_select, make_offspring, pareto_front_mask, sequential_select
from tune_v017_helpers import objective_vector, select_buffered_pareto

def optimise(p):
    a=p.args; cache:dict[tuple[float,...],dict[str,Any]]={}; order=[]; population=p.population
    def evaluate(genes):
        key=tuple(float(x) for x in np.round(genes,6))
        if key in cache: return cache[key]
        config=decode_sequential_genes(genes,num_exits=p.num_exits,num_labels=len(p.labels))
        selection=sequential_select(exit_probabilities=p.probs,thresholds_by_exit=p.thresholds,risk_weights_by_exit=p.risk,config=config)
        row=evaluate_sequential_candidate(strategy=f"sequential_{p.num_exits}exit",parameters=config.to_dict(),selected_probabilities=selection["selected_probabilities"],selected_exit=selection["selected_exit"],y_true=p.y,thresholds_by_exit=p.thresholds,parent_context=p.parent,flops_by_exit=p.flops,max_macro_drop=a.max_macro_f1_drop,max_micro_drop=a.max_micro_f1_drop,max_exact_drop=a.max_exact_match_drop,max_hamming_increase=a.max_hamming_increase,min_total_early_fraction=a.min_total_early_fraction,min_exit1_fraction=a.min_exit1_fraction)
        row["candidate_id"]=len(cache); row["genes_json"]=json.dumps([float(x) for x in genes]); cache[key]=row; order.append(key); return row
    history=[]
    for generation in range(int(a.generations)):
        rows=[evaluate(g) for g in population]; objectives=np.vstack([objective_vector(r) for r in rows]); violations=np.asarray([r["constraint_violation"] for r in rows],float); feasible=[r for r in rows if r["quality_constraints_met"]]
        history.append({"generation":generation,"unique_candidates":len(cache),"feasible_population":len(feasible),"best_feasible_flops_saved_pct":max([r["estimated_flops_saved_pct"] for r in feasible],default=0.0),"minimum_constraint_violation":float(violations.min())})
        offspring=make_offspring(population=population,objectives=objectives,violations=violations,lower=p.lower,upper=p.upper,rng=p.rng,crossover_probability=a.crossover_probability,mutation_probability=a.mutation_probability,mutation_scale=a.mutation_scale)
        offrows=[evaluate(g) for g in offspring]
        population,_,_=environmental_select(population=np.vstack([population,offspring]),objectives=np.vstack([objectives,np.vstack([objective_vector(r) for r in offrows])]),violations=np.concatenate([violations,np.asarray([r["constraint_violation"] for r in offrows],float)]),size=int(a.population_size))
    all_df=pd.DataFrame([cache[k] for k in order]).drop_duplicates("genes_json"); final=pd.DataFrame([evaluate(g) for g in population]).drop_duplicates("genes_json")
    pareto=final.loc[pareto_front_mask(objective_matrix(final),final["constraint_violation"].to_numpy(float))].copy(); feasible=all_df[all_df["quality_constraints_met"]==True].copy()
    if feasible.empty:
        selected=all_df.sort_values(["constraint_violation","estimated_flops_saved_pct"],ascending=[True,False]).iloc[0]; status="fallback_minimum_constraint_violation"; eligible=False
    else:
        selected,status=select_buffered_pareto(feasible,max_macro=a.max_macro_f1_drop,max_micro=a.max_micro_f1_drop,max_exact=a.max_exact_match_drop,max_hamming=a.max_hamming_increase,safety_fraction=a.safety_fraction); eligible=True
    return SimpleNamespace(all_df=all_df,pareto=pareto,history=history,selected=selected,status=status,eligible=eligible)
