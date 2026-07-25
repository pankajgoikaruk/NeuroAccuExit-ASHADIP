from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, torch
from common_v017 import load_feature
from models.anytime_exit_net import AnytimeExitState
from policies.sequential_anytime_exit import SequentialPolicyConfig, SequentialStageConfig

def subset_state(state: AnytimeExitState, indices: torch.Tensor) -> AnytimeExitState:
    hint=None if state.prev_hint is None else state.prev_hint.index_select(0,indices)
    return AnytimeExitState(feature_map=state.feature_map.index_select(0,indices),block_index=int(state.block_index),next_exit_index=int(state.next_exit_index),prev_hint=hint,finished=bool(state.finished))

def config_from_payload(payload: dict[str,Any]) -> SequentialPolicyConfig:
    stages=tuple(SequentialStageConfig(mean_confidence_threshold=float(x["mean_confidence_threshold"]),max_probability_delta=float(x["max_probability_delta"]),max_label_risk=float(x["max_label_risk"]),per_label_margins=tuple(float(v) for v in x["per_label_margins"]),require_previous_label_stability=bool(x.get("require_previous_label_stability",i>0))) for i,x in enumerate(payload["stages"]))
    return SequentialPolicyConfig(num_exits=int(payload["num_exits"]),stages=stages,allow_empty_stop=bool(payload.get("allow_empty_stop",False)))

def ablation_config(config: SequentialPolicyConfig,name:str):
    minimum=1; zero=False; stages=[]
    for stage in config.stages:
        item=stage
        if name=="no_stability": item=replace(item,require_previous_label_stability=False)
        elif name=="no_risk": item=replace(item,max_label_risk=1.0); zero=True
        elif name=="no_label_margins": item=replace(item,per_label_margins=tuple(0.0 for _ in item.per_label_margins))
        elif name=="confidence_only": item=replace(item,max_probability_delta=1.0,max_label_risk=1.0,per_label_margins=tuple(0.0 for _ in item.per_label_margins),require_previous_label_stability=False); zero=True
        stages.append(item)
    if name=="no_exit1": minimum=2
    return SequentialPolicyConfig(num_exits=config.num_exits,stages=tuple(stages),allow_empty_stop=config.allow_empty_stop),minimum,zero

def thresholds_from_policy(policy:dict[str,Any],labels:list[str])->list[np.ndarray]:
    return [np.asarray([float(policy["thresholds_by_exit"][f"exit{e}"][label]) for label in labels],dtype=np.float32) for e in range(1,int(policy["architecture"]["num_exits"])+1)]

def load_features(frame:pd.DataFrame,root:Path)->list[torch.Tensor]:
    tensors=[load_feature(root/Path(v.replace("\\","/"))) for v in frame["feat_relpath"].astype(str)]
    if len({tuple(t.shape) for t in tensors})!=1: raise RuntimeError("Holdout features have inconsistent shapes.")
    return tensors

def make_batches(length:int,batch_size:int)->list[np.ndarray]:
    return [np.arange(s,min(s+batch_size,length),dtype=np.int64) for s in range(0,length,batch_size)]
