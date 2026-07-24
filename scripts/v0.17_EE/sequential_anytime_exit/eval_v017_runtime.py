from __future__ import annotations
import time
import numpy as np, torch
from common_v017 import synchronize
from eval_v017_config import subset_state
from models.anytime_exit_net import AnytimeExitNet
from policies.sequential_anytime_exit import SequentialPolicyConfig, stage_diagnostics

def run_always_final(*,model:AnytimeExitNet,tensors,batches,device:str,num_labels:int,collect:bool):
    output=None if not collect else {"selected_probabilities":np.zeros((len(tensors),num_labels),np.float32),"selected_exit":np.full(len(tensors),model.num_exits,np.int8)}
    synchronize(device); started=time.perf_counter()
    with torch.no_grad():
        for indices in batches:
            x=torch.cat([tensors[int(i)] for i in indices],dim=0).to(device); logits,state=model.start(x)
            while not state.finished: logits,state=model.continue_from(state)
            if output is not None: output["selected_probabilities"][indices]=torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    synchronize(device); return output,float(time.perf_counter()-started)

def run_sequential(*,model:AnytimeExitNet,tensors,batches,config:SequentialPolicyConfig,thresholds:list[np.ndarray],risk_weights:np.ndarray,minimum_exit:int,device:str,num_labels:int,collect:bool):
    output=None if not collect else {"selected_probabilities":np.zeros((len(tensors),num_labels),np.float32),"selected_exit":np.full(len(tensors),model.num_exits,np.int8),"continuation_count":np.zeros(len(tensors),np.int8)}
    model_seconds=policy_seconds=0.0
    with torch.no_grad():
        for global_indices in batches:
            x=torch.cat([tensors[int(i)] for i in global_indices],dim=0).to(device); active=np.asarray(global_indices,np.int64); previous=None
            synchronize(device); started=time.perf_counter(); logits,state=model.start(x); synchronize(device); model_seconds+=time.perf_counter()-started
            for idx in range(model.num_exits):
                exit_no=idx+1; probs=torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                if exit_no==model.num_exits:
                    if output is not None: output["selected_probabilities"][active]=probs; output["selected_exit"][active]=exit_no
                    break
                policy_start=time.perf_counter(); diag=stage_diagnostics(current_probabilities=probs,current_thresholds=thresholds[idx],risk_weights=risk_weights[idx],previous_probabilities=previous,previous_thresholds=None if idx==0 else thresholds[idx-1]); stage=config.stages[idx]; stop=np.ones(len(active),bool)
                if exit_no<minimum_exit: stop[:]=False
                else:
                    if stage.require_previous_label_stability and idx>0: stop &= diag["label_set_stability"]
                    if not config.allow_empty_stop: stop &= diag["non_empty"]
                    stop &= diag["mean_binary_confidence"]>=stage.mean_confidence_threshold
                    stop &= diag["maximum_probability_delta"]<=stage.max_probability_delta
                    stop &= np.all(diag["decision_margin"]>=np.asarray(stage.per_label_margins,np.float32).reshape(1,-1),axis=1)
                    stop &= diag["maximum_label_risk"]<=stage.max_label_risk
                policy_seconds+=time.perf_counter()-policy_start
                if output is not None:
                    output["selected_probabilities"][active[stop]]=probs[stop]; output["selected_exit"][active[stop]]=exit_no; output["continuation_count"][active[~stop]]+=1
                continuing=np.flatnonzero(~stop)
                if len(continuing)==0: break
                state=subset_state(state,torch.as_tensor(continuing,dtype=torch.long,device=state.feature_map.device)); active=active[continuing]; previous=probs[continuing]
                synchronize(device); started=time.perf_counter(); logits,state=model.continue_from(state); synchronize(device); model_seconds+=time.perf_counter()-started
    return output,{"model_seconds":model_seconds,"policy_seconds":policy_seconds,"total_seconds":model_seconds+policy_seconds}
