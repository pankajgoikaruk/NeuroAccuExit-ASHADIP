from __future__ import annotations
import random
from types import SimpleNamespace
import torch
from common_v017 import synchronize
from eval_v017_reporting import timing_stats
from eval_v017_runtime import run_always_final,run_sequential

def execute(p):
    a=p.args
    with torch.no_grad():
        warm=torch.cat([p.tensors[int(i)] for i in p.batches[0]],dim=0).to(a.device); _,state=p.model.start(warm)
        while not state.finished: _,state=p.model.continue_from(state)
    synchronize(a.device); always,_=run_always_final(model=p.model,tensors=p.tensors,batches=p.batches,device=a.device,num_labels=len(p.labels),collect=True)
    names=["always_final",*p.methods]; times={n:[] for n in names}; rng=random.Random(int(a.timing_seed))
    for _ in range(int(a.timing_repeats)):
        order=names.copy(); rng.shuffle(order)
        for name in order:
            if name=="always_final": _,seconds=run_always_final(model=p.model,tensors=p.tensors,batches=p.batches,device=a.device,num_labels=len(p.labels),collect=False)
            else:
                item=p.methods[name]; _,timing=run_sequential(model=p.model,tensors=p.tensors,batches=p.batches,config=item["config"],thresholds=p.thresholds,risk_weights=item["risk_weights"],minimum_exit=item["minimum_exit"],device=a.device,num_labels=len(p.labels),collect=False); seconds=timing["total_seconds"]
            times[name].append(float(seconds))
    outputs={"always_final":always}; single={}
    for name,item in p.methods.items():
        output,timing=run_sequential(model=p.model,tensors=p.tensors,batches=p.batches,config=item["config"],thresholds=p.thresholds,risk_weights=item["risk_weights"],minimum_exit=item["minimum_exit"],device=a.device,num_labels=len(p.labels),collect=True); outputs[name]=output; single[name]=timing
    return SimpleNamespace(outputs=outputs,single=single,timing={n:timing_stats(v) for n,v in times.items()})
