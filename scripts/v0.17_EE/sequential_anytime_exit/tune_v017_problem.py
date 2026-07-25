from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common_v017 import ParentMetricContext, collect_outputs, load_checkpoint, load_labels, load_run_config, load_thresholds_by_exit, parse_tap_blocks, resolve_model_cfg
from data.datasets_multilabel import make_multilabel_loaders
from policies.sequential_anytime_exit import derive_validation_risk_weights, make_sequential_bounds, random_population
from tune_v017_helpers import parse_pair, quantile_margin_seed
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn

def prepare_problem(args):
    run_dir=args.run_dir.resolve(); out_dir=args.out_dir.resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    cfg=load_run_config(run_dir)
    manifest=args.manifest.resolve() if args.manifest else Path(cfg["manifest"]).resolve()
    features=args.features_root.resolve() if args.features_root else Path(cfg["features_root"]).resolve()
    checkpoint=args.checkpoint.resolve() if args.checkpoint else run_dir/"ckpt"/"best.pt"
    for path in (manifest,features,checkpoint,args.labels_json.resolve(),args.lats_config_json.resolve()):
        if not path.exists(): raise FileNotFoundError(f"Required path not found: {path}")
    labels=load_labels(args.labels_json.resolve(),cfg); taps=parse_tap_blocks(cfg.get("tap_blocks","1,3")); exits=len(taps)+1
    if exits not in (3,5): raise RuntimeError(f"v0.17 supports exactly 3 or 5 exits, got {exits}.")
    n_mels=int(cfg.get("n_mels",64)); batch=int(args.batch_size or cfg.get("batch_size",64))
    train,val,test,loaded=make_multilabel_loaders(manifest_csv=manifest,features_root=features,labels_json=args.labels_json.resolve(),batch_size=batch,num_workers=int(args.num_workers),seed=int(cfg.get("seed",args.seed)),label_balance_power=0.0,synthetic_balance_power=0.0)
    del train,test
    if list(loaded)!=labels: raise RuntimeError("Label order mismatch between schema and loader.")
    metadata=val.dataset.df.reset_index(drop=True)
    model=build_audio_exit_net(num_classes=len(labels),n_mels=n_mels,tap_blocks=taps,model_cfg=resolve_model_cfg(cfg)).to(args.device)
    load_checkpoint(model,checkpoint,args.device); model.eval()
    y,probs,frames=collect_outputs(model,val,args.device)
    thresholds=load_thresholds_by_exit(run_dir=run_dir,labels=labels,num_exits=exits,threshold_mode=args.threshold_mode,fixed_threshold=args.fixed_threshold)
    risk,risk_counts=derive_validation_risk_weights(y_true=y,exit_probabilities=probs,thresholds_by_exit=thresholds)
    parent=ParentMetricContext.build(metadata_df=metadata,labels=labels,parent_id_col=args.parent_id_col,lats_config_json=args.lats_config_json.resolve(),reference_probabilities=probs[-1],cv_folds=args.cv_folds)
    flops=estimate_flops_tiny_audiocnn(n_mels=n_mels,frames=frames,num_classes=len(labels),tap_blocks=taps)
    lower,upper=make_sequential_bounds(num_exits=exits,num_labels=len(labels),confidence_bounds=parse_pair(args.confidence_bounds,"confidence_bounds"),delta_bounds=parse_pair(args.delta_bounds,"delta_bounds"),risk_bounds=parse_pair(args.risk_bounds,"risk_bounds"),margin_bounds=parse_pair(args.margin_bounds,"margin_bounds"))
    conservative=[]; permissive=[]; graduated=[]
    for i in range(exits-1):
        conservative += [.99,.01,.01,*([.50]*len(labels))]
        permissive += [.50,1.,1.,*([0.]*len(labels))]
        graduated += [max(.60,.95-.08*i),1. if i==0 else min(.50,.12+.10*i),min(.90,.20+.15*i),*([max(.02,.20-.04*i)]*len(labels))]
    seeds=[np.asarray(x,float) for x in (conservative,permissive,graduated)]
    seeds += [quantile_margin_seed(y_true=y,probabilities=probs,thresholds=thresholds,quantile=q) for q in (.50,.75,.90)]
    rng=np.random.default_rng(int(args.seed)); population=random_population(size=int(args.population_size),lower=lower,upper=upper,rng=rng,seeds=seeds)
    return SimpleNamespace(args=args,run_dir=run_dir,out_dir=out_dir,cfg=cfg,manifest=manifest,features=features,checkpoint=checkpoint,labels=labels,taps=taps,num_exits=exits,n_mels=n_mels,y=y,probs=probs,frames=frames,thresholds=thresholds,risk=risk,risk_counts=risk_counts,parent=parent,flops=flops,lower=lower,upper=upper,rng=rng,population=population)
