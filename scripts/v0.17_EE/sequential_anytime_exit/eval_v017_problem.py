from __future__ import annotations
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from common_v017 import load_checkpoint,load_json,load_labels,load_run_config,parse_tap_blocks,resolve_model_cfg
from eval_v017_config import ablation_config,config_from_payload,load_features,make_batches,thresholds_from_policy
from models.anytime_exit_net import AnytimeExitNet
from utils.model_factory import build_audio_exit_net

def prepare(args):
    torch.set_num_threads(max(1,int(args.torch_threads))); run=args.run_dir.resolve(); out=args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True); checkpoint=args.checkpoint.resolve() if args.checkpoint else run/"ckpt"/"best.pt"
    for path in (checkpoint,args.policy_json.resolve(),args.holdout_manifest.resolve(),args.features_root.resolve(),args.labels_json.resolve(),args.lats_config_json.resolve()):
        if not path.exists(): raise FileNotFoundError(f"Required path not found: {path}")
    cfg=load_run_config(run); labels=load_labels(args.labels_json.resolve(),cfg); policy=load_json(args.policy_json.resolve())
    if policy.get("experiment")!="v0.17_EE_sequential_active_budget_anytime_exit" or policy.get("labels")!=labels: raise RuntimeError("Frozen policy is incompatible with the checkpoint.")
    taps=parse_tap_blocks(cfg.get("tap_blocks","1,3")); exits=len(taps)+1; n_mels=int(cfg.get("n_mels",64))
    model=build_audio_exit_net(num_classes=len(labels),n_mels=n_mels,tap_blocks=taps,model_cfg=resolve_model_cfg(cfg)).to(args.device); load_checkpoint(model,checkpoint,args.device); model.eval(); anytime=AnytimeExitNet(model).to(args.device); anytime.eval()
    frame=pd.read_csv(args.holdout_manifest.resolve(),low_memory=False).reset_index(drop=True); tensors=load_features(frame,args.features_root.resolve()); batches=make_batches(len(frame),int(args.batch_size)); thresholds=thresholds_from_policy(policy,labels); config=config_from_payload(policy["selected_policy"]["parameters"]); risk=np.asarray(policy["risk_design"]["weights_by_early_exit"],np.float32); y=frame[labels].astype(int).to_numpy(np.int8)
    methods={}
    for name in ["full_sequential","no_exit1","no_stability","no_risk","no_label_margins","confidence_only"]:
        if name=="full_sequential": c,minimum,zero=config,1,False
        else: c,minimum,zero=ablation_config(config,name)
        methods[name]={"config":c,"minimum_exit":minimum,"risk_weights":np.zeros_like(risk) if zero else risk,"validation_eligible":bool(policy["selected_policy"]["deployment_eligible"] if name=="full_sequential" else False)}
    return SimpleNamespace(args=args,run=run,out=out,checkpoint=checkpoint,cfg=cfg,labels=labels,policy=policy,taps=taps,exits=exits,n_mels=n_mels,model=anytime,frame=frame,tensors=tensors,batches=batches,thresholds=thresholds,methods=methods,y=y)
