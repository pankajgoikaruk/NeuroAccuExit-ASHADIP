from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
LABELS=['Brene_Brown','Eckhart_Tolle','Eric_Thomas','Gary_Vee','Jay_Shetty','Nick_Vujicic','other_speaker_present','music_present','audience_reaction_present','silence_present']
METHODS=['mean','max','top2mean','top3mean','top4mean','top5mean','median','p75','p90','noisy_or']
THRESHOLDS=np.round(np.arange(0.10,0.95+1e-9,0.01),2)
LATS_V2_BASELINE_CONFIG={
"Brene_Brown":{"aggregation":"top3mean","threshold":0.50},"Eckhart_Tolle":{"aggregation":"top2mean","threshold":0.50},"Eric_Thomas":{"aggregation":"mean","threshold":0.54},"Gary_Vee":{"aggregation":"top3mean","threshold":0.50},"Jay_Shetty":{"aggregation":"mean","threshold":0.82},"Nick_Vujicic":{"aggregation":"mean","threshold":0.43},"other_speaker_present":{"aggregation":"top3mean","threshold":0.76},"music_present":{"aggregation":"mean","threshold":0.49},"audience_reaction_present":{"aggregation":"max","threshold":0.68},"silence_present":{"aggregation":"p75","threshold":0.34}}

def agg_arr(vals,method):
    vals=np.asarray(vals,dtype=float); vals=vals[~np.isnan(vals)]
    if len(vals)==0:return 0.0
    if method=='mean':return float(np.mean(vals))
    if method=='max':return float(np.max(vals))
    if method=='median':return float(np.median(vals))
    if method=='p75':return float(np.percentile(vals,75))
    if method=='p90':return float(np.percentile(vals,90))
    if method=='noisy_or':return float(1-np.prod(1-np.clip(vals,0,1)))
    m=re.fullmatch(r'top(\d+)mean',method)
    if m:
        k=int(m.group(1)); top=np.sort(vals)[::-1][:min(k,len(vals))]; return float(np.mean(top))
    raise ValueError(method)

def build(df):
    parents=sorted(df.parent_clip_id.dropna().unique())
    ytrue=df.groupby('parent_clip_id')[LABELS].max().loc[parents].astype(int).to_numpy()
    scores={m:np.zeros((len(parents),len(LABELS)),dtype=float) for m in METHODS}
    g=df.groupby('parent_clip_id',sort=True)
    for li,lab in enumerate(LABELS):
        col='exit3_prob_'+lab
        for mi,m in enumerate(METHODS):
            if m=='mean': s=g[col].mean()
            elif m=='max': s=g[col].max()
            elif m=='median': s=g[col].median()
            else: s=g[col].apply(lambda x,mm=m: agg_arr(x.to_numpy(),mm))
            scores[m][:,li]=s.loc[parents].to_numpy(dtype=float)
    return parents,ytrue,scores

def metrics(ytrue,ypred):
    ytrue=np.asarray(ytrue,dtype=int); ypred=np.asarray(ypred,dtype=int)
    tp=((ytrue==1)&(ypred==1)).sum(axis=0).astype(float)
    fp=((ytrue==0)&(ypred==1)).sum(axis=0).astype(float)
    fn=((ytrue==1)&(ypred==0)).sum(axis=0).astype(float)
    prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    rec=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
    f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp),where=(prec+rec)>0)
    micro_tp=tp.sum(); micro_fp=fp.sum(); micro_fn=fn.sum()
    micro_p=micro_tp/(micro_tp+micro_fp) if (micro_tp+micro_fp)>0 else 0.0
    micro_r=micro_tp/(micro_tp+micro_fn) if (micro_tp+micro_fn)>0 else 0.0
    micro_f=2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r)>0 else 0.0
    row_tp=((ytrue==1)&(ypred==1)).sum(axis=1).astype(float)
    row_true=ytrue.sum(axis=1).astype(float); row_pred=ypred.sum(axis=1).astype(float)
    samples=np.divide(2*row_tp,row_true+row_pred,out=np.zeros_like(row_tp),where=(row_true+row_pred)>0)
    inter=row_tp
    union=((ytrue==1)|(ypred==1)).sum(axis=1).astype(float)
    jac=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    return {'macro_f1':float(f1.mean()), 'micro_f1':float(micro_f), 'samples_f1':float(samples.mean()), 'exact_match':float(np.mean(np.all(ytrue==ypred,axis=1))), 'hamming_loss':float(np.mean(ytrue!=ypred)), 'jaccard_score':float(jac.mean()), 'avg_true_labels':float(row_true.mean()), 'avg_pred_labels':float(row_pred.mean()), 'parent_clips':int(ytrue.shape[0]), 'num_labels':int(ytrue.shape[1])}

def objective(m):
    return 0.40*m['macro_f1']+0.20*m['micro_f1']+0.20*m['samples_f1']+0.15*m['exact_match']-0.05*m['hamming_loss']-0.05*abs(m['avg_pred_labels']-m['avg_true_labels'])

def pred_from_config(scores,cfg):
    yp=[]
    for li,lab in enumerate(LABELS):
        c=cfg[lab]; yp.append((scores[c['aggregation']][:,li]>=float(c['threshold'])).astype(int))
    return np.stack(yp,axis=1)

def lats_v1(ytrue,scores):
    cfg={}; rows=[]; pref={m:i for i,m in enumerate(['mean','top2mean','top3mean','top4mean','top5mean','median','p75','p90','max','noisy_or'])}
    for li,lab in enumerate(LABELS):
        yt=ytrue[:,li]; best=None
        for m in METHODS:
            sc=scores[m][:,li]
            for t in THRESHOLDS:
                yp=(sc>=t).astype(int)
                tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum()); p=float(tp/(tp+fp)) if (tp+fp)>0 else 0.0; r=float(tp/(tp+fn)) if (tp+fn)>0 else 0.0; f=float(2*p*r/(p+r)) if (p+r)>0 else 0.0
                err=fp+fn
                row={'label':lab,'aggregation':m,'threshold':float(t),'precision':p,'recall':r,'f1':f,'fp':fp,'fn':fn,'hamming_errors':err}
                key=(f,-err,-fp,-abs(float(t)-0.5),-pref.get(m,99))
                if best is None or key>best[0]: best=(key,row)
        cfg[lab]={'aggregation':best[1]['aggregation'],'threshold':best[1]['threshold']}
        rows.append(best[1])
    return cfg,pd.DataFrame(rows)

def coordinate(ytrue,scores,start_cfg,max_passes=20):
    cfg=json.loads(json.dumps(start_cfg)); ypred=pred_from_config(scores,cfg)
    cur=metrics(ytrue,ypred); cur_score=objective(cur)
    hist=[{'pass':0,'label':'START','score':cur_score,**cur}]
    for p in range(1,max_passes+1):
        changed=False
        for li,lab in enumerate(LABELS):
            best_score=cur_score; best=None; best_predcol=None; best_m=None
            for m in METHODS:
                sc=scores[m][:,li]
                for t in THRESHOLDS:
                    predcol=(sc>=t).astype(int)
                    if np.array_equal(predcol, ypred[:,li]) and m==cfg[lab]['aggregation'] and abs(float(t)-cfg[lab]['threshold'])<1e-12:
                        continue
                    trial=ypred.copy(); trial[:,li]=predcol
                    mt=metrics(ytrue,trial); scobj=objective(mt)
                    if scobj>best_score+1e-12:
                        best_score=scobj; best={'aggregation':m,'threshold':float(t)}; best_predcol=predcol; best_m=mt
            if best is not None:
                cfg[lab]=best; ypred[:,li]=best_predcol; cur=best_m; cur_score=best_score; changed=True
                hist.append({'pass':p,'label':lab,'aggregation':best['aggregation'],'threshold':best['threshold'],'score':cur_score,**cur})
        if not changed: break
    return cfg,pd.DataFrame(hist)

def per_label(ytrue,ypred,cfg):
    rows=[]
    for li,lab in enumerate(LABELS):
        yt=ytrue[:,li]; yp=ypred[:,li]
        tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum()); tn=int(((yt==0)&(yp==0)).sum()); prec=float(tp/(tp+fp)) if (tp+fp)>0 else 0.0; rec=float(tp/(tp+fn)) if (tp+fn)>0 else 0.0; f1=float(2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0; rows.append({'label':lab,'aggregation':cfg[lab]['aggregation'],'threshold':cfg[lab]['threshold'],'precision':prec,'recall':rec,'f1':f1,'support':int(yt.sum()),'predicted_positive':int(yp.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'hamming_errors':int((yt!=yp).sum())})
    return pd.DataFrame(rows)

def save_config(path,name,run,cfg,ytrue,ypred):
    m=metrics(ytrue,ypred); m['objective']=objective(m)
    with open(path,'w',encoding='utf-8') as f: json.dump({'name':name,'run':run,'config':cfg,'metrics':m},f,indent=2)
    return m

def run_one(run,csv,out):
    df=pd.read_csv(csv,low_memory=False); parents,ytrue,scores=build(df)
    rows=[]
    for name,cfg in [('frozen_v09_lats_v2_transfer',LATS_V2_BASELINE_CONFIG)]:
        yp=pred_from_config(scores,cfg); m=save_config(out/f'{run}_{name}_config.json',name,run,cfg,ytrue,yp); m.update({'run':run,'method':name}); rows.append(m); per_label(ytrue,yp,cfg).to_csv(out/f'{run}_{name}_per_label.csv',index=False)
    cfg1,sel=lats_v1(ytrue,scores); yp=pred_from_config(scores,cfg1); m=save_config(out/f'{run}_lats_v1_reoptimized_config.json','lats_v1_reoptimized',run,cfg1,ytrue,yp); m.update({'run':run,'method':'v010_lats_v1_labelwise_reoptimized'}); rows.append(m); per_label(ytrue,yp,cfg1).to_csv(out/f'{run}_lats_v1_reoptimized_per_label.csv',index=False); sel.to_csv(out/f'{run}_lats_v1_selected_rules.csv',index=False)
    cfg2,hist=coordinate(ytrue,scores,cfg1); yp=pred_from_config(scores,cfg2); m=save_config(out/f'{run}_lats_v2_coordinate_reoptimized_config.json','lats_v2_coordinate_reoptimized_from_lats_v1',run,cfg2,ytrue,yp); m.update({'run':run,'method':'v010_lats_v2_metric_coordinate_reoptimized'}); rows.append(m); per_label(ytrue,yp,cfg2).to_csv(out/f'{run}_lats_v2_coordinate_reoptimized_per_label.csv',index=False); hist.to_csv(out/f'{run}_lats_v2_coordinate_history.csv',index=False)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--no-hint-csv',required=True); ap.add_argument('--hint-csv',required=True); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    baseline={'run':'v09_4_baseline','method':'frozen_lats_v2_baseline_recheck','macro_f1':0.867256,'micro_f1':0.945786,'samples_f1':0.951700,'exact_match':0.860438,'hamming_loss':0.015802,'jaccard_score':0.930892,'avg_true_labels':1.469435,'avg_pred_labels':1.445213,'parent_clips':867,'num_labels':10}
    baseline['objective']=objective(baseline)
    rows=[baseline]
    rows += run_one('no_hint',args.no_hint_csv,out)
    rows += run_one('hint_pass',args.hint_csv,out)
    df=pd.DataFrame(rows)
    cols=['run','method','macro_f1','micro_f1','samples_f1','exact_match','hamming_loss','avg_pred_labels','objective']
    df[cols].to_csv(out/'v010_lats_reoptimized_comparison_summary.csv',index=False)
    print(df[cols].to_string(index=False), flush=True)
if __name__=='__main__':
    main()
