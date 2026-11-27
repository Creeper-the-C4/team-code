"""Task 1 fixed LightGBM baseline on independently prepared 2 s features.

All 13 labels and all trial-local windows are retained. Each outer test subject
is predicted exactly once. Early stopping uses only the fixed inner validation
subjects. No hyperparameter search, smoothing, imputation or scaler fitting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','12')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

LABELS=list(range(13))
NAMES=['Others','lying','sitting','standing','slow_walking','moderate_walking',
       'brisk_walking','ascending_stairs','descending_stairs','cycling',
       'running','jumping','rowing']


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def metrics(y,pred):
    cm=confusion_matrix(y,pred,labels=LABELS)
    return metrics_from_cm(cm)


def metrics_from_cm(cm):
    true=cm.sum(axis=1).astype(float)
    predicted=cm.sum(axis=0).astype(float)
    tp=cm.diagonal().astype(float)
    f1=np.divide(2*tp,true+predicted,out=np.zeros(13),where=(true+predicted)>0)
    recall=np.divide(tp,true,out=np.zeros(13),where=true>0)
    return {'n_windows':int(cm.sum()),'accuracy_13':float(tp.sum()/cm.sum()) if cm.sum() else None,
            'macro_f1_13':float(f1.mean()),'macro_f1_named12_all_windows':float(f1[1:].mean()),
            'walk3_macro_f1_all_windows':float(f1[4:7].mean()),
            'balanced_accuracy_present_classes':float(recall[true>0].mean()) if np.any(true>0) else None}


def eval_macro(y_true,probability):
    assert probability.ndim==2 and probability.shape[1]==13
    score=metrics(y_true,probability.argmax(axis=1))['macro_f1_13']
    return 'macro_f1_13',score,True


def load_subject(root,subject):
    path=root/'subjects'/f'P{subject:02d}'
    if not (path/'features_2s_complete.json').exists():raise ValueError('Feature cache incomplete')
    f=np.load(path/'baseline_features_2s.npy',mmap_mode='r')
    w=pd.read_parquet(path/'windows_2s.parquet')
    assert len(f)==len(w) and f.shape[1:]==(17,40)
    assert (w.subject==subject).all()
    return f.reshape(len(f),-1),w


def load_split(root,subjects,cap_per_class=None):
    arrays=[];frames=[]
    for subject in subjects:
        f,w=load_subject(root,subject)
        w=w.copy();w['window_id']=np.arange(len(w))
        arrays.append(f);frames.append(w)
    x=np.concatenate(arrays,axis=0).astype(np.float32,copy=False)
    meta=pd.concat(frames,ignore_index=True)
    y=meta.majority_label.to_numpy(dtype=np.int32)
    if cap_per_class is not None:
        rng=np.random.default_rng(42)
        selected=np.concatenate([rng.choice(np.flatnonzero(y==i),min(cap_per_class,int((y==i).sum())),replace=False) for i in LABELS])
        selected.sort();x=x[selected];meta=meta.iloc[selected].reset_index(drop=True);y=y[selected]
    assert np.isin(y,LABELS).all() and not np.isinf(x).any()
    return x,y,meta


def per_class(y,pred):
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
    return pd.DataFrame({'activity':LABELS,'name':NAMES,'precision':precision,
                         'recall':recall,'f1':f1,'support':support})


def formal_fold(root,out,split,config):
    fold=split['fold'];dest=out/f'fold_{fold}';dest.mkdir(exist_ok=True)
    done=dest/'complete.json'
    if done.exists():
        print(f'FOLD {fold} already complete',flush=True);return
    train,val,test=split['train'],split['validation'],split['test']
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    print(f'FOLD {fold} TRAIN_START train={train} validation={val} held_out_test={test}',flush=True)
    start=time.monotonic()
    x,y,_=load_split(root,train)
    vx,vy,_=load_split(root,val)
    assert np.array_equal(np.unique(y),np.arange(13))
    model=lgb.LGBMClassifier(**config['model'])
    history={}
    model.fit(x,y,eval_set=[(vx,vy)],eval_names=['inner_validation'],eval_metric=eval_macro,
              callbacks=[lgb.early_stopping(config['early_stopping_rounds'],first_metric_only=True,verbose=False),
                         lgb.log_evaluation(25),lgb.record_evaluation(history)])
    elapsed_fit=time.monotonic()-start
    model.booster_.save_model(str(dest/'model.txt'))
    save_json(dest/'validation_history.json',history)
    del x,y,vx,vy
    print(f'FOLD {fold} TRAIN_DONE best_iteration={model.best_iteration_} training_seconds={elapsed_fit:.1f}; TEST_ONCE_START',flush=True)
    # Test feature values are loaded only after fitting and early stopping finish.
    tx,ty,meta=load_split(root,test)
    prob=model.predict_proba(tx,num_iteration=model.best_iteration_).astype(np.float32)
    assert prob.shape==(len(ty),13) and np.isfinite(prob).all()
    assert np.allclose(prob.sum(axis=1),1,atol=1e-5)
    prediction=prob.argmax(axis=1).astype(np.int8)
    keep=['subject','trial','window_id','source_start','source_stop_exclusive','purity','mixed_labels']
    result=meta[keep].copy();result['y_true']=ty;result['y_pred']=prediction;result['fold']=fold
    for i in LABELS:result[f'p_{i}']=prob[:,i]
    result.to_parquet(dest/'test_predictions.parquet',index=False)
    table=per_class(ty,prediction);table.to_csv(dest/'per_class_test.csv',index=False)
    np.save(dest/'confusion_test.npy',confusion_matrix(ty,prediction,labels=LABELS))
    scores=metrics(ty,prediction)
    named=ty>0
    scores['macro_f1_named12_true_named_only']=float(per_class(ty[named],prediction[named]).f1.iloc[1:].mean())
    scores.update({'fold':fold,'best_iteration':int(model.best_iteration_),
                   'best_validation_macro_f1_13':float(model.best_score_['inner_validation']['macro_f1_13']),
                   'training_seconds':elapsed_fit,'total_seconds':time.monotonic()-start,
                   'training_subjects':train,'validation_subjects':val,'test_subjects':test})
    save_json(done,scores)
    print('FOLD_COMPLETE '+json.dumps(scores),flush=True)


def aggregate(root,out,config):
    pieces=[pd.read_parquet(out/f'fold_{i}'/'test_predictions.parquet') for i in range(6)]
    oof=pd.concat(pieces,ignore_index=True)
    assert not oof.duplicated(['subject','window_id']).any()
    assert set(oof.subject)==set(range(1,31))
    assert len(oof)==json.loads((root/'summary.json').read_text())['windows_2s']
    oof=oof.sort_values(['subject','window_id']).reset_index(drop=True)
    oof.to_parquet(out/'oof_predictions.parquet',index=False)
    y=oof.y_true.to_numpy();pred=oof.y_pred.to_numpy()
    cm=confusion_matrix(y,pred,labels=LABELS)
    np.save(out/'confusion_oof.npy',cm)
    pd.DataFrame(cm,index=NAMES,columns=NAMES).to_csv(out/'confusion_oof.csv',index_label='true_activity')
    per_class(y,pred).to_csv(out/'per_class_oof.csv',index=False)
    folds=[json.loads((out/f'fold_{i}'/'complete.json').read_text()) for i in range(6)]
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,list)} for row in folds]).to_csv(out/'fold_metrics.csv',index=False)
    subject_rows=[];subject_cms=[]
    for subject,g in oof.groupby('subject',sort=True):
        m=metrics(g.y_true,g.y_pred);m['subject']=int(subject);m['fold']=int(g.fold.iloc[0])
        subject_rows.append(m);subject_cms.append(confusion_matrix(g.y_true,g.y_pred,labels=LABELS))
    pd.DataFrame(subject_rows).to_csv(out/'per_subject_metrics.csv',index=False)
    group_rows=[]
    for name,mask in [('all',np.ones(len(oof),dtype=bool)),('pure',oof.purity.to_numpy()==1),
                      ('stable_purity_ge_0_9',oof.purity.to_numpy()>=0.9),
                      ('transition_purity_lt_0_9',oof.purity.to_numpy()<0.9),
                      ('mixed_labels',oof.mixed_labels.to_numpy()),('others_true',y==0),('named_true',y>0)]:
        if not mask.any():continue
        row=metrics(y[mask],pred[mask]);row['group']=name
        group_rows.append(row)
        table=per_class(y[mask],pred[mask]);table.insert(0,'group',name)
        table.to_csv(out/f'per_class_{name}.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'window_group_metrics.csv',index=False)
    # Subject-cluster bootstrap, conditional on trained models, not a CI over
    # retraining or completely unseen future populations.
    rng=np.random.default_rng(42);subject_cms=np.asarray(subject_cms)
    bootstrap=[]
    for _ in range(2000):
        draw=rng.integers(0,30,size=30)
        bootstrap.append(metrics_from_cm(subject_cms[draw].sum(axis=0))['macro_f1_13'])
    summary={'model':'LightGBM','data_version':'independent_v1','window_seconds':2,'features':680,
             'subjects':30,'oof_windows':len(oof),'oof':metrics_from_cm(cm),
             'macro_f1_named12_true_named_only':float(per_class(y[y>0],pred[y>0]).f1.iloc[1:].mean()),
             'fold_mean':{},'fold_sd':{},'per_subject_macro_f1_13_mean':float(np.mean([s['macro_f1_13'] for s in subject_rows])),
             'subject_cluster_bootstrap_macro_f1_13_ci95':np.quantile(bootstrap,[0.025,0.975]).tolist(),
             'bootstrap_scope':'Conditional on the six trained models; does not include model retraining uncertainty.',
             'window_label_policy':'majority labels; all 13 classes and all purity levels retained',
             'primary_metric':'13-class macro F1; fixed labels 0..12, all held-out windows',
             'named12_all_windows_definition':'F1 averaged over labels 1..12, including false positives from true Others',
             'named12_true_named_definition':'Diagnostic computed only where y_true>0; for contextual teammate comparison',
             'early_stopping':'validation participants only; macro F1 over all 13 classes',
             'test_evaluation':'Once per outer fold after selection; every subject predicted exactly once',
             'total_training_seconds':sum(f['training_seconds'] for f in folds),
             'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for key in ['accuracy_13','macro_f1_13','macro_f1_named12_all_windows','walk3_macro_f1_all_windows','macro_f1_named12_true_named_only']:
        values=[f[key] for f in folds]
        summary['fold_mean'][key]=float(np.mean(values));summary['fold_sd'][key]=float(np.std(values,ddof=1))
    save_json(out/'summary.json',summary)
    print('ALL_SIX_FOLDS_COMPLETE '+json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    splits=json.loads((args.root/'subject_splits_6fold.json').read_text())['folds']
    config={'model':{'objective':'multiclass','num_class':13,'metric':'None',
                     'n_estimators':300,'learning_rate':0.08,'num_leaves':31,
                     'max_bin':63,'min_child_samples':50,'n_jobs':12,'verbosity':-1,
                     'random_state':42,'deterministic':True,'force_col_wise':True},
            'early_stopping_rounds':40,'data_root':str(args.root),'features':680,
            'split_file_sha256':hashlib.sha256((args.root/'subject_splits_6fold.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'feature_schema_sha256':hashlib.sha256((args.root/'baseline_feature_schema.json').read_bytes()).hexdigest(),
            'class_names':NAMES,'normalization':'none; LightGBM native missing handling',
            'smoothing':'none','all_window_purity_levels_retained':True}
    if args.smoke:
        start=time.monotonic()
        x,y,_=load_split(args.root,splits[0]['train'],cap_per_class=500)
        vx,vy,_=load_split(args.root,splits[0]['validation'],cap_per_class=100)
        params=config['model'].copy();params['n_estimators']=8;params['num_leaves']=15
        model=lgb.LGBMClassifier(**params)
        model.fit(x,y,eval_set=[(vx,vy)],eval_metric=eval_macro,callbacks=[lgb.log_evaluation(0)])
        pred=model.predict(vx);result=metrics(vy,pred)
        result.update({'smoke_only':True,'test_subjects_used':False,'training_samples':len(y),
                       'validation_samples':len(vy),'seconds':time.monotonic()-start})
        save_json(args.out/'smoke_result.json',result)
        print('SMOKE_PASSED '+json.dumps(result),flush=True);return
    config_path=args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Existing run config differs; use a new output directory')
    save_json(config_path,config)
    print('CONFIG '+json.dumps(config),flush=True)
    for split in splits:formal_fold(args.root,args.out,split,config)
    aggregate(args.root,args.out,config)


if __name__=='__main__':main()
