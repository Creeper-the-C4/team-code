"""Independent C-problem data preparation. No model fitting or raw-data mutation.

Signal cache preserves original values as float32 [frame, sensor, channel].
Trial boundaries are the authors' zero-based INCLUSIVE endpoints; internal
window bounds are zero-based HALF-OPEN. All transforms with fitted parameters
belong in the fold-specific training pipeline, not in this shared cache.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_key, '4')
import argparse
import collections
import datetime
import hashlib
import json
import shutil
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.spatial.transform import Rotation

SENSORS = ['LowerBack','RightThigh','RightShank','RightFoot','LeftThigh',
           'LeftShank','LeftFoot','UpperBack','Head','RightShoulder',
           'RightUpperArm','RightForeArm','RightWrist','LeftShoulder',
           'LeftUpperArm','LeftForeArm','LeftWrist']
CHANNELS = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z',
            'Quat_X','Quat_Y','Quat_Z','Quat_W']
SIGNAL_COLUMNS = [f'{c}_{s}' for s in SENSORS for c in CHANNELS]
FS = 60


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    sha = hashlib.sha256()
    crc = 0
    lines = 0
    last = b''
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            lines += block.count(b'\n')
            last = block[-1:]
    return {'bytes':path.stat().st_size,'sha256':sha.hexdigest(),
            'crc32':f'{crc & 0xffffffff:08x}',
            'physical_lines':lines + int(bool(last) and last != b'\n')}


def validate_trials(df, nrows):
    bounds = df[['Start Index','End Index']].to_numpy(dtype=np.int64)
    if len(bounds) == 0:
        raise ValueError('No official trial boundaries')
    if np.any(bounds[:,0] < 0) or np.any(bounds[:,1] < bounds[:,0]) or np.any(bounds[:,1] >= nrows):
        raise ValueError('Invalid trial boundaries or wrong index convention')
    if np.any(bounds[1:,0] <= bounds[:-1,1]):
        raise ValueError('Unsorted or overlapping official trials')
    covered = int(np.sum(bounds[:,1]-bounds[:,0]+1))
    return bounds, {'trial_count':len(bounds),'covered_rows':covered,
                    'uncovered_rows':nrows-covered,'first_start':int(bounds[0,0]),
                    'last_end_inclusive':int(bounds[-1,1]),
                    'gaps_between_trials':int(np.sum(bounds[1:,0] != bounds[:-1,1]+1))}


def signal_chunk(df):
    if list(df.columns) != ['Activity'] + SIGNAL_COLUMNS:
        raise ValueError('CSV column names/order differ from fixed sensor schema')
    y = df['Activity'].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y) & (y == np.floor(y)) & (y >= 0) & (y <= 12)):
        raise ValueError('Missing, non-integer or out-of-range source labels')
    return df[SIGNAL_COLUMNS].to_numpy(dtype=np.float32).reshape(-1,17,10), y.astype(np.int8)


def create_windows(y, invalid, bounds, subject):
    frames = []
    for width in [120,240]:
        parts = []
        for tid, (start,end) in enumerate(bounds):
            starts = np.arange(start, end+2-width, width//2, dtype=np.int64)
            if len(starts) == 0:
                continue
            # Sparse counting per trial avoids materializing signal windows.
            local = y[start:end+1]
            a = starts-start
            counts = np.empty((len(starts),13),dtype=np.int32)
            for label in range(13):
                cum=np.r_[0,np.cumsum(local==label,dtype=np.int64)]
                counts[:,label]=cum[a+width]-cum[a]
            major=counts.argmax(axis=1).astype(np.int8)  # deterministic ties: smaller label ID
            bad=np.r_[0,np.cumsum(invalid[start:end+1],dtype=np.int64)]
            bad_counts=bad[a+width]-bad[a]
            transitions=np.r_[False,local[1:]!=local[:-1]]
            ct=np.r_[0,np.cumsum(transitions,dtype=np.int64)]
            nchanges=ct[a+width]-ct[a+1]
            data={'subject':subject,'trial':tid,'source_start':starts,
                  'source_stop_exclusive':starts+width,'width_frames':width,
                  'stride_frames':width//2,'majority_label':major,
                  'center_label':local[a+width//2],
                  'purity':counts.max(axis=1)/width,
                  'mixed_labels':(counts>0).sum(axis=1)>1,
                  'label_changes':nchanges,'invalid_signal_frames':bad_counts,
                  'tie_for_majority':(counts==counts.max(axis=1)[:,None]).sum(axis=1)>1}
            data.update({f'label_count_{i}':counts[:,i] for i in range(13)})
            parts.append(pd.DataFrame(data))
        win=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        if len(win):
            b=bounds[win.trial.to_numpy()]
            assert np.all(win.source_start.to_numpy()>=b[:,0])
            assert np.all(win.source_stop_exclusive.to_numpy()<=b[:,1]+1)
            assert np.all(win[[f'label_count_{i}' for i in range(13)]].sum(axis=1)==width)
        frames.append((width,win))
    return frames


def rotation_diagnostics(x,y,bounds):
    # Sample every second, only pairs inside a trial; use BOTH adjacent frames.
    # Fits diagnose conventions, not independent pose ground truth.
    ids=[]
    flips=np.zeros(17,dtype=np.int64)
    pairs=np.zeros(17,dtype=np.int64)
    for start,end in bounds:
        if end<=start: continue
        q=x[start:end+1,:,6:10]
        dot=np.sum(q[1:]*q[:-1],axis=2)
        valid=np.isfinite(dot) & (np.linalg.norm(q[1:],axis=2)>1e-6) & (np.linalg.norm(q[:-1],axis=2)>1e-6)
        flips+=np.sum((dot<0)&valid,axis=0)
        pairs+=valid.sum(axis=0)
        ids.extend(np.arange(start,end,60).tolist())
    ids=np.asarray(ids,dtype=np.int64)
    out=[]
    for j,sensor in enumerate(SENSORS):
        q0=np.asarray(x[ids,j,6:10],dtype=np.float64)
        q1=np.asarray(x[ids+1,j,6:10],dtype=np.float64)
        observed=(np.asarray(x[ids,j,3:6],dtype=np.float64)+np.asarray(x[ids+1,j,3:6],dtype=np.float64))/2
        ok=np.isfinite(q0).all(axis=1)&np.isfinite(q1).all(axis=1)&np.isfinite(observed).all(axis=1)&(np.linalg.norm(q0,axis=1)>1e-6)&(np.linalg.norm(q1,axis=1)>1e-6)
        q0,q1,observed=q0[ok],q1[ok],observed[ok]
        if not len(q0):
            out.append({'sensor':sensor,'error':'no valid quaternion pairs'});continue
        r0,r1=Rotation.from_quat(q0),Rotation.from_quat(q1)
        predictions={'world':(r1*r0.inv()).as_rotvec()*FS,
                     'body':(r0.inv()*r1).as_rotvec()*FS}
        results={}
        for mode,predicted in predictions.items():
            active=(np.linalg.norm(predicted,axis=1)>0.2)&(np.linalg.norm(predicted,axis=1)<20)
            pred,obs=predicted[active],observed[active]
            denom=float(np.sum(pred*pred))
            if denom<=0: continue
            scale=float(np.sum(pred*obs)/denom)
            sst=float(np.sum((obs-obs.mean(axis=0))**2))
            error=float(np.sum((obs-scale*pred)**2))
            results[mode]={'sample_pairs':int(active.sum()),'observed_over_quat_rate_scale':scale,
                           'r2_scaled':1-error/sst if sst>0 else None}
        standing=np.flatnonzero(np.asarray(y[ids])==3)
        acc=np.linalg.norm(x[ids[standing],j,:3],axis=1)
        acc=acc[np.isfinite(acc)]
        out.append({'sensor':sensor,'quat_sign_flip_pairs':int(flips[j]),
                    'valid_adjacent_pairs':int(pairs[j]),'gyro_convention_fits':results,
                    'sample_standing_acc_norm_median_raw_units':float(np.median(acc)) if len(acc) else None})
    return out


def process_subject(subject, args, trial_table, zip_entries):
    sid=f'P{subject:02d}'
    dest=args.output/'subjects'/sid
    marker=dest/'complete.json'
    if marker.exists():
        print(f'{sid} already complete; skipping',flush=True)
        return json.loads(marker.read_text())
    dest.mkdir(parents=True,exist_ok=True)
    source=args.source/f'{sid}.csv'
    integrity=digest(source)
    expected=zip_entries.get('dataset/'+source.name)
    if expected is None or expected.file_size!=integrity['bytes'] or f'{expected.CRC:08x}'!=integrity['crc32']:
        raise ValueError(f'{sid}: extracted CSV differs from ZIP entry size/CRC')
    nrows=integrity['physical_lines']-1
    trials=trial_table[trial_table['Participant ID']==subject]
    bounds,boundary_report=validate_trials(trials,nrows)
    estimate=nrows*(170*4+1)
    if shutil.disk_usage(dest).free < estimate+3*1024**3:
        raise RuntimeError(f'{sid}: insufficient disk space; keep 3 GiB reserve')
    xp=dest/'signals.npy.partial'
    yp=dest/'labels.npy.partial'
    x=open_memmap(xp,mode='w+',dtype=np.float32,shape=(nrows,17,10))
    y=open_memmap(yp,mode='w+',dtype=np.int8,shape=(nrows,))
    invalid=np.zeros(nrows,dtype=np.bool_)
    missing=np.zeros((17,10),dtype=np.int64)
    infinity=np.zeros((17,10),dtype=np.int64)
    finite_count=np.zeros((17,10),dtype=np.int64)
    sums=np.zeros((17,10),dtype=np.float64)
    sums2=np.zeros((17,10),dtype=np.float64)
    minimum=np.full((17,10),np.inf)
    maximum=np.full((17,10),-np.inf)
    hist=np.zeros(13,dtype=np.int64)
    qbad=np.zeros(17,dtype=np.int64)
    qnonunit=np.zeros(17,dtype=np.int64)
    qnorm_min=np.full(17,np.inf)
    qnorm_max=np.full(17,-np.inf)
    pos=0
    for df in pd.read_csv(source,chunksize=25000,encoding='utf-8-sig'):
        values,labels=signal_chunk(df)
        stop=pos+len(values)
        x[pos:stop]=values
        y[pos:stop]=labels
        ok=np.isfinite(values)
        invalid[pos:stop]=~ok.all(axis=(1,2))
        missing+=np.isnan(values).sum(axis=0)
        infinity+=np.isinf(values).sum(axis=0)
        finite_count+=ok.sum(axis=0)
        safe=np.where(ok,values,0).astype(np.float64)
        sums+=safe.sum(axis=0)
        sums2+=(safe*safe).sum(axis=0)
        minimum=np.minimum(minimum,np.min(np.where(ok,values,np.inf),axis=0))
        maximum=np.maximum(maximum,np.max(np.where(ok,values,-np.inf),axis=0))
        qnorm=np.linalg.norm(values[:,:,6:10].astype(np.float64),axis=2)
        validq=np.isfinite(qnorm)&(qnorm>1e-6)
        qbad+=(~validq).sum(axis=0)
        qnonunit+=(validq&(np.abs(qnorm-1)>1e-3)).sum(axis=0)
        qnorm_min=np.minimum(qnorm_min,np.min(np.where(validq,qnorm,np.inf),axis=0))
        qnorm_max=np.maximum(qnorm_max,np.max(np.where(validq,qnorm,-np.inf),axis=0))
        hist+=np.bincount(labels,minlength=13)
        pos=stop
    if pos!=nrows: raise ValueError(f'{sid}: row-count mismatch')
    x.flush(); y.flush()
    windows_summary={}
    for width,win in create_windows(y,invalid,bounds,subject):
        win.to_parquet(dest/f'windows_{width//60}s.parquet',index=False)
        windows_summary[str(width//60)]={'count':len(win),'mixed_label_windows':int(win.mixed_labels.sum()) if len(win) else 0,
            'purity_below_0_9':int((win.purity<0.9).sum()) if len(win) else 0,
            'invalid_signal_windows':int((win.invalid_signal_frames>0).sum()) if len(win) else 0,
            'label_counts':{str(int(k)):int(v) for k,v in win.majority_label.value_counts().items()} if len(win) else {}}
    rows=[]
    for tid,(start,end) in enumerate(bounds):
        h=np.bincount(y[start:end+1],minlength=13)
        row={'subject':subject,'trial':tid,'source_start':int(start),
             'source_end_inclusive':int(end),'source_stop_exclusive':int(end)+1,
             'frames':int(end-start+1),'duration_s':float((end-start+1)/FS),
             'majority_label':int(h.argmax()),'purity':float(h.max()/h.sum())}
        row.update({f'label_count_{i}':int(h[i]) for i in range(13)})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(dest/'trials.parquet',index=False)
    means=np.divide(sums,finite_count,out=np.zeros_like(sums),where=finite_count>0)
    variances=np.maximum(np.divide(sums2,finite_count,out=np.zeros_like(sums2),where=finite_count>0)-means**2,0)
    channels=[]
    for j,sensor in enumerate(SENSORS):
        for k,channel in enumerate(CHANNELS):
            channels.append({'sensor':sensor,'channel':channel,'finite_count':int(finite_count[j,k]),
                             'nan_count':int(missing[j,k]),'inf_count':int(infinity[j,k]),
                             'mean':float(means[j,k]),'std':float(np.sqrt(variances[j,k])),
                             'min':float(minimum[j,k]) if np.isfinite(minimum[j,k]) else None,
                             'max':float(maximum[j,k]) if np.isfinite(maximum[j,k]) else None})
    pd.DataFrame(channels).to_csv(dest/'channel_quality.csv',index=False)
    rotation=rotation_diagnostics(x,y,bounds)
    report={'subject':subject,'rows':nrows,'shape':[nrows,17,10],
            'source_integrity':integrity,'extracted_crc_matches_zip':True,
            'boundaries':boundary_report,'labels':{str(i):int(n) for i,n in enumerate(hist)},
            'signal_nan_count':int(missing.sum()),'signal_inf_count':int(infinity.sum()),
            'quaternion_invalid_count':qbad.tolist(),
            'quaternion_norm_deviation_over_1e_3':qnonunit.tolist(),
            'quaternion_norm_min':[float(v) if np.isfinite(v) else None for v in qnorm_min],
            'quaternion_norm_max':[float(v) if np.isfinite(v) else None for v in qnorm_max],
            'windows':windows_summary,'rotation_diagnostics':rotation}
    dump(dest/'quality.json',report)
    del x,y
    xp.replace(dest/'signals.npy'); yp.replace(dest/'labels.npy')
    result={'subject':subject,'rows':nrows,'trials':len(bounds),
            'windows_2s':windows_summary['2']['count'],'windows_4s':windows_summary['4']['count'],
            'nan':int(missing.sum()),'inf':int(infinity.sum()),'uncovered_rows':boundary_report['uncovered_rows'],
            'cache_bytes':estimate,'source_sha256':integrity['sha256']}
    dump(marker,result)
    print(json.dumps(result),flush=True)
    return result


def publish_summary(args):
    completed=[json.loads(p.read_text()) for p in sorted((args.output/'subjects').glob('P*/complete.json'))]
    table=pd.DataFrame(completed)
    table.to_csv(args.output/'subject_summary.csv',index=False)
    report={'completed_subjects':len(completed),'expected_subjects':30,
            'total_rows':sum(r['rows'] for r in completed),
            'total_trials':sum(r['trials'] for r in completed),
            'windows_2s':sum(r['windows_2s'] for r in completed),
            'windows_4s':sum(r['windows_4s'] for r in completed),
            'nan_count':sum(r['nan'] for r in completed),'inf_count':sum(r['inf'] for r in completed),
            'uncovered_rows':sum(r['uncovered_rows'] for r in completed),
            'free_disk_bytes':shutil.disk_usage(args.output).free,
            'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_started':False}
    dump(args.output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/dataset'))
    parser.add_argument('--archive',type=Path,default=Path('/root/autodl-tmp/dataset.zip'))
    parser.add_argument('--trials',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('/root/autodl-tmp/C_Problem/data/processed/independent_v1'))
    parser.add_argument('--subjects',nargs='+',type=int,default=list(range(1,31)))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    trial_table=pd.read_csv(args.trials)
    expected={'Participant ID','Start Index','End Index'}
    if set(trial_table.columns)!=expected: raise ValueError('Unexpected trial metadata schema')
    if trial_table.isna().any().any(): raise ValueError('Missing trial metadata')
    for col in expected:
        if not np.all(trial_table[col]==np.floor(trial_table[col])): raise ValueError('Non-integer boundary')
    dump(args.output/'schema.json',{
        'version':'independent_v1','sample_rate_hz':FS,'sensors':SENSORS,'channels':CHANNELS,
        'signals_shape':'[frame,17,10]','signals_dtype':'float32','labels_dtype':'int8',
        'units':{'acceleration':'source unchanged; publication declares m/s^2; disputed g interpretation not applied',
                 'angular_velocity':'publication declares rad/s','quaternion':'xyzw, source unchanged'},
        'public_source':'https://raw.githubusercontent.com/FudanBSRL/Comprehensive-IMU-Dataset/main/trial_information.csv',
        'trials_sha256':digest(args.trials)['sha256'],
        'window_bounds':'zero-based half-open, within official zero-based inclusive trial bounds',
        'windows_seconds':[2,4],'overlap':0.5,
        'normalization':'none; fit transforms per training fold only',
        'quaternion_processing':'raw retained; normalize and sign-align within trial in loader',
        'missing_data_policy':'retain and flag; no imputation in shared cache',
        'majority_tie_rule':'lowest activity ID; tie flag retained',
        'quality_channel_means':'descriptive QC only; NEVER training normalization statistics'} )
    splits=[]
    for fold in range(6):
        test=[s for s in range(1,31) if (s-1)%6==fold]
        val=[s for s in range(1,31) if (s-1)%6==(fold+1)%6]
        train=[s for s in range(1,31) if s not in test+val]
        assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
        splits.append({'fold':fold,'train':train,'validation':val,'test':test})
    dump(args.output/'subject_splits_6fold.json',{'purpose':'compatibility with teammate; fixed inner validation, not full nested CV',
         'selection':'training/validation only; test subjects for reporting only','folds':splits})
    with zipfile.ZipFile(args.archive) as z:
        zip_entries={e.filename:e for e in z.infolist()}
        for subject in args.subjects:
            process_subject(subject,args,trial_table,zip_entries)
            publish_summary(args)
    shutil.copy2(args.source/'anthropometric information.csv',args.output/'anthropometric_original.csv')
    print('PREPROCESS_COMPLETE '+json.dumps(publish_summary(args)),flush=True)


if __name__=='__main__':
    main()
