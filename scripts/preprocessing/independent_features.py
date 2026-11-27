"""Fixed, sensor-local 2-second baseline features. No training or fitted transform.

This produces [window,17,40] float32 so Task 2 can remove WHOLE sensor blocks.
No cross-sensor feature, label-conditioned gating, or normalization is applied.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(k,'4')
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

VECTOR_CHANNELS=['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z']
FEATURES=[f'{ch}_{stat}' for ch in VECTOR_CHANNELS for stat in ['mean','std','rms','range']]
FEATURES += [f'{kind}_magnitude_{stat}' for kind in ['acc','gyr'] for stat in ['mean','std','rms']]
FEATURES += ['acc_y_dominant_frequency_hz','acc_y_peak_fraction_0_5_to_4_hz']
FEATURES += [f'orientation_center_R_{i}{j}' for j in [0,1] for i in [0,1,2]]
FEATURES += ['orientation_increment_mean_rad','orientation_increment_rms_rad']
assert len(FEATURES)==40


def sensor_features(values):
    # values: [B,T,S,10]. All operations preserve S, so masks cannot leak.
    vector=values[:,:,:,:6].astype(np.float64)
    mean=vector.mean(axis=1)
    std=vector.std(axis=1)
    rms=np.sqrt(np.mean(vector*vector,axis=1))
    span=np.ptp(vector,axis=1)
    pieces=[np.stack([mean,std,rms,span],axis=-1).reshape(len(values),values.shape[2],24)]
    for sl in [slice(0,3),slice(3,6)]:
        mag=np.linalg.norm(vector[:,:,:,sl],axis=-1)
        pieces.append(np.stack([mag.mean(axis=1),mag.std(axis=1),np.sqrt(np.mean(mag*mag,axis=1))],axis=-1))
    ay=vector[:,:,:,1]
    centered=ay-ay.mean(axis=1,keepdims=True)
    n=values.shape[1]
    taper=np.hanning(n).reshape(1,n,1)
    power=np.abs(np.fft.rfft(centered*taper,axis=1))**2
    freq=np.fft.rfftfreq(n,1/60)
    band=(freq>=0.5)&(freq<=4)
    selected=power[:,band,:]
    peak=selected.argmax(axis=1)
    best=np.take_along_axis(selected,peak[:,None,:],axis=1)[:,0,:]
    total=selected.sum(axis=1)
    dominant=freq[band][peak]
    active=np.var(ay,axis=1)>1e-10
    dominant=np.where(active,dominant,np.nan)
    ratio=np.divide(best,total,out=np.zeros_like(best),where=total>1e-15)
    ratio=np.where(active,ratio,0)
    pieces.append(np.stack([dominant,ratio],axis=-1))
    q=values[:,:,:,6:10].astype(np.float64)
    norms=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norms<1e-6):
        raise ValueError('Invalid orientation; inspect QC before feature extraction')
    q=q/norms
    x,y,z,w=np.moveaxis(q[:,n//2],-1,0)
    # First two columns of R(q); sign-invariant. This is not calibrated body pose.
    rot=np.stack([1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),
                  2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],axis=-1)
    pieces.append(rot)
    a,b=q[:,:-1],q[:,1:]
    sign=np.where(np.sum(a*b,axis=-1)<0,-1,1)
    chord=np.linalg.norm(b*sign[:,:,:,None]-a,axis=-1)
    angle=4*np.arcsin(np.clip(chord/2,0,1))
    pieces.append(np.stack([angle.mean(axis=1),np.sqrt(np.mean(angle*angle,axis=1))],axis=-1))
    return np.concatenate(pieces,axis=-1).astype(np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads((args.root/'schema.json').read_text())
    spec={'feature_names':FEATURES,'shape':'[window,sensor,40]','dtype':'float32',
          'window_seconds':2,'label_policy':'all source labels/windows retained in companion index',
          'fit_parameters':'none','source_sensors':meta['sensors'],
          'provenance':'Each sensor block depends ONLY on the corresponding physical sensor.',
          'unit_policy':'Original acceleration units preserved; no gravity compensation or global re-rotation.',
          'frequency_note':'A spectral peak within 0.5-4 Hz is not automatically a step frequency; 2 s gives 0.5 Hz bin spacing.',
          'orientation_note':'Normalized source quaternion, sign-invariant R(q) encoding; no assumed sensor-to-bone calibration.',
          'nan_policy':'near-constant acc_y has undefined dominant frequency (NaN); no shared imputation.'}
    (args.root/'baseline_feature_schema.json').write_text(json.dumps(spec,indent=2))
    rows=[]
    for sid in range(1,31):
        dest=args.root/'subjects'/f'P{sid:02d}'
        deadline=time.monotonic()+300
        if not (dest/'complete.json').exists():
            print(f'Waiting for P{sid:02d} preprocessing',flush=True)
        while not (dest/'complete.json').exists():
            if time.monotonic()>deadline: raise ValueError(f'P{sid:02d} incomplete after 300 seconds')
            time.sleep(5)
        done=dest/'features_2s_complete.json'
        if done.exists(): rows.append(json.loads(done.read_text()));continue
        windows=pd.read_parquet(dest/'windows_2s.parquet')
        x=np.load(dest/'signals.npy',mmap_mode='r')
        partial=dest/'baseline_features_2s.npy.partial'
        feat=open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(windows),17,40))
        starts=windows.source_start.to_numpy()
        for b in range(0,len(windows),128):
            pos=starts[b:b+128,None]+np.arange(120)[None,:]
            feat[b:b+128]=sensor_features(x[pos])
        nan=int(np.isnan(feat).sum())
        inf=int(np.isinf(feat).sum())
        feat.flush();del feat,x
        partial.replace(dest/'baseline_features_2s.npy')
        rec={'subject':sid,'windows':len(windows),'features_per_sensor':40,'sensors':17,
             'nan_values':nan,'inf_values':inf,'bytes':(dest/'baseline_features_2s.npy').stat().st_size}
        done.write_text(json.dumps(rec,indent=2));rows.append(rec)
        print(json.dumps(rec),flush=True)
    summary={'subjects':len(rows),'windows':sum(r['windows'] for r in rows),
             'bytes':sum(r['bytes'] for r in rows),'nan_values':sum(r['nan_values'] for r in rows),
             'inf_values':sum(r['inf_values'] for r in rows),'training_started':False}
    (args.root/'baseline_feature_summary.json').write_text(json.dumps(summary,indent=2))
    print('FEATURE_EXTRACTION_COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
