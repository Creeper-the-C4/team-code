"""Memory-mapped access for Task 1/2 windows and Task 3 full trials.

Apply the sensor mask BEFORE computing normalization, relative orientation,
gait, or any other feature. No hidden sensor is used by these accessors.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ['LowerBack','RightThigh','RightShank','RightFoot','LeftThigh',
           'LeftShank','LeftFoot','UpperBack','Head','RightShoulder',
           'RightUpperArm','RightForeArm','RightWrist','LeftShoulder',
           'LeftUpperArm','LeftForeArm','LeftWrist']


def selected_indices(sensor_names):
    if sensor_names is None:
        raise ValueError('Pass explicit available sensor names (use SENSORS for full layout)')
    if not sensor_names or len(set(sensor_names)) != len(sensor_names):
        raise ValueError('Require a non-empty list of distinct sensor names')
    return [SENSORS.index(name) for name in sensor_names]


def prepare_quaternions(values):
    """Normalize and make signs continuous within this single trial/window.

    Source cache is unchanged. Acc/Gyr values and coordinates are unchanged.
    """
    result=np.array(values,dtype=np.float32,copy=True)
    q=result[:,:,6:10]
    norms=np.linalg.norm(q,axis=2,keepdims=True)
    if not np.isfinite(q).all() or np.any(norms<1e-6):
        raise ValueError('Invalid orientation; consult QC flags, do not silently impute')
    q/=norms
    if len(q)>1:
        dot=np.sum(q[1:]*q[:-1],axis=2)
        multiplier=np.cumprod(np.where(dot<0,-1,1),axis=0)
        q[1:]*=multiplier[:,:,None]
    return result


class SubjectData:
    def __init__(self,root,subject):
        self.path=Path(root)/'subjects'/f'P{subject:02d}'
        if not (self.path/'complete.json').exists():
            raise ValueError('Subject processing has not completed')
        self.signals=np.load(self.path/'signals.npy',mmap_mode='r')
        self.labels=np.load(self.path/'labels.npy',mmap_mode='r')
        self.trials=pd.read_parquet(self.path/'trials.parquet')

    def read_trial(self,trial_id,sensor_names,prepare_q=True):
        row=self.trials.loc[self.trials.trial==trial_id].iloc[0]
        return self._read(int(row.source_start),int(row.source_stop_exclusive),sensor_names,prepare_q)

    def read_window(self,row,sensor_names,prepare_q=True):
        trial=self.trials.loc[self.trials.trial==int(row['trial'])].iloc[0]
        start,stop=int(row['source_start']),int(row['source_stop_exclusive'])
        if start<int(trial.source_start) or stop>int(trial.source_stop_exclusive) or stop<=start:
            raise ValueError('Window outside official trial')
        return self._read(start,stop,sensor_names,prepare_q)

    def _read(self,start,stop,sensor_names,prepare_q):
        # Basic time slice, then select available sensors; no full-layout features.
        values=np.asarray(self.signals[start:stop])[:,selected_indices(sensor_names),:].copy()
        if prepare_q:
            values=prepare_quaternions(values)
        return values,np.asarray(self.labels[start:stop]).copy()


def fit_acc_gyr_scaler(root,training_subjects,sensor_names,held_out_subjects):
    """Fit only selected sensors and explicit TRAINING subjects. No quat Z-score.

    held_out_subjects must include BOTH validation and test subjects. Call anew
    for each outer/inner training fit; never load quality-table means here.
    """
    training_subjects=list(training_subjects)
    if not training_subjects or len(set(training_subjects))!=len(training_subjects):
        raise ValueError('Distinct, non-empty training subject list required')
    if set(training_subjects)&set(held_out_subjects):
        raise ValueError('Training scaler would include held-out participants')
    indices=selected_indices(sensor_names)
    count=np.zeros((len(indices),6),dtype=np.int64)
    total=np.zeros_like(count,dtype=np.float64)
    squares=np.zeros_like(total)
    for subject in training_subjects:
        data=SubjectData(root,subject)
        for start in range(0,len(data.labels),25000):
            values=np.asarray(data.signals[start:start+25000])[:,indices,:6].astype(np.float64)
            valid=np.isfinite(values)
            clean=np.where(valid,values,0)
            count+=valid.sum(axis=0)
            total+=clean.sum(axis=0)
            squares+=(clean*clean).sum(axis=0)
    if np.any(count==0):
        raise ValueError('Selected channel has no finite training observations')
    mean=total/count
    std=np.sqrt(np.maximum(squares/count-mean**2,0))
    std=np.where(std<1e-8,1,std)
    return {'mean':mean,'std':std,'training_subjects':training_subjects,
            'sensor_names':list(sensor_names),'source':'continuous training frames only'}


def apply_acc_gyr_scaler(values,scaler,sensor_names):
    if list(sensor_names)!=scaler['sensor_names']:
        raise ValueError('Scaler sensor names/order do not match input')
    result=np.array(values,dtype=np.float32,copy=True)
    result[:,:,:6]=(result[:,:,:6]-scaler['mean'])/scaler['std']
    return result


FEATURE_SOURCES={
    'lowerback_step_frequency':['LowerBack'],
    'lowerback_step_confidence':['LowerBack'],
    'lowerback_acc_y_rms':['LowerBack'],
    'bilateral_foot_horizontal_jerk':['LeftFoot','RightFoot'],
    'bilateral_shank_gyr_magnitude':['LeftShank','RightShank'],
    'thigh_relative_to_lowerback_rom':['LeftThigh','RightThigh','LowerBack'],
    'lowerback_vertical_horizontal_variance_ratio':['LowerBack'],
    'step_frequency_times_thigh_rom':['LowerBack','LeftThigh','RightThigh']}


def available_features(sensor_names):
    allowed=set(sensor_names)
    return {name:deps for name,deps in FEATURE_SOURCES.items() if set(deps)<=allowed}
