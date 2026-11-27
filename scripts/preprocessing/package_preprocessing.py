from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd

project=Path('/root/autodl-tmp/C_Problem')
root=project/'data/processed/independent_v1'
feat_summary=json.loads((root/'baseline_feature_summary.json').read_text())
assert feat_summary['subjects']==30 and feat_summary['windows']==220702
allowed_nan_feature_index=30
undefined=0
for subject in range(1,31):
    d=root/'subjects'/f'P{subject:02d}'
    q=json.loads((d/'quality.json').read_text())
    x=np.load(d/'signals.npy',mmap_mode='r')
    y=np.load(d/'labels.npy',mmap_mode='r')
    f=np.load(d/'baseline_features_2s.npy',mmap_mode='r')
    w=pd.read_parquet(d/'windows_2s.parquet')
    assert x.shape==(q['rows'],17,10) and x.dtype==np.float32
    assert y.shape==(q['rows'],) and y.dtype==np.int8
    assert f.shape==(len(w),17,40) and f.dtype==np.float32
    for start in range(0,len(f),2048):
        a=np.asarray(f[start:start+2048])
        assert not np.isinf(a).any()
        bad=np.isnan(a)
        undefined+=int(bad[:,:,allowed_nan_feature_index].sum())
        bad[:,:,allowed_nan_feature_index]=False
        assert not bad.any(),'Unexpected undefined feature outside dominant frequency'
assert undefined==feat_summary['nan_values']
validation={'all_30_subject_shapes_valid':True,'feature_rows_align_with_window_index':True,
            'feature_infinities':0,'undefined_dominant_frequency_values':undefined,
            'unexpected_feature_nan':0,'models_trained':False}
(root/'final_storage_validation.json').write_text(json.dumps(validation,indent=2))
readme='''C Problem independent preprocessing handoff

All data are in /root/autodl-tmp/C_Problem/data/processed/independent_v1.
This archive includes scripts, metadata and QC reports, NOT the large signal
or feature arrays. Originals are unchanged in /root/autodl-tmp/dataset.

Use /root/autodl-tmp/C_Problem/.venv/bin/python on the cloud instance.

Reproduce on the instance (completed subjects are skipped):
python independent_preprocess.py --trials trial_information.csv
python independent_features.py --root /root/autodl-tmp/C_Problem/data/processed/independent_v1
python finalize_preprocessing.py

For another machine pass --source, --archive and --output to preprocessing.
The report/packaging scripts contain this instance's project root and need
that root changed when used elsewhere. Dependencies are recorded in the lock.

Task 1: use windows_2s.parquet + baseline_features_2s.npy (same row order).
Task 2: feature[:,sensor_indices,:] uses only the selected sensors. Recompute
cross-sensor gait/relative features only when all dependencies are available.
Task 3: SubjectData.read_trial() keeps a whole trial; per-trial quaternion
normalization/sign handling is implemented without changing source arrays.

No standardization is stored in the common cache. Fit scaler/feature choice/
hyperparameters using each fold's training/validation subjects only.
Feature NaNs are undefined low-energy frequency estimates, NOT missing raw
sensor values. Any fitted imputer must also be training-fold-only.

The source publication declares acceleration m/s^2. The teammate's claimed g
interpretation is unverified. No g conversion, gravity subtraction, extra
global rotation, smoothing or label alteration was applied.

Provenance: authors FudanBSRL/Comprehensive-IMU-Dataset, trial_information.csv
https://raw.githubusercontent.com/FudanBSRL/Comprehensive-IMU-Dataset/main/trial_information.csv
Dataset: https://doi.org/10.6084/m9.figshare.30234940.v1
Paper: https://doi.org/10.1038/s41597-026-06710-9
Retain authors' attribution and source license when sharing source data.
'''
(root/'HANDOFF_README.txt').write_text(readme)
out=project/'results/independent_preprocessing_handoff.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['independent_preprocess.py','independent_data_access.py','independent_features.py','finalize_preprocessing.py','package_preprocessing.py']:
        z.write(project/'scripts'/name,'scripts/'+name)
    z.write(project/'requirements-project.txt','environment/requirements-project.txt')
    z.write(project/'docs/environment-lock.txt','environment/environment-lock.txt')
    z.write(project/'data/raw/trial_information.csv','metadata/trial_information.csv')
    for p in root.iterdir():
        if p.is_file(): z.write(p,'reports/'+p.name)
    for p in sorted((root/'subjects').glob('P*/*')):
        if p.name in ['quality.json','complete.json','channel_quality.csv','features_2s_complete.json']:
            z.write(p,'reports/subjects/'+p.parent.name+'/'+p.name)
print(json.dumps({'package':str(out),'bytes':out.stat().st_size,'validation':validation},indent=2))
