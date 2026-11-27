"""Summarize independently prepared data, including QC and reproducible handoff."""
import json,hashlib,re,zipfile,datetime,shutil
from pathlib import Path
import numpy as np
import pandas as pd

project=Path('/root/autodl-tmp/C_Problem')
root=project/'data/processed/independent_v1'
reports=[json.loads(p.read_text()) for p in sorted((root/'subjects').glob('P*/quality.json'))]
assert len(reports)==30,'Full processing must finish before publishing report'
assert json.loads((root/'summary.json').read_text())['completed_subjects']==30
frame_hist=np.zeros(13,dtype=np.int64)
window_hist={2:np.zeros(13,dtype=np.int64),4:np.zeros(13,dtype=np.int64)}
window_stats={2:{'mixed':0,'purity_below_0_9':0,'invalid':0},4:{'mixed':0,'purity_below_0_9':0,'invalid':0}}
physics=[]
subjects=[]
for report in reports:
    frame_hist+=np.array([report['labels'][str(i)] for i in range(13)])
    subject=report['subject']
    for seconds in [2,4]:
        rec=report['windows'][str(seconds)]
        for label,n in rec['label_counts'].items(): window_hist[seconds][int(label)]+=n
        window_stats[seconds]['mixed']+=rec['mixed_label_windows']
        window_stats[seconds]['purity_below_0_9']+=rec['purity_below_0_9']
        window_stats[seconds]['invalid']+=rec['invalid_signal_windows']
    for p in report['rotation_diagnostics']:
        row={'subject':subject,'sensor':p['sensor'],
             'standing_acc_norm_median_source_units':p.get('sample_standing_acc_norm_median_raw_units'),
             'sign_flip_pairs':p.get('quat_sign_flip_pairs'),
             'valid_pairs':p.get('valid_adjacent_pairs')}
        for mode,rec in p.get('gyro_convention_fits',{}).items():
            row[mode+'_r2']=rec['r2_scaled']
            row[mode+'_scale']=rec['observed_over_quat_rate_scale']
        physics.append(row)
    row={'subject':subject,'rows':report['rows'],'trials':report['boundaries']['trial_count'],
         'two_second_windows':report['windows']['2']['count'],
         'mixed_window_fraction':report['windows']['2']['mixed_label_windows']/report['windows']['2']['count'],
         'others_frame_fraction':report['labels']['0']/report['rows']}
    subjects.append(row)
pd.DataFrame(physics).to_csv(root/'physics_diagnostics.csv',index=False)
pd.DataFrame(subjects).to_csv(root/'subject_distribution.csv',index=False)
pd.DataFrame({'activity':range(13),'source_frames':frame_hist,
              'windows_2s_majority':window_hist[2],'windows_4s_majority':window_hist[4]}).to_csv(root/'activity_distribution.csv',index=False)
body=pd.read_csv(root/'anthropometric_original.csv')
numeric_ids=pd.to_numeric(body['Participant ID'],errors='coerce')
notes=body.loc[numeric_ids.isna()]
assert notes.drop(columns=['Participant ID']).isna().all().all(),'Unexpected non-participant data rows'
(root/'anthropometric_definitions.txt').write_text('\n'.join(notes['Participant ID'].astype(str)))
body=body.loc[numeric_ids.notna()].copy()
assert np.all(numeric_ids.dropna()==np.floor(numeric_ids.dropna()))
body['Participant ID']=numeric_ids.dropna().astype(int)
assert len(body)==30 and body['Participant ID'].nunique()==30
assert set(body['Participant ID'])==set(range(1,31))
converted=pd.DataFrame({'subject':body['Participant ID']})
column_mapping={}
for col in body.columns:
    if '(cm)' in col:
        name=re.sub(r'[^a-z0-9]+','_',col.replace('(cm)','').strip().lower()).strip('_')+'_m'
        values=pd.to_numeric(body[col],errors='raise')/100
        assert values.notna().all() and (values>0).all()
        converted[name]=values
        column_mapping[name]={'source_column':col,'conversion':'divide by 100','source_unit':'cm','output_unit':'m'}
converted.to_csv(root/'body_dimensions_m.csv',index=False)
(root/'body_dimension_schema.json').write_text(json.dumps(column_mapping,indent=2))
manifest=[]
for r in reports:
    manifest.append({'file':f'dataset/P{r["subject"]:02d}.csv',**r['source_integrity'],
                     'crc_matches_zip_entry':r['extracted_crc_matches_zip']})
anthro=project.parent/'dataset'/'anthropometric information.csv'
with zipfile.ZipFile(project.parent/'dataset.zip') as z:
    import zlib
    data=anthro.read_bytes()
    e=z.getinfo('dataset/anthropometric information.csv')
    crc=zlib.crc32(data)&0xffffffff
    assert crc==e.CRC and len(data)==e.file_size
manifest.append({'file':'dataset/anthropometric information.csv','bytes':len(data),
                 'sha256':hashlib.sha256(data).hexdigest(),'crc32':f'{crc:08x}','crc_matches_zip_entry':True})
(root/'source_file_manifest.json').write_text(json.dumps(manifest,indent=2))
summary=json.loads((root/'summary.json').read_text())
summary['free_disk_bytes']=shutil.disk_usage(root).free
summary['report_generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
access=json.loads((root/'access_validation.json').read_text())
summary.update({'source_files_checked':len(manifest),'all_extracted_crc_match_archive':True,
                'total_duration_hours':summary['total_rows']/60/3600,
                'window_statistics':window_stats,'anthropometric_subjects':len(body),
                'anthropometric_definition_rows':len(notes),
                'access_validation_passed':access['passed'],
                'source_crc_scope':'Extracted file size and full-file CRC compared with uploaded ZIP directory; no upstream cryptographic checksum available.',
                'unit_decision':'No acceleration rescaling; no gravity subtraction; raw channels retained.',
                'teammate_2s_windows_reported':220702,
                'teammate_window_count_difference':summary['windows_2s']-220702})
summary['invalid_quaternion_samples']=sum(sum(r['quaternion_invalid_count']) for r in reports)
summary['quaternion_norm_deviations_over_1e_3']=sum(sum(r['quaternion_norm_deviation_over_1e_3']) for r in reports)
summary['quaternion_sign_flip_pairs']=sum(p.get('quat_sign_flip_pairs',0) for r in reports for p in r['rotation_diagnostics'])
phys=pd.DataFrame(physics)
summary['gyro_world_fit_median_r2']=float(phys.world_r2.median())
summary['gyro_body_fit_median_r2']=float(phys.body_r2.median())
summary['gyro_diagnostics_scope']='Every-second adjacent-frame pairs inside trials, quaternion-rate magnitude 0.2-20 rad/s; best scalar fit. Convention diagnostic only.'
features=root/'baseline_feature_summary.json'
if features.exists():summary['baseline_features']=json.loads(features.read_text())
(root/'final_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
text=f'''C题独立数据预处理结果
更新时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}

完成范围
30名参与者、{summary['total_trials']:,}个作者提供的试次。
原始帧数：{summary['total_rows']:,}；60 Hz，总时长约{summary['total_duration_hours']:.2f}小时。
2秒窗口：{summary['windows_2s']:,}；4秒窗口：{summary['windows_4s']:,}；步长均为半窗。
2秒窗数量与队友报告差值：{summary['teammate_window_count_difference']}。数量相同不证明全部预处理一致。
全量信号NaN：{summary['nan_count']}；无穷值：{summary['inf_count']}；未被试次覆盖的行：{summary['uncovered_rows']}。
2秒混合标签窗：{window_stats[2]['mixed']:,}；纯度低于0.90的窗：{window_stats[2]['purity_below_0_9']:,}。
所有窗口均保留来源索引、主标签、中心标签、13类计数、标签变化次数和纯度。
31份原始CSV的大小和全文件CRC均与现有压缩包条目一致，并记录SHA256；未获得上游独立校验值。

共享数据约定
缓存为float32连续信号[frame,17,10]及int8标签，无重复物化窗口。
保留原始单位、坐标系和通道值；未乘9.81、未去重力、未对Acc/Gyr重复旋转。
原论文声明加速度单位为m/s²；队友的g解释尚未独立证实，后续物理量分析需继续核对。
四元数原始xyzw保留在缓存；读取时在单一试次/窗口内部归一化及符号连续化。
人体测量原表保留，另生成明确由cm换算为m的身体尺寸表。
人体表包含30行参与者及3行解剖定义注释；注释已单独保留，不作为缺失参与者或参与建模。
没有进行公共标准化、插补、滤波或标签改写。通道质量表均值仅作描述，不能当训练归一化参数。

评估与屏蔽
保存与队友兼容的六折参与者划分；每折20人训练、5人验证、5人测试。
这是固定内层验证，不能称为完整内层交叉验证。模型选型和布局选择仅使用训练/验证参与者。
读取接口先选择可用传感器，再进行变换；基础特征每传感器独立，避免通过隐藏传感器的派生量泄漏。
2秒基础频谱峰不自动称为步频，频率分辨率为0.5 Hz；低能量窗口的主频允许NaN，插补只能在训练折拟合。
所有正式训练和模型评价尚未启动。

云端位置
数据：{root}
脚本：{project/'scripts'}
环境：{project/'.venv/bin/python'}
日志：{project/'logs/independent_all.log'}

下一步
基于此底座跑分类基线；与队友代码、划分、缓存及逐样本预测进行对照。
Task 3使用完整试次及身体尺寸，传感器到身体节段对齐仍需验证。
'''
(root/'preprocessing_report.txt').write_text(text)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
