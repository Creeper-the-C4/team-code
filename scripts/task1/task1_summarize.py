from pathlib import Path
import argparse,json,zipfile,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    s=json.loads((r/'summary.json').read_text())
    classes=pd.read_csv(r/'per_class_oof.csv')
    subjects=pd.read_csv(r/'per_subject_metrics.csv')
    folds=pd.read_csv(r/'fold_metrics.csv')
    groups=pd.read_csv(r/'window_group_metrics.csv')
    cm=pd.read_csv(r/'confusion_oof.csv',index_col=0)
    values=cm.to_numpy();norm=values/np.maximum(values.sum(axis=1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(12,10),layout='constrained')
    im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    labels=[x.replace('_',' ') for x in classes['name']]
    ax.set_xticks(range(13),labels,rotation=55,ha='right',fontsize=9)
    ax.set_yticks(range(13),labels,fontsize=9)
    for i in range(13):
        for j in range(13):
            if norm[i,j]>=0.005:
                ax.text(j,i,f'{norm[i,j]:.0%}',ha='center',va='center',fontsize=8,
                        color='white' if norm[i,j]>.55 else 'black')
    ax.set_xlabel('Predicted activity');ax.set_ylabel('True activity')
    ax.set_title('Task 1 LightGBM baseline: held-out predictions from all six folds')
    fig.colorbar(im,ax=ax,label='Fraction within true activity',shrink=.8)
    fig.savefig(r/'confusion_oof_normalized.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    ax.barh(labels,classes.f1,color=['#9B6A37']+['#346A96']*12)
    ax.invert_yaxis();ax.set_xlim(0,1.08)
    for i,v in enumerate(classes.f1):ax.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_xlabel('F1 score');ax.set_title('Per-activity F1: pooled held-out windows, including Others')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(r/'per_class_f1.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.3),layout='constrained')
    ax.bar(subjects.subject,subjects.accuracy_13,color='#346A96')
    ax.set_xticks(subjects.subject,labels=[str(s) for s in subjects.subject],rotation=60,fontsize=8)
    ax.set_ylim(0,1.02);ax.set_xlabel('Held-out participant');ax.set_ylabel('Accuracy')
    ax.set_title('Across-participant variation: each participant tested once')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(r/'per_subject_accuracy.png',dpi=170);plt.close(fig)
    pairs=[]
    for i in range(13):
        for j in range(13):
            if i!=j:pairs.append({'true':labels[i],'predicted':labels[j],'count':int(values[i,j]),'true_class_fraction':float(norm[i,j])})
    pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(r/'confusion_pairs.csv',index=False)
    worst=classes.sort_values('f1').head(4)
    report=f'''Task 1 LightGBM 基线测试结果

数据与协议
独立底座 independent_v1；30名参与者；220,702个2秒窗；17个传感器、每传感器40项，共680项固定特征。
保留全部13类和所有纯度窗口。每折20人训练、5人验证、5人测试；按参与者隔离。
固定LightGBM参数，最多300轮；仅在内层验证参与者的13类Macro-F1上早停。没有超参数搜索、平滑或全体标准化。
每位参与者恰好得到一次外层测试预测；全部220,702个窗口均已覆盖。

主要测试结果
合并OOF 13类Macro-F1：{s['oof']['macro_f1_13']:.6f}
合并OOF 准确率：{s['oof']['accuracy_13']:.6f}
六折13类Macro-F1均值：{s['fold_mean']['macro_f1_13']:.6f}
六折13类Macro-F1标准差：{s['fold_sd']['macro_f1_13']:.6f}
12个命名活动Macro-F1（全部测试窗，含Others造成的误报）：{s['oof']['macro_f1_named12_all_windows']:.6f}
12个命名活动Macro-F1（仅真实标签非Others窗口）：{s['macro_f1_named12_true_named_only']:.6f}
三档步行Macro-F1（全部测试窗）：{s['oof']['walk3_macro_f1_all_windows']:.6f}
参与者聚类bootstrap的13类Macro-F1 95%区间：{s['subject_cluster_bootstrap_macro_f1_13_ci95']}
该区间以现有六个模型为条件，不包含重新训练模型带来的不确定性。折间标准差为描述量，不按独立试验解释。

逐类结果
{classes.to_string(index=False)}

逐折结果
{folds[['fold','accuracy_13','macro_f1_13','best_iteration','training_seconds']].to_string(index=False)}

稳定与过渡窗口
{groups[['group','n_windows','accuracy_13','macro_f1_13']].to_string(index=False)}
分组13类Macro-F1仍固定包含0至12类；当某分组缺少类别时该指标受类别覆盖影响，应结合逐类支持数和准确率阅读。

主要待改善类别
{worst[['name','f1','support']].to_string(index=False)}

解释与后续
这是LightGBM独立基线，不是队友CNN的同实现复现。队友报告的12类分数与本轮主指标口径不同，不能直接据此认定模型或处理流程更优。
加速度保留原始数值；g与m/s²的分歧尚未解决。未进行额外去重力或重复旋转；基础主频也不自动称为步频。
后续若改特征、Others处理、过渡窗口或时间平滑，只用训练/验证参与者选型；不得用本轮外层测试分数反复择优后仍宣称独立最终评估。

云端结果目录
{r}
六折模型在 fold_0 至 fold_5/model.txt；逐窗预测在 oof_predictions.parquet。
小体积报告包不含模型和逐窗概率数组，完整产物保留在云端。
'''
    (r/'Task1_LightGBM_baseline_report.txt').write_text(report)
    archive=r.parent/'task1_lightgbm_baseline_v1_report.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in r.iterdir():
            if f.is_file() and f.suffix in ['.json','.csv','.txt','.png']:
                z.write(f,f.name)
        script=Path('/root/autodl-tmp/C_Problem/scripts/task1_lightgbm_baseline.py')
        z.write(script,'scripts/'+script.name)
    print(json.dumps({'report_archive':str(archive),'bytes':archive.stat().st_size,
                       'macro_f1_13':s['oof']['macro_f1_13'],'accuracy':s['oof']['accuracy_13']},indent=2),flush=True)

if __name__=='__main__':main()
