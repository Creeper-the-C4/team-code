"""Read-only terminal progress display for the isolated Task 1 V4 run."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import subprocess
import sys
import time

DEFAULT_ROOT = Path('/root/autodl-tmp/C_Problem/results/task1_v4_20261128')
BJT = timezone(timedelta(hours=8))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(root):
    state = read_json(root / 'stage_status.json')
    manifest = read_json(root / 'run_manifest.json')
    specs = manifest['specifications']
    rows = []
    finished = 0
    for spec in specs:
        name = f"{spec['family']}_{spec['window_seconds']}s"
        marks = []
        count = 0
        for fold in range(6):
            path = root / 'task1_v4' / name / f'fold_{fold}' / 'complete.json'
            done = path.exists() and read_json(path).get('complete') is True
            count += int(done)
            marks.append('OK' if done else '--')
        finished += count
        active = '  <-- 当前配置' if name == state.get('current_experiment') and not state['complete'] else ''
        rows.append(f"{name:<9}  {' '.join(marks)}    {count}/6{active}")
    total = len(specs) * 6
    filled = int(30 * finished / total)
    bar = '=' * filled + '.' * (30 - filled)
    process = subprocess.run(['pgrep', '-f',
        r'/root/autodl-tmp/C_Problem/scripts/run_task1_v4_cloud[.]py'],
        capture_output=True, text=True)
    running = process.returncode == 0
    failure = root / 'failure.json'
    if failure.exists():
        status = '出现错误，请检查下方错误记录'
    elif state['complete']:
        status = '全部完成，汇总核验已通过'
    elif running:
        status = '云端任务运行中'
    else:
        status = '运行进程已退出，尚无完成标记，请检查总日志'
    phase = {'feature_extraction': '特征提取', 'feature_validation': '特征核验',
             'training': '正式训练', 'complete': '已完成'}.get(state['stage'], state['stage'])
    if finished == total and not state['complete'] and not failure.exists():
        phase = '训练折次已齐，等待汇总核验'
    started = datetime.fromisoformat(manifest['started_utc'])
    elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
    lines = ['第一题 V4 云端进度',
        '刷新时间：' + datetime.now(BJT).strftime('%Y-%m-%d %H:%M:%S 北京时间'),
        '任务状态：' + status,
        f'当前阶段：{phase}    本轮入口运行时间：{elapsed // 3600:02}:{elapsed % 3600 // 60:02}:{elapsed % 60:02}',
        f"特征任务：{len(state['finished_feature_jobs'])}/60",
        f'训练折次：[{bar}] {finished}/{total} ({finished / total:.1%})',
        f"完整配置：{len(state['experiments_complete'])}/{len(specs)}（含结果核验）", '',
        '配置       六折依次为 fold_0 至 fold_5，OK=完成', *rows, '']
    current = state.get('current_experiment')
    if current:
        log = root / 'logs' / f'{current}.log'
        if log.exists():
            lines += [f'当前配置最近日志：{current}', *log.read_text(encoding='utf-8', errors='replace').splitlines()[-6:], '']
    if failure.exists():
        lines += ['最近错误：', read_json(failure).get('traceback', '')[-2000:]]
    lines += ['按折更新；单折训练期间数字可能数分钟不变。',
              '此面板只读取进度。按 Ctrl+C 退出查看，后台训练继续。',
              '总日志：/root/autodl-tmp/C_Problem/logs/task1_v4_20261128.log']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--follow', action='store_true', help='Refresh every five seconds until Ctrl+C')
    args = parser.parse_args()
    try:
        while True:
            view = render(args.root)
            if args.follow and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(view, flush=True)
            if not args.follow:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n已退出进度查看，后台训练继续。')


if __name__ == '__main__':
    main()
