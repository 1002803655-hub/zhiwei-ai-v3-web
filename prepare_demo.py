# -*- coding: utf-8 -*-
"""Build a small, provenance-recorded, *real CWRU* demo from locally downloaded MAT files.
NEVER fabricates sample data; files must be present and parseable.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from signal_core import ANALYSIS_FS, DEMO, DEMO_POINTS, file_sha256, read_mat, to_analysis_fs

BASE = Path(__file__).resolve().parent

def build(input_dir: Path, output_dir: Path) -> dict:
    if not input_dir.is_dir():
        raise FileNotFoundError(f'数据文件夹不存在：{input_dir}')
    out = {}
    manifest = {
        'version': 'v3.0-demo-1',
        'source': 'CWRU Bearing Data Center (real experimental test rig recordings)',
        'source_url': 'https://engineering.case.edu/bearingdatacenter/download-data-file',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_fs_hz': ANALYSIS_FS,
        'demo_points_per_record': DEMO_POINTS,
        'classifier_available': False,
        'samples': [],
        'limitation': 'Known CWRU source labels, not model predictions or field fault confirmations. Four source records, 0 HP, not an independent classification validation.',
    }
    for s in DEMO:
        f = input_dir / f'{s.number}.mat'
        if not f.exists():
            raise FileNotFoundError(f'缺少{s.number}.mat；需要先从CWRU官网下载后放入data文件夹')
        x, mat_key, rpm = read_mat(f)
        unified = to_analysis_fs(x, s.original_fs)
        if len(unified) < DEMO_POINTS:
            raise ValueError(f'{s.number}.mat转换后不足{DEMO_POINTS}个点')
        demo = np.asarray(unified[:DEMO_POINTS], dtype=np.float32)
        out[str(s.number)] = demo
        m = {
            'number': s.number,
            'label': s.label,
            'label_code': s.label_code,
            'source_known_label_not_prediction': True,
            'load_hp': s.hp,
            'rpm': rpm,
            'source_fs_hz': s.original_fs,
            'analysis_fs_hz': ANALYSIS_FS,
            'resampling': 'polyphase anti-alias 1/4' if s.original_fs != ANALYSIS_FS else 'none',
            'source_url': f'https://engineering.case.edu/sites/default/files/{s.number}.mat',
            'source_page': s.page,
            'raw_sha256': file_sha256(f),
            'derived_signal_sha256': hashlib.sha256(demo.tobytes()).hexdigest(),
            'variable': mat_key,
            'raw_points': int(len(x)),
            'demo_points': int(len(demo)),
            'offset_points': 0,
            'signal_units': 'source-native units; not calibrated physical severity',
        }
        manifest['samples'].append(m)
        print(f'✓ 已核实{s.number}.mat | {s.label} | {len(x):,} 原始点 → {len(demo):,} 网页演示点')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/'demo_samples.npz', **out)
    (output_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=BASE/'data')
    p.add_argument('--output-dir', type=Path, default=BASE/'assets')
    args = p.parse_args()
    print('开始制作真实CWRU网页演示样本；不会生成任何模拟数据。')
    build(args.input_dir, args.output_dir)
    print(f'\n完成！演示数据: {args.output_dir/"demo_samples.npz"}；来源记录: {args.output_dir/"manifest.json"}')
