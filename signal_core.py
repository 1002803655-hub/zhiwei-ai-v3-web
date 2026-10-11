# -*- coding: utf-8 -*-
"""CWRU vibration sample explorer: deterministic signal processing, no classifier.

Note: Normal data (97.mat) are sourced at 48kHz; DE fault files here are 12kHz.
Resampling uses a proper anti-aliasing polyphase filter.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import kurtosis

ANALYSIS_FS = 12000
WINDOW = 4096
DEMO_POINTS = 16384
DATA_SOURCE = 'https://engineering.case.edu/bearingdatacenter/download-data-file'

@dataclass(frozen=True)
class Record:
    number: int
    label: str
    label_code: str
    hp: int
    original_fs: int
    page: str

DEMO = (
    Record(97, '正常轴承', 'normal', 0, 48000, 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'),
    Record(105, '内圈缺陷', 'inner_race', 0, 12000, 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'),
    Record(118, '滚动体缺陷', 'rolling_element', 0, 12000, 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'),
    Record(130, '外圈缺陷（6点钟）', 'outer_race_6', 0, 12000, 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'),
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_mat(path: Path | bytes) -> tuple[np.ndarray, str, Optional[float]]:
    if isinstance(path, bytes):
        mat = loadmat(io.BytesIO(path))
    else:
        mat = loadmat(str(path))
    de_keys = [key for key in mat if key.lower().endswith('_de_time')]
    if len(de_keys) != 1:
        raise ValueError(f'需要恰好一个*_DE_time变量；实际找到：{de_keys}')
    key = de_keys[0]
    data = np.asarray(mat[key])
    if data.dtype.kind not in 'fi' or data.ndim > 2:
        raise ValueError('驱动端振动信号格式错误')
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if len(data) < WINDOW or not np.isfinite(data).all():
        raise ValueError('振动采样点不足4096个，或包含无效数值')
    if len(data) > 10_000_000:
        raise ValueError('采样点数量超过网页版处理限制')
    rpm_keys = [k for k in mat if k.lower().endswith('rpm')]
    rpm = None
    if rpm_keys:
        rpm_data = np.asarray(mat[rpm_keys[0]]).reshape(-1)
        if rpm_data.size:
            v = float(rpm_data[0])
            if math.isfinite(v) and v > 0:
                rpm = v
    return data, key, rpm


def to_analysis_fs(values: np.ndarray, original_fs: int) -> np.ndarray:
    if original_fs not in (12000, 48000):
        raise ValueError('本版仅支持12kHz和48kHz采样数据')
    if original_fs == 48000:
        return np.asarray(resample_poly(values, up=1, down=4), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def extract_features(values: np.ndarray, fs: int = ANALYSIS_FS) -> dict:
    if fs <= 0:
        raise ValueError('采样率必须大于0')
    if values.ndim != 1 or len(values) != WINDOW:
        raise ValueError(f'分析窗口必须为{WINDOW}个点')
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('窗口存在无效数值')
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    kurt = float(kurtosis(x, fisher=False, bias=False)) if rms else 0.0
    win = np.hanning(len(x) + 1)[:-1]  # periodic Hann, same as scipy get_window('hann', ...)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spectrum = np.fft.rfft(x * win)
    amps = (2.0 / float(np.sum(win))) * np.abs(spectrum)
    amps[0] *= 0.5
    if len(x) % 2 == 0:
        amps[-1] *= 0.5
    idx = int(np.argmax(amps[1:]) + 1)
    centroid = float(np.sum(f * np.abs(spectrum)**2) / np.sum(np.abs(spectrum)**2)) if np.any(spectrum) else 0.0
    return {
        'rms_signal_units': rms,
        'peak_signal_units': peak,
        'kurtosis_pearson': kurt,
        'crest_factor': peak/rms if rms else None,
        'dominant_fft_hz': float(f[idx]),
        'spectral_centroid_hz': centroid,
        'waveform': x,
        'time_seconds': np.arange(len(x)) / fs,
        'fft_hz': f,
        'fft_amplitude': amps,
    }


def load_builtin_assets(base: Path) -> tuple[dict, dict]:
    npz_path = base / 'assets' / 'demo_samples.npz'
    manifest_path = base / 'assets' / 'manifest.json'
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('缺少内置演示样本包。请先执行01_制作内置演示样本.bat。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('analysis_fs_hz') != ANALYSIS_FS:
        raise ValueError('内置样本包的统一采样率不符合当前版本')
    arrs = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for m in manifest['samples']:
            k = str(m['number'])
            x = np.asarray(z[k], dtype=np.float64)
            if x.ndim != 1 or len(x) < WINDOW or len(x) > 100_000 or not np.isfinite(x).all():
                raise ValueError(f'{k}样本长度或数值不符合要求')
            sig_hash = hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()
            if m.get('derived_signal_sha256') != sig_hash:
                raise ValueError(f'{k}演示数据校验失败，拒绝分析')
            arrs[k] = x
    return manifest, arrs
