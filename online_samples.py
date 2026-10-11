# -*- coding: utf-8 -*-
"""Acquire CWRU originals from the university on first boot; build ephemeral in-app examples.

This package does not redistribute recordings or excerpts in its GitHub repository.
The official website may be inaccessible; the app then allows direct user MAT uploads.
"""
from __future__ import annotations

from pathlib import Path
import os
import time
import urllib.request
from urllib.error import HTTPError, URLError

from prepare_demo import build
from signal_core import DEMO


def _fetch(url: str, path: Path) -> None:
    error = None
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        tmp = path.with_suffix('.mat.part')
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ZhiweiAI-Research/3.0; educational-demo)',
                'Accept': '*/*',
            })
            with urllib.request.urlopen(req, timeout=35) as response, tmp.open('wb') as fd:
                written = 0
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > 25*1024*1024:
                        raise ValueError('官方文件超过25 MB，拒绝加载')
                    fd.write(chunk)
            if written < 64*1024:
                raise ValueError(f'下载内容过短：{written} bytes')
            with tmp.open('rb') as fd:
                hdr=fd.read(64)
            if not hdr.startswith(b'MATLAB 5.0 MAT-file'):
                raise ValueError('下载内容不是预期的MATLAB v5数据，可能被网站重定向')
            os.replace(tmp,path)
            return
        except Exception as exc:
            error=exc
            try:tmp.unlink(missing_ok=True)
            except OSError:pass
            if attempt==0:time.sleep(1)
    raise RuntimeError(f'从官方获取 {url} 失败：{type(error).__name__}: {error}')


def ensure_demo_assets(root: Path) -> None:
    assets=root/'assets'
    if (assets/'demo_samples.npz').exists() and (assets/'manifest.json').exists():
        return
    data=root/'data'
    for s in DEMO:
        f=data/f'{s.number}.mat'
        if not f.exists():
            _fetch(f'https://engineering.case.edu/sites/default/files/{s.number}.mat', f)
    build(data,assets)
