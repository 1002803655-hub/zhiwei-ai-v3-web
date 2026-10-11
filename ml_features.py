# -*- coding: utf-8 -*-
"""Exact 25-dimensional feature schema copied from the frozen CWRU benchmark.
IMPORTANT: keep this file unchanged unless model is re-trained and re-validated.
"""
import numpy as np
from scipy.stats import kurtosis, skew

FS=12000


WIN=2048


BANDS=[(0,400),(400,800),(800,1600),(1600,2400),(2400,3200),(3200,4400),(4400,6000)]


TIME_FEATURES=['rms','mean_abs','std','peak_abs','peak_to_peak','skewness','kurtosis_pearson','crest_factor','impulse_factor','shape_factor','clearance_factor','zero_crossing_fraction']


FREQ_FEATURES=['spectral_centroid','spectral_spread','spectral_flatness','spectral_entropy','spectral_rolloff_85','peak_frequency']+[f'band_energy_ratio_{lo}_{hi}' for lo,hi in BANDS]


ALL_FEATURES=TIME_FEATURES+FREQ_FEATURES


def extract_ml_features(window:np.ndarray)->dict:
    x=np.asarray(window,dtype=np.float64).reshape(-1)
    if x.size!=WIN or not np.isfinite(x).all():
        raise ValueError('窗口长度或数值异常')
    x=x-np.mean(x)
    absx=np.abs(x)
    rms=float(np.sqrt(np.mean(x*x)))
    avg_abs=float(np.mean(absx))
    peak=float(np.max(absx))
    ptp=float(np.ptp(x))
    std=float(np.std(x))
    epsilon=1e-15
    win=np.hanning(WIN+1)[:-1]
    fft=np.fft.rfft(x*win)
    psd=np.abs(fft)**2
    freq=np.fft.rfftfreq(WIN,1.0/FS)
    total=float(np.sum(psd[1:]))
    if total<epsilon:
        spec=np.zeros_like(psd)
    else:
        spec=psd/total
        spec[0]=0.
    centroid=float(np.sum(freq*spec))
    spread=float(np.sqrt(np.sum((freq-centroid)**2*spec)))
    valid_psd=psd[1:]
    flatness=float(np.exp(np.mean(np.log(valid_psd+epsilon)))/(np.mean(valid_psd)+epsilon))
    positive=spec[1:]
    entropy=float(-np.sum(positive*np.log(positive+epsilon))/np.log(len(positive)))
    rollover=int(np.searchsorted(np.cumsum(spec),0.85))
    idx=int(np.argmax(psd[1:])+1)
    d={
      'rms':rms,'mean_abs':avg_abs,'std':std,'peak_abs':peak,'peak_to_peak':ptp,
      'skewness':float(skew(x,bias=False)) if std>epsilon else 0.,
      'kurtosis_pearson':float(kurtosis(x,fisher=False,bias=False)) if std>epsilon else 0.,
      'crest_factor':float(peak/(rms+epsilon)),
      'impulse_factor':float(peak/(avg_abs+epsilon)),
      'shape_factor':float(rms/(avg_abs+epsilon)),
      'clearance_factor':float(peak/(float(np.mean(np.sqrt(absx)))**2+epsilon)),
      'zero_crossing_fraction':float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))),
      'spectral_centroid':centroid,'spectral_spread':spread,
      'spectral_flatness':flatness,'spectral_entropy':entropy,
      'spectral_rolloff_85':float(freq[min(rollover,len(freq)-1)]),
      'peak_frequency':float(freq[idx]),
    }
    for lo,hi in BANDS:
        is_last=hi==BANDS[-1][1]
        mask=(freq>=lo)&((freq<=hi) if is_last else (freq<hi))
        d[f'band_energy_ratio_{lo}_{hi}']=float(np.sum(spec[mask]))
    if sorted(d)!=sorted(ALL_FEATURES) or not all(np.isfinite(list(d.values()))):
        raise ValueError('特征提取输出错误或者存在非有限数')
    return d

