# -*- coding: utf-8 -*-
"""Read-only inference for the SHA-256 frozen CWRU RF experimental classifier.
The trained bundle comes from the user's own lab benchmark; never load joblib from an untrusted upload.
"""
from __future__ import annotations
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from ml_features import extract_ml_features, ALL_FEATURES, WIN, FS

MODEL_SHA256 = "8f1506d31e3265baeb056969335ed13468eb0ff64f1e34f9ba5fe2945f9e7b5d"
MODEL_FILE = Path(__file__).resolve().parent / "cwru_rf_combined.joblib"
CLASS_CN={"normal":"正常轴承", "inner_race":"内圈缺陷", "rolling_element":"滚动体缺陷", "outer_race_6":"外圈缺陷（6点钟）"}

@lru_cache(maxsize=1)
def verified_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("模型文件缺失：cwru_rf_combined.joblib")
    found=sha256(MODEL_FILE.read_bytes()).hexdigest()
    if found != MODEL_SHA256:
        raise ValueError("模型文件与冻结评测版本的SHA-256不一致，禁止推理")
    # Important: joblib.load is pickle based. Only deserialize the fixed, verified
    # model bundled with this repository; NEVER deserialize arbitrary user uploads.
    bundle=joblib.load(MODEL_FILE)
    if not isinstance(bundle,dict):
        raise ValueError("冻结模型的数据结构异常")
    expected={"analysis_sample_rate_hz":FS,"window_points":WIN,"source_training_hp":0,
              "sample_origin":"CWRU laboratory"}
    for name,val in expected.items():
        if bundle.get(name)!=val:
            raise ValueError(f"模型元数据不符：{name}")
    if list(bundle.get('feature_names',[]))!=list(ALL_FEATURES):
        raise ValueError("模型的25维特征顺序与训练代码不一致")
    model=bundle.get('model')
    if model is None or int(getattr(model,'n_features_in_',-1))!=len(ALL_FEATURES):
        raise ValueError("模型缺失或输入维度不是25")
    classes=set(getattr(model,'classes_',[]).tolist())
    if classes != set(CLASS_CN):
        raise ValueError("模型类别与四分类定义不匹配")
    return model


def predict_signal(signal:np.ndarray, max_windows:int=24)->dict:
    """Select deterministic nonoverlapping 2048pt windows across given signal and vote.
    Voting fractions are not calibrated probabilities or field failure risks.
    """
    model=verified_model()
    x=np.asarray(signal,dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("振动数据存在NaN/Inf")
    usable=len(x)//WIN
    if usable<1:
        raise ValueError("分类至少需要2048个12kHz采样点")
    selected=np.linspace(0,usable-1,num=min(int(max_windows),usable),dtype=int)
    if len(set(selected.tolist()))!=len(selected):
        raise ValueError("窗口索引发生重复")
    features=[extract_ml_features(x[i*WIN:(i+1)*WIN]) for i in selected]
    matrix=np.asarray([[d[k] for k in ALL_FEATURES] for d in features],dtype=np.float64)
    if matrix.shape!=(len(selected),25):
        raise ValueError("特征矩阵维度不正确")
    predictions=[str(v) for v in model.predict(matrix)]
    counts=Counter(predictions)
    highest=max(counts.values())
    winners=[p for p,n in counts.items() if n==highest]
    # A tie is insufficient for a unique majority prediction.
    voted=winners[0] if len(winners)==1 else None
    return {
       'status':'predicted' if voted is not None else 'tied',
       'predicted_class':voted,
       'predicted_label':CLASS_CN.get(voted,'投票并列，无法给出唯一类别'),
       'windows_evaluated':len(selected),
       'window_size':WIN,
       'analysis_fs_hz':FS,
       'model_sha256':MODEL_SHA256,
       'training_load_hp':0,
       'vote_counts':{k:int(counts.get(k,0)) for k in CLASS_CN},
       'window_predictions':[{'window_index':int(i),'time_s':round(int(i)*WIN/FS,6),
                              'predicted_class':p, 'predicted_label':CLASS_CN[p]}
                             for i,p in zip(selected.tolist(),predictions)],
       'confidence_note':'窗口投票比例不是故障概率；只适用于CWRU同实验台的探索性四分类任务。',
    }
