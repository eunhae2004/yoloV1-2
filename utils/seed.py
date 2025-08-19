# -*- coding: utf-8 -*-
"""
재현성 보장을 위한 시드 고정 유틸리티
- YOLO v1 실험에서 같은 입력/하이퍼파라미터로 동일 결과가 나오도록 난수원을 고정한다.
"""
from __future__ import annotations
import os, random
import numpy as np


try:
    import torch
except ImportError:
        torch = None




def set_seed(seed: int = 42, deterministic: bool = True):
    """
    모든 주요 난수원을 고정한다.
    Args:
    seed: 난수 시드 값
    deterministic: True면 CUDA 연산을 결정적으로 설정(약간의 속도 저하 가능)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True