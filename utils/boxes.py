# -*- coding: utf-8 -*-
"""
YOLO v1 박스 유틸리티
- 좌표계 변환: xywh↔xyxy (정규화/픽셀 좌표)
- IoU 계산, NMS(비최대억제)
- 스케일 복원: 모델 입력 크기(예: 448) → 원본 이미지 크기(W,H)
- YOLO v1 셀 좌표 복원: (cx, cy)는 그리드 셀 내부 상대좌표, (w,h)는 이미지 전체에 대한 비율


용어
- xyxy: [x1, y1, x2, y2] (좌상단, 우하단) 픽셀 좌표
- xywh: [xc, yc, w, h] (중심, 폭, 높이) 픽셀 좌표
- nxywh: [xc, yc, w, h] (0~1 정규화)


참고(논문 요지)
- 이미지가 SxS 그리드로 분할될 때 각 셀은 B개의 박스를 예측
- (x,y): 셀 좌상단을 (0,0)로 하여 셀 내부에서의 상대 위치(0~1)
- (w,h): 전체 이미지 대비 비율(0~1)
- confidence = P(object) * IoU(pred, gt)
"""
from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np


# ----------------------------
# 좌표 변환
# ----------------------------


def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """중심-폭-높이(xywh) → 모서리(xyxy) (픽셀 좌표 기준)
    Args:
    xc, yc: 중심
    w, h : 폭, 높이
    """
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2




def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """모서리(xyxy) → 중심-폭-높이(xywh) (픽셀 좌표 기준)"""
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc, yc, w, h




def clamp_xyxy(x1, y1, x2, y2, W, H):
    """박스를 이미지 경계 안으로 자른다."""
    x1 = float(max(0, min(W - 1, x1)))
    y1 = float(max(0, min(H - 1, y1)))
    x2 = float(max(0, min(W - 1, x2)))
    y2 = float(max(0, min(H - 1, y2)))
    return x1, y1, x2, y2


# ----------------------------
# 정규화 <-> 픽셀 스케일
# ----------------------------


def nxywh_to_xyxy(nxc: float, nyc: float, nw: float, nh: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """0~1 정규화 xywh → 픽셀 xyxy"""
    xc, yc, w, h = nxc * W, nyc * H, nw * W, nh * H
    return xywh_to_xyxy(xc, yc, w, h)




def xyxy_to_nxywh(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """픽셀 xyxy → 0~1 정규화 xywh"""
    xc, yc, w, h = xyxy_to_xywh(x1, y1, x2, y2)
    return xc / W, yc / H, w / W, h / H


# ----------------------------
# 스케일 복원(리사이즈만 가정)
# ----------------------------


def scale_boxes_from_model(xyxy_boxes: np.ndarray, src_W: int, src_H: int, model_W: int, model_H: int) -> np.ndarray:
    """
    모델 입력 크기(model_W, model_H) 기준의 픽셀 xyxy → 원본 이미지(src_W, src_H) 크기로 복원
    - 단순 리사이즈 가정(letterbox 미사용)
    """
    sx = float(src_W) / float(model_W)
    sy = float(src_H) / float(model_H)
    out = xyxy_boxes.copy().astype(np.float32)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


# ----------------------------
# IOU & NMS
# ----------------------------


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """두 박스(a,b)의 IoU. 각 shape=(4,) [x1,y1,x2,y2]"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, topk: int | None = None) -> List[int]:
    """비최대억제(NMS)
    Args:
    boxes: (N,4) xyxy
    scores: (N,)
    iou_thr: IoU 임계값
    topk: 선택 개수 제한(옵션)
    Returns:
    keep 인덱스 리스트
    """
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1 and scores.shape[0] == boxes.shape[0]


    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if topk is not None and len(keep) >= topk:
            break
        if order.size == 1:
         break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_thr]
    return keep


# ----------------------------
# YOLO v1 셀 좌표 복원
# ----------------------------


def cell_pred_to_nxywh(cx: float, cy: float, w: float, h: float, i: int, j: int, S: int) -> Tuple[float, float, float, float]:
    """
    YOLO v1 헤드 출력(셀 단위) → 정규화 xywh
    Args:
    cx, cy: 셀 내부 상대 좌표(0~1)
    w, h : 이미지 전체 대비 비율(0~1)
    i, j : 셀 인덱스 (row=i, col=j)
    S : 그리드 크기
    """
    # (x,y)는 셀 좌상단 기준에서 상대좌표이므로 전체 좌표로 변환 시 (j + cx)/S, (i + cy)/S
    nxc = (j + cx) / float(S)
    nyc = (i + cy) / float(S)
    return nxc, nyc, w, h




def cell_pred_to_xyxy(cx: float, cy: float, w: float, h: float, i: int, j: int, S: int, W: int, H: int) -> Tuple[float, float, float, float]:
    """셀 단위 예측을 원본 픽셀 xyxy로 변환"""
    nxc, nyc, nw, nh = cell_pred_to_nxywh(cx, cy, w, h, i, j, S)
    return nxywh_to_xyxy(nxc, nyc, nw, nh, W, H)