# -*- coding: utf-8 -*-
"""
시각화 유틸리티
- 탐지 결과를 이미지에 그리기(박스+라벨 텍스트)
- 라벨 텍스트: type / color(영→한 축약) / score
- 클래스별 고정 팔레트(시드 고정 시 재현성 유지)
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import random
import cv2
import numpy as np


from .io import color_en_to_kr


# 고정 팔레트 생성 (재현성을 위해 고정 시드 사용)
random.seed(777)
PALETTE = [
    (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    for _ in range(256)
]




def _color_for_class(cid: int) -> Tuple[int, int, int]:
    return PALETTE[cid % len(PALETTE)]




def draw_box(img: np.ndarray, xyxy: Tuple[float,float,float,float], color: Tuple[int,int,int], thickness: int = 2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)




def draw_label(img: np.ndarray, x1: int, y1: int, text: str, color: Tuple[int,int,int]):
# 텍스트 배경 박스
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    t = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, t)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 3), font, scale, (0,0,0), t, cv2.LINE_AA)




def draw_detections(
    img: np.ndarray,
    dets: List[dict],
    id2meta: Dict[int, Dict[str, str]] | None = None,
    show_score: bool = True,
) -> np.ndarray:
    """
    Args:
    img: BGR 이미지 (cv2.imread 결과)
    dets: 각 원소는 {"bbox_xyxy": [x1,y1,x2,y2], "class_id": int, "confidence": float}
    id2meta: {class_id: {"type": str, "color": str, ...}}
    Returns:
    시각화가 그려진 이미지 (복사본)
    """
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cid = int(d.get("class_id", 0))
        score = float(d.get("confidence", 0.0))
        color = _color_for_class(cid)


        draw_box(out, (x1,y1,x2,y2), color, 2)


        typ = color_en = None
        if id2meta is not None and cid in id2meta:
            typ = id2meta[cid].get("type", None)
            color_en = id2meta[cid].get("color", None)
        color_kr = color_en_to_kr(color_en) if color_en else None


        label_parts = []
        if typ: label_parts.append(str(typ))
        if color_kr: label_parts.append(str(color_kr))
        if show_score: label_parts.append(f"{score:.2f}")
        text = " | ".join(label_parts) if label_parts else f"id={cid} | {score:.2f}"


        draw_label(out, int(x1), int(y1), text, color)
    return out