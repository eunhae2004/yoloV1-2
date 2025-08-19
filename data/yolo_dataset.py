# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
YOLO v1 데이터셋 로더
- 입력 이미지: {images_root}/**/*.jpg|png
- 라벨: 이미지와 동명이인의 .txt (YOLO v1 스타일) — 각 줄: `class x y w h` (모두 0~1 정규화)
- 인코딩: 타깃 텐서를 S×S×(B*5 + C) 형태로 구성
* 각 셀은 B개의 박스를 가짐: [tx, ty, tw, th, conf]
* 클래스 확률은 셀 단위 원-핫(one-hot) C 차원으로 저장
- 증강: 수평 반전, 컬러 지터(밝기/대비/채도)


핵심 수식(논문 개념) 요약
- 이미지가 S×S 그리드로 분할될 때, 객체 중심이 속한 셀만 해당 객체의 책임을 가짐
- (x, y): 셀 좌상단 기준 셀 내부 상대좌표(0~1), (w, h): 전체 이미지 대비 비율(0~1)
- 총 손실은 좌표/크기/객체/비객체/클래스 항의 가중합이며, 박스 책임은 IoU 최대 박스가 담당(손실에서 처리)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import random
import glob
import math


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.io import load_classes_meta

@dataclass
class AugmentCfg:
    hflip: bool = True
    color_jitter: bool = True




@dataclass
class DataCfg:
    images_root: str
    labels_root: str
    img_size: int = 448
    S: int = 7
    B: int = 2
    C: int = 20
    augment: AugmentCfg = AugmentCfg()


class YoloV1Dataset(Dataset):
    def __init__(self, root: str|Path, img_size: int=448, S: int=7, B: int=2, C: int|None=None,
                    classes_json: str|Path|None=None, augment: dict|None=None):
        self.root = Path(root)
        self.img_size = img_size
        self.S, self.B, self.C = S, B, C
        self.img_dir = self.root / "images"
        self.label_dir = self.root
        self.image_paths = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        assert self.image_paths, f"이미지 파일이 없습니다: {self.img_dir}"


        # 클래스 메타
        self.id2meta = load_classes_meta(classes_json) if classes_json else None
        if C == "auto" and self.id2meta:
            self.C = max(self.id2meta.keys()) + 1
        assert self.C is not None, "클래스 수 C를 지정하거나 classes.json을 제공하세요."


        self.augment = augment or {}


    def __len__(self):
        return len(self.image_paths)


    def _load_label(self, label_path: Path) -> List[Tuple[int,float,float,float,float]]:
        labels = []
        if not label_path.exists():
            return labels
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cid, x, y, w, h = parts
                labels.append((int(cid), float(x), float(y), float(w), float(h)))
        return labels


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")


        img = cv2.imread(str(img_path))
        assert img is not None, f"이미지를 열 수 없습니다: {img_path}"
        H, W = img.shape[:2]
        labels = self._load_label(label_path)


        # 리사이즈
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # 증강
        if self.augment.get("hflip", False) and np.random.rand() < 0.5:
            img = img[:, ::-1, :]
            new_labels = []
            for (cid, x, y, w, h) in labels:
                new_labels.append((cid, 1-x, y, w, h))
            labels = new_labels


        # TODO: 색상 변화(ColorJitter) 단순 구현
        if self.augment.get("color_jitter", False) and np.random.rand() < 0.5:
            img = np.clip(img * (0.8 + 0.4*np.random.rand()), 0, 255).astype(np.uint8)


        # 정규화
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1)


        # 타겟 텐서 (S,S,B*5+C)
        assert isinstance(self.C, int), "self.C must be an integer before using in tensor shape."
        target = torch.zeros((self.S, self.S, self.B*5 + self.C), dtype=torch.float32)
        cell_size = 1.0 / self.S


        for (cid, x, y, w, h) in labels:
            i, j = int(y * self.S), int(x * self.S)
            if i >= self.S or j >= self.S:
                continue
            cx, cy = x * self.S - j, y * self.S - i
            for b in range(self.B):
                target[i,j,b*5:(b+1)*5] = torch.tensor([cx, cy, w, h, 1.0])
            target[i,j,self.B*5+cid] = 1.0


        return img, target




def build_dataloader(root: str|Path, batch_size: int=16, img_size: int=448, S: int=7, B: int=2, C: int|None=None,
            classes_json: str|Path|None=None, augment: dict|None=None, num_workers: int=4, shuffle: bool=True):
    dataset = YoloV1Dataset(root, img_size, S, B, C, classes_json, augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


