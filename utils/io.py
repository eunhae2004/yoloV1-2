# -*- coding: utf-8 -*-
"""
입출력/경로/설정 로딩 유틸리티
- YAML cfg 로딩, classes.json 파싱( class_id → {type, color} ), 카테고리/색상 맵 제공
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import yaml


COLOR_EN_TO_KR = {
    "red": "빨",
    "orange": "주",
    "yellow": "노",
    "green": "초",
    "blue": "파",
    "purple": "보",
    "black": "검",
    "white": "흰",
}




def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p




def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def save_json(path: str | Path, obj: Any):
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)




def load_classes_meta(classes_json_path: str | Path) -> Dict[int, Dict[str, str]]:
    """
    classes.json을 읽어 {class_id: {"type": str, "color": str, "label_key": str}} 사전으로 변환
    - 다양한 스키마(루트가 리스트이거나 {"classes": [...]} 형태)를 허용
    """
    p = Path(classes_json_path)
    assert p.exists(), f"classes.json이 존재하지 않습니다: {p}"


    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)


    # 허용 스키마 정규화
    if isinstance(data, dict) and "classes" in data:
        entries = data["classes"]
    elif isinstance(data, list):
        entries = data
    else:
        # {"0": {...}, "1": {...}} 와 같은 사전도 허용
        entries = list(data.values())


    id2meta: Dict[int, Dict[str, str]] = {}
    for e in entries:
        cid = int(e["class_id"]) if "class_id" in e else int(e.get("id", -1))
        typ = e.get("type", "unknown")
        color = e.get("color", "unknown")
        label_key = e.get("label_key", f"type={typ}|color={color}")
        id2meta[cid] = {"type": typ, "color": color, "label_key": label_key}
    return id2meta




def select_device(device_cfg: str = "auto") -> str:
    """
    cfg의 device 설정을 바탕으로 실제 사용 장치를 문자열로 반환("cpu", "cuda:0" 등)
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False


    if device_cfg == "auto":
        return "cuda:0" if has_cuda else "cpu"
    return device_cfg




def classes_count_from_meta(id2meta: Dict[int, Dict[str, str]]) -> int:
    # 클래스 개수 C 계산(최대 class_id + 1 기준)
    if not id2meta:
        return 0
    return max(id2meta.keys()) + 1




def type_to_category_map(path: str | Path) -> Dict[str, str]:
    cfg = load_yaml(path)
    mapping = cfg.get("type_to_category", {})
    # key를 소문자로 정규화
    return {str(k).lower(): str(v) for k, v in mapping.items()}


def color_en_to_kr(color_en: str) -> str:
    return COLOR_EN_TO_KR.get(color_en.lower(), color_en)