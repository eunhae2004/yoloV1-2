# YOLO v1 의류 검출 (Detection + E2E 파이프라인)

- 목적: YOLO v1 기반 의류 검출. 검출 박스 → 원본 크롭 → (옵션) 분류 모델 연계.
- 색상 판정: **별도 알고리즘 없음**. `classes.json`의 `color` 필드 그대로 사용.

## 환경

- Python ≥ 3.10, PyTorch ≥ 2.0
- OS: Windows (기본 안내), 다른 OS도 경로만 조정 시 사용 가능

## 설치

```bash
cd C:\YoloV1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
