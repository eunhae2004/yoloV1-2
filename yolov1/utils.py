import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def decode_predictions(output, S=7, B=2, C=20, conf_thres=0.1):
    """
    [네트워크 출력 -> 실제 박스/점수/클래스 리스트로 디코딩]
    - 각 셀의 B박스에 대해 (x,y,w,h,conf)와 클래스 확률(softmax/여기선 그대로) 이용
    - 클래스 점수 = Pr(class|object) * confidence
    - (x,y)는 셀 기준 상대좌표 -> 전체 상대좌표로 변환
    - (w,h)는 sqrt 예측 -> 제곱 복원
    - conf_thres로 너무 약한 박스는 초기에 제거(성능/속도 ↑)
    반환: [ (x1,y1,x2,y2, score, cls_idx), ... ]  (상대좌표 0~1)
    """
    N = output.size(0)
    assert N == 1, "단일 이미지 추론 기준의 간단 디코더 (배치는 외부 루프에서 처리)"
    out = output[0]  # (S,S,C+5B)
    cell_size = 1.0 / S

    cls_pred = out[..., :C]             # (S,S,C)
    boxes = []
    for b in range(B):
        base = C + 5*b
        xb = out[..., base+0]
        yb = out[..., base+1]
        wb = out[..., base+2].clamp(min=0)**2
        hb = out[..., base+3].clamp(min=0)**2
        cb = out[..., base+4]           # confidence

        for j in range(S):
            for i in range(S):
                conf = cb[j,i].item()
                if conf < conf_thres:
                    continue
                # 셀 좌상단 기준 -> 전체 상대좌표
                cx = (i + xb[j,i].item()) * cell_size
                cy = (j + yb[j,i].item()) * cell_size
                w  = wb[j,i].item()
                h  = hb[j,i].item()

                x1 = cx - w/2; y1 = cy - h/2
                x2 = cx + w/2; y2 = cy + h/2

                # 클래스 점수: Pr(class|object) * confidence
                class_scores = cls_pred[j,i].detach().cpu().numpy() * conf
                cls_idx = int(class_scores.argmax())
                score   = float(class_scores[cls_idx])

                boxes.append([x1,y1,x2,y2, score, cls_idx])

    return boxes

def nms(boxes, iou_thres=0.45):
    """
    [비최대 억제(Non-Maximum Suppression)]
    - 같은 객체를 중복 검출한 박스들 중 점수가 가장 높은 것만 남기고 제거
    - 클래스별로 분리하여 NMS 수행하는 것이 일반적
    """
    if not boxes:
        return []
    # 클래스별로 그룹
    by_cls = {}
    for b in boxes:
        by_cls.setdefault(b[5], []).append(b)

    kept = []
    for cls, group in by_cls.items():
        # 점수 내림차순
        group.sort(key=lambda x: x[4], reverse=True)
        used = [False]*len(group)

        for i in range(len(group)):
            if used[i]: 
                continue
            kept.append(group[i])
            xi1, yi1, xi2, yi2, si, _ = group[i]
            ai = max(0.0, xi2-xi1) * max(0.0, yi2-yi1)

            for j in range(i+1, len(group)):
                if used[j]: 
                    continue
                xj1, yj1, xj2, yj2, sj, _ = group[j]

                # IoU 계산
                xx1 = max(xi1, xj1); yy1 = max(yi1, yj1)
                xx2 = min(xi2, xj2); yy2 = min(yi2, yj2)
                w = max(0.0, xx2-xx1); h = max(0.0, yy2-yy1)
                inter = w*h
                aj = max(0.0, xj2-xj1) * max(0.0, yj2-yj1)
                iou = inter / (ai + aj - inter + 1e-9)

                if iou >= iou_thres:
                    used[j] = True
    return kept
