import torch
import torch.nn as nn

class YoloV1Loss(nn.Module):
    """
    [YOLO v1 손실 설계 철학]
    - 단일 SSE(평균제곱오차) 프레임 안에 '좌표/크기/존재성/클래스'를 모두 포함
    - 중요한 박스 좌표 정확도에 높은 가중치(λ_coord=5)
    - 객체가 없는 셀의 존재성(confidence)에는 낮은 가중치(λ_noobj=0.5)
      -> 배경이 훨씬 많으므로 '배경의 학습 신호'가 학습을 망치지 않게 완화
    - 각 셀에서 '책임 박스'(responsible box) 1개만 좌표/존재성 로스에 참여
      -> IoU가 가장 큰 박스를 자동으로 선택(학습 중 동적으로 결정)
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lc, self.lno = lambda_coord, lambda_noobj

    @staticmethod
    def bbox_iou_xyxy(a, b, eps=1e-9):
        """
        축 기준 박스(x1,y1,x2,y2) IoU 계산.
        - 교집합/합집합 비율로 두 박스의 겹침 정도를 정량화
        - YOLO v1의 confidence 학습 타깃이 IoU이므로 매우 중요
        """
        tl = torch.max(a[..., :2], b[..., :2])
        br = torch.min(a[..., 2:], b[..., 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[...,0]*wh[...,1]
        area_a = (a[...,2]-a[...,0]).clamp(min=0) * (a[...,3]-a[...,1]).clamp(min=0)
        area_b = (b[...,2]-b[...,0]).clamp(min=0) * (b[...,3]-b[...,1]).clamp(min=0)
        return inter / (area_a + area_b - inter + eps)

    def forward(self, pred, target):
        """
        pred, target: (N, S, S, C+5B)
        - pred는 네트워크 예측, target은 데이터셋에서 인코딩한 라벨
        - 큰 흐름:
          1) pred/target을 (cls, boxes[B], conf[B])로 분리
          2) 셀 좌표계를 이미지 좌표계(xyxy)로 복원해 IoU 계산
          3) IoU가 최대인 박스를 '책임박스'로 선택(resp_mask)
          4) 좌표/크기, 존재성(객체/비객체), 클래스에 대해 MSE 합산
        """
        N, S, B, C = pred.size(0), self.S, self.B, self.C
        device = pred.device

        # 1) 예측 분해
        pred_cls = pred[..., :C]
        pred_boxes = []
        pred_conf  = []
        for b in range(B):
            base = C + b*5
            pred_boxes.append(pred[..., base:base+4])       # (x, y, sqrt(w), sqrt(h))
            pred_conf.append(pred[..., base+4:base+5])      # confidence
        pred_boxes = torch.stack(pred_boxes, dim=3)         # (N,S,S,B,4)
        pred_conf  = torch.cat(pred_conf, dim=-1)           # (N,S,S,B)

        # 2) 타깃 분해
        tgt_cls = target[..., :C]
        tgt_boxes = []
        tgt_conf  = []
        for b in range(B):
            base = C + b*5
            tgt_boxes.append(target[..., base:base+4])
            tgt_conf.append(target[..., base+4:base+5])
        tgt_boxes = torch.stack(tgt_boxes, dim=3)           # (N,S,S,B,4)
        tgt_conf  = torch.cat(tgt_conf, dim=-1)             # (N,S,S,B)

        # 3) 셀 좌표계 -> 이미지 상대좌표계로 복원 (IoU 계산을 위해)
        cell_size = 1.0 / S
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device), torch.arange(S, device=device), indexing='ij'
        )
        gx = grid_x[None, ..., None] * cell_size   # 각 셀의 좌상단 x좌표(상대)
        gy = grid_y[None, ..., None] * cell_size   # 각 셀의 좌상단 y좌표(상대)

        # (x,y): 셀 내부 상대 -> 전체 상대좌표로 변환
        px = (pred_boxes[...,0] + gx)
        py = (pred_boxes[...,1] + gy)
        # (w,h): sqrt로 예측 → 제곱으로 복원 (0 미만 방지)
        pw = pred_boxes[...,2].clamp(min=0)**2
        ph = pred_boxes[...,3].clamp(min=0)**2
        # xywh -> xyxy
        pxyxy = torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2], dim=-1)

        tx = (tgt_boxes[...,0] + gx)
        ty = (tgt_boxes[...,1] + gy)
        tw = tgt_boxes[...,2].clamp(min=0)**2
        th = tgt_boxes[...,3].clamp(min=0)**2
        txyxy = torch.stack([tx - tw/2, ty - th/2, tx + tw/2, ty + th/2], dim=-1)

        # 4) IoU 계산 및 책임박스 선택
        ious = self.bbox_iou_xyxy(pxyxy, txyxy)             # (N,S,S,B)
        obj_mask = (tgt_conf.max(dim=-1, keepdim=True).values > 0).float()  # 해당 셀에 객체가 있는가?
        best_iou, best_idx = ious.max(dim=-1, keepdim=True) # IoU 최대 박스 인덱스
        resp_mask = torch.zeros_like(ious)                  # 책임박스 1-hot
        resp_mask.scatter_(-1, best_idx, 1.0)
        resp_mask = resp_mask * obj_mask                    # 객체 있는 셀에서만 유효

        # 5) 좌표/크기 손실 (책임박스만)
        def gather_b(t):  # t: (N,S,S,B,d) -> 책임박스만 모아 (N,S,S,d)
            return (t * resp_mask[...,None]).sum(dim=3)

        p_box = gather_b(pred_boxes)
        t_box = gather_b(tgt_boxes)

        # (x,y): MSE
        coord_loss = ((p_box[...,0:2] - t_box[...,0:2])**2).sum()

        # (w,h): sqrt 공간에서 MSE (논문 설계)
        coord_loss += (
            (torch.sign(p_box[...,2:4]) * torch.sqrt(p_box[...,2:4].clamp(1e-6))
           - torch.sign(t_box[...,2:4]) * torch.sqrt(t_box[...,2:4].clamp(1e-6)))**2
        ).sum()
        coord_loss *= self.lc  # λcoord 가중

        # 6) confidence 손실
        # 6-1) 객체 있는 셀: 책임박스의 conf를 IoU에 맞추도록 회귀
        p_conf_obj = (pred_conf * resp_mask).sum(dim=-1, keepdim=True)  # (N,S,S,1)
        conf_obj_loss = ((p_conf_obj - best_iou[...,None])**2 * obj_mask).sum()

        # 6-2) 객체 없는 셀: 모든 B 박스 conf는 0으로 가깝게 (λnoobj로 완화)
        noobj_mask = 1.0 - obj_mask
        conf_noobj_loss = (((pred_conf) - 0.0)**2 * noobj_mask).sum() * self.lno

        # 7) 클래스 손실(객체 있는 셀만)
        cls_loss = (((pred_cls - tgt_cls)**2) * obj_mask).sum()

        # 배치 평균
        loss = (coord_loss + conf_obj_loss + conf_noobj_loss + cls_loss) / pred.size(0)

        return loss, dict(
            coord=coord_loss/pred.size(0),
            conf_obj=conf_obj_loss/pred.size(0),
            conf_noobj=conf_noobj_loss/pred.size(0),
            cls=cls_loss/pred.size(0)
        )
