import yaml, torch, torch.optim as optim
from torch.utils.data import DataLoader
from yolov1.dataset import VOCDataset
from yolov1.model import YOLOv1
from yolov1.loss import YoloV1Loss
from yolov1.utils import set_seed

def adjust_lr(optimizer, base_lr, epoch, steps, gamma):
    """
    [Step Decay 스케줄]
    - 특정 epoch들(steps)에서 학습률을 gamma배로 감소시켜 수렴 안정화
    - YOLO v1 논문 재현에서 흔히 쓰이는 단순/견고한 스케줄
    """
    lr = base_lr
    for s in steps:
        if epoch >= s:
            lr *= gamma
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(42)

    # 1) 데이터 로더: 각 이미지 → (텐서, SxSx(C+5B) 타깃)
    train_ds = VOCDataset(cfg['data_root'], cfg['train_list'],
                          img_size=cfg['img_size'], S=cfg['S'], B=cfg['B'], C=cfg['C'], augment=True)
    dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # 2) 모델/손실/최적화기
    model = YOLOv1(S=cfg['S'], B=cfg['B'], C=cfg['C']).cuda()
    criterion = YoloV1Loss(S=cfg['S'], B=cfg['B'], C=cfg['C'],
                           lambda_coord=cfg['lambda_coord'], lambda_noobj=cfg['lambda_noobj'])
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'],
                          momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    global_step = 0
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        lr = adjust_lr(optimizer, cfg['lr'], epoch, cfg['lr_steps'], cfg['lr_gamma'])

        for imgs, targets in dl:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # 3) 순전파: 모든 셀/박스/클래스를 한 번에 예측
            preds = model(imgs)

            # 4) 손실: 책임박스/IoU/클래스 SSE를 한 번에 계산
            loss, parts = criterion(preds, targets)

            # 5) 역전파/최적화
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 6) 로깅: 각 항목의 기여도를 모니터링하여 튜닝 포인트 파악
            if global_step % 50 == 0:
                print(f"Epoch {epoch:03d} | step {global_step:06d} | "
                      f"loss={loss.item():.4f} | "
                      f"coord={parts['coord']:.3f} | "
                      f"conf_obj={parts['conf_obj']:.3f} | "
                      f"conf_noobj={parts['conf_noobj']:.3f} | "
                      f"cls={parts['cls']:.3f} | lr={lr:.2e}")
            global_step += 1

        # 7) 체크포인트 저장(나중에 mAP 평가/추론에 사용)
        torch.save({'model': model.state_dict(), 'epoch': epoch}, f'checkpoint_{epoch:03d}.pth')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/voc.yaml')
    args = ap.parse_args()
    main(args.cfg)
