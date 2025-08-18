import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

VOC_CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
"cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep",
"sofa","train","tvmonitor"]
CLASS_TO_IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

class VOCDataset(torch.utils.data.Dataset):
    """
    [YOLO v1의 라벨 인코딩 핵심]
    - 이미지를 SxS 셀로 나눈다.
    - 각 GT 박스의 중심점이 속한 '딱 한 셀'이 그 객체를 '책임'진다(responsible cell).
    - 그 셀의 타깃 텐서에:
        * 클래스 원-핫 벡터(Pr(class|object))를 기록
        * B개의 박스 중 하나에 (x, y, sqrt(w), sqrt(h), objectness=1) 기록
      (학습 시에는 'IoU가 가장 높은 박스'가 책임박스로 선택되어 좌표/존재성 로스를 받음)
    """
    def __init__(self, root, id_list_file, img_size=448, S=7, B=2, C=20, augment=True):
        self.root = Path(root)
        self.ids = [x.strip() for x in open(id_list_file)]
        self.img_size, self.S, self.B, self.C = img_size, S, B, C
        self.augment = augment

        # 입력 정규화: [-1, 1] 범위로 맞추어 학습 안정화
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self): return len(self.ids)

    def _load_anno(self, vid):
        """
        VOC 어노테이션(XML)에서 객체 박스를 읽어옵니다.
        반환: 이미지(RGB), 박스(xmin,ymin,xmax,ymax), 클래스 인덱스, 원본(H,W)
        """
        year, stem = vid.split('/')
        img = cv2.imread(str(self.root/year/'JPEGImages'/f'{stem}.jpg'))[..., ::-1]  # BGR->RGB
        h, w = img.shape[:2]
        xml = ET.parse(str(self.root/year/'Annotations'/f'{stem}.xml')).getroot()
        boxes, labels = [], []
        for obj in xml.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASS_TO_IDX: 
                continue
            bb = obj.find('bndbox')
            xmin = float(bb.find('xmin').text); ymin = float(bb.find('ymin').text)
            xmax = float(bb.find('xmax').text); ymax = float(bb.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[cls])
        return img, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), (h,w)

    def _resize(self, img, boxes):
        """
        YOLO v1은 고정 입력(보통 448x448)을 사용.
        - 이미지 리사이즈하면서 박스 좌표도 같은 비율로 스케일링.
        """
        H, W = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        scale_x, scale_y = self.img_size/W, self.img_size/H
        boxes = boxes.copy()
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
        return img, boxes

    def _encode(self, boxes, labels):
        """
        SxSx(C+5B) 타깃 텐서 생성:
        - 각 GT의 중심(cx,cy)가 속한 셀(i,j)을 찾아서 그 위치에 정보 기록
        - (x,y): 셀 내부 상대좌표(0~1)
        - (w,h): 이미지 대비 비율을 sqrt로 변환해 기록
        - objectness(=1)로 '해당 셀에 객체가 있다'를 표시
        - 클래스 원-핫도 같은 (i,j)에 기록
        """
        S, B, C = self.S, self.B, self.C
        target = np.zeros((S, S, C + 5*B), dtype=np.float32)
        cell = self.img_size / S

        for b,(xmin,ymin,xmax,ymax) in enumerate(boxes):
            cx = (xmin + xmax)/2.0
            cy = (ymin + ymax)/2.0
            w  = (xmax - xmin)
            h  = (ymax - ymin)

            i = int(cx / cell)      # x축으로 몇 번째 셀
            j = int(cy / cell)      # y축으로 몇 번째 셀
            i = min(max(i,0), S-1)  # 경계 안전
            j = min(max(j,0), S-1)

            # 셀 내부 상대좌표(0~1): 셀의 좌상단 기준
            x_cell = (cx / cell) - i
            y_cell = (cy / cell) - j
            # 폭/높이는 이미지 비율로 정규화 후 sqrt 적용(큰/작은 박스 균형)
            w_sqrt = np.sqrt(max(w / self.img_size, 1e-6))
            h_sqrt = np.sqrt(max(h / self.img_size, 1e-6))

            cls = labels[b]

            # B개의 박스 자리에 채우되, 학습 시에는 IoU로 '책임 박스'를 고르므로
            # 여기서는 비어있는 슬롯(=objectness 0)을 우선 채워 놓기만 하면 충분
            for bb in range(B):
                base = C + bb*5
                if target[j,i, base+4] == 0:      # 아직 비어있으면
                    target[j,i, base+0] = x_cell
                    target[j,i, base+1] = y_cell
                    target[j,i, base+2] = w_sqrt
                    target[j,i, base+3] = h_sqrt
                    target[j,i, base+4] = 1.0     # objectness
                    break

            # 클래스 원-핫 (객체가 있는 셀에만 의미 있음)
            target[j,i, cls] = 1.0

        return target

    def __getitem__(self, idx):
        vid = self.ids[idx]
        img, boxes, labels, _ = self._load_anno(vid)

        # 객체가 전혀 없는 이미지도 존재할 수 있음 → 타깃은 전부 0
        if boxes.size == 0:
            img = cv2.resize(img, (self.img_size, self.img_size))
            target = np.zeros((self.S,self.S,self.C+5*self.B), dtype=np.float32)
            return self.to_tensor(img), torch.from_numpy(target)

        img, boxes = self._resize(img, boxes)

        # 간단한 증강: 좌우 반전 (YOLO v1은 간단 증강만으로도 성능 개선 가능)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            # 좌우 반전 시 xmin/xmax 뒤집기
            boxes[:, [0,2]] = self.img_size - boxes[:, [2,0]]

        target = self._encode(boxes, labels)
        return self.to_tensor(img), torch.from_numpy(target)
