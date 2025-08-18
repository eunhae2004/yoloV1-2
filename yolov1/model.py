import torch
import torch.nn as nn

def conv_bn_lrelu(c_in: int, c_out: int, k: int, s:int = 1, p: int = 0):
    # [Con2d -> BatchNorm2d -> LeakyReLU] 기본 블록 생성
    # Yolov1은 BN을 명시하지 않았지만, 일반적으로 BN을 사용
    # LeakReLU의 negative slope는 0.1로 설정
    # K(커널), s(스트라이드), p(패딩)
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )

class YOLOv1(nn.Module):
    # YOLO v1 모델의 전체 네트워크
    # 출력 텐서 형태: (N, S, S, C + 5B)
    #   - S: 그리드 분할 수(논문 기본 7)
    #   - B: 각 셀에서 예측하는 박스 수(논문 기본 2)
    #   - C: 클래스 수(VOC는 20)
    #   - 5B는 각 박스별 (x, y, w, h, confidence)
    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        # 특징 추출부 (24개의 컨볼루션 근사)
        # 논문 구조를 따르되, 채널 수/블록 배치는 Darknet v1 레퍼런스를 참고
        # MaxPool로 해상도를 점진적으로 줄이며 receptive field 확장

        self.features = nn.Sequential(
            # 입력 448x448x3 -> Conv7x7 s=2 로 큰 receptive field 확보 후 다운샘플링
            conv_bn_lrelu(3, 64, k=7, s=2, p=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 중간 깊이 확장
            conv_bn_lrelu(64, 192, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 1x1과 3x3을 번갈아 사용하여 채널 혼합과 국소 특징 추출
            conv_bn_lrelu(192, 128, k=1),
            conv_bn_lrelu(128, 256, k=3, s=1, p=1),
            conv_bn_lrelu(256, 256, k=1),
            conv_bn_lrelu(256, 512, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 반복 블록 ×4 (1x1 -> 3x3)
            conv_bn_lrelu(512, 256, k=1),
            conv_bn_lrelu(256, 512, k=3, s=1, p=1),
            conv_bn_lrelu(512, 256, k=1),
            conv_bn_lrelu(256, 512, k=3, s=1, p=1),
            conv_bn_lrelu(512, 256, k=1),
            conv_bn_lrelu(256, 512, k=3, s=1, p=1),
            conv_bn_lrelu(512, 256, k=1),
            conv_bn_lrelu(256, 512, k=3, s=1, p=1),

            # 채널 확장으로 더 풍부한 표현 확보
            conv_bn_lrelu(512, 512, k=1),
            conv_bn_lrelu(512, 1024, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 반복 블록 ×2
            conv_bn_lrelu(1024, 512, k=1),
            conv_bn_lrelu(512, 1024, k=3, s=1, p=1),
            conv_bn_lrelu(1024, 512, k=1),
            conv_bn_lrelu(512, 1024, k=3, s=1, p=1),

            # 해상도/특징 심화
            conv_bn_lrelu(1024, 1024, k=3, s=1, p=1),
            conv_bn_lrelu(1024, 1024, k=3, s=2, p=1),  # s=2로 다운샘플링

            conv_bn_lrelu(1024, 1024, k=3, s=1, p=1),
            conv_bn_lrelu(1024, 1024, k=3, s=1, p=1),
        )

        # 7x7 feature map을 펼쳐 FC로 전역 정보를 결합
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),  # 논문: FC 4096
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),                 # 과적합 방지
            nn.Linear(4096, S * S * (C + B * 5)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        # 최종 출력은 (N, S, S, C+5B)
        return x.view(-1, self.S, self.S, self.C + self.B * 5)