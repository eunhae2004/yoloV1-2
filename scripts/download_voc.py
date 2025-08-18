# 필요 모듈 다운로드

import os, tarfile, urllib.request, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
VOC = DATA/'VOCdevkit'
VOC.mkdir(parents=True, exist_ok=True)

# Pascal VOC 2007, 2012 데이터셋 다운로드
# Pascal VOC dataset: Object Detection용 Dataset
# person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motobike, train, bottle, chair, dining, table, potted, plant, sofa, tv/monitor
URLS = [
  ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', 'VOCtrainval_06-Nov-2007.tar'),
  ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',     'VOCtest_06-Nov-2007.tar'),
  ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar', 'VOCtrainval_11-May-2012.tar'),
]

def download(url, dst):
    if os.path.exists(dst): return
    print(f'Downloading {url}')
    urllib.request.urlretrieve(url, dst)

def extract(tar_path, dst):
    print(f'Extract {tar_path}')
    with tarfile.open(tar_path) as t:
        t.extractall(dst)

if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    for url, name in URLS:
        fp = DATA/name
        download(url, fp)
        extract(fp, DATA)
        