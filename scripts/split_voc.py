import pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOC07 = ROOT/'data'/'VOCdevkit'/'VOC2007'
VOC12 = ROOT/'data'/'VOCdevkit'/'VOC2012'
OUT = ROOT/'data'/'cache'
OUT.mkdir(parents=True, exist_ok=True)

def read_ids(txt): return[x.strip() for x in open(txt)]

# 분할 리스트 생성 
if __name__ == '__main__':
    ids07_trainval = read_ids(VOC07/'ImageSets'/'Main'/'trainval.txt')
    ids12_trainval = read_ids(VOC12/'ImageSets'/'Main'/'trainval.txt')
    ids_train = [f'VOC2007/{i}' for i in ids07_trainval] + [f'VOC2012/{i}' for i in ids12_trainval]
    random.shuffle(ids_train)
    open(OUT/'train.txt', 'w').write('\n'.join(ids_train))

    ids07_test = read_ids(VOC07/'ImageSets'/'Main'/'test.txt')
    open(OUT/'test07.txt', 'w').write('\n'.join([f'VOC2007/{i}' for i in ids07_test]))