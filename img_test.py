# -*- coding: utf-8 -*-
# SyntaxError: Non-ASCII character '\xeb' 방지 

# img를 input으로 넣어 결과값 출력해보는 py파일
from PIL import Image
import os,sys
from torchvision import transforms, datasets
from glob import glob
import argparse
import torch

# 라이브러리가 접근 가능하도록 path를 추가한다.
sys.path.append('./EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet

# 디렉토리 만드는 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# 파라미터로 img 파일과 저장된 모델 경로를 가져온다.
# num 은 반복문을 돌리 때 인덱스 번호가 된다.
def test(img, num, model, device):

    model.to(device)
    model.eval()

    IMG = Image.open(img).convert('RGB')
    inputs = Image.open(img).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    
    inputs = transform(inputs)
    print(type(inputs))
    print(inputs.shape)
    inputs = inputs.reshape((1,) + inputs.shape)  # 3차원을 4차원으로 변경
    inputs = inputs.to(device)

    # 예측값
    outputs = model(inputs)  # 0번째 인덱스는 disposable , 1번째 인덱스는 no_recycle, 2번째 인덱스는 recycle이다.
    _, preds = torch.max(outputs, 1)

    createFolder('./results')
    
    count_0, count_1, count_2  = 0 , 0, 0  # 초기화 - 0:disposable , 1:no_recycle, 2:recycle 분류된 클래스가 1로 변경
    if preds == 0:
        print('{} 번째 이미지  :  disposable !!'.format(num))
        #IMG.save('./results/can_{}.jpg'.format(num))
        count_0 = 1
    elif preds == 1:
        print('{} 번째 이미지  :  no_recycle !!'.format(num))
        #IMG.save('./results/can_{}.jpg'.format(num))
        count_1 = 1
    else:
        print('{} 번째 이미지  :  recycle !!'.format(num))
        #IMG.save('./results/pet_{}.jpg'.format(num))
        count_2 = 1

    return count_0, count_1, count_2

# img에는 이미지들이 있는 폴더 경로를 넣는다.
def run(img, weights):
    img_list = glob(os.path.join(img, '*'))
    model_name = 'efficientnet-b0'  # pre-trained 모델 이름 
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model_name = 'efficientnet-b0'  # b5
    print(device)
    if device == 'cuda:0':
        model = EfficientNet.from_pretrained(model_name, num_classes=3)
        model.load_state_dict(torch.load(weights))
    else:
        model = EfficientNet.from_pretrained(model_name, num_classes=3)
        model.load_state_dict(torch.load(weights, map_location=device))

    disposable_count = 0
    norecycle_count = 0
    recycle_count = 0
    

    for num, img in enumerate(img_list):
        count_0, count_1, count_2 = test(img, num, model, device)
        disposable_count += count_0
        norecycle_count += count_1
        recycle_count += count_2
    print('전체 이미지 개수 : {}'.format(len(img_list)))
    print('disposable으로 분류한 이미지 개수 : {}'.format(disposable_count))
    print('no_recycle으로 분류한 이미지 개수 : {}'.format(norecycle_count))
    print('recycle으로 분류한 이미지 개수 : {}'.format(recycle_count))

def parse_opt():
    parser = argparse.ArgumentParser(description="efficientnet으로 새로운 샘플 테스트 해보기...")
    parser.add_argument('--img', type=str, required=True, help='테스트할 이미지가 있는 폴더 위치')
    parser.add_argument('--weights', type=str, required=True, help='저장된 pt파일 불러오기')
    #parser.add_argument('--model_name', type=str, required=True, help='pre-trained 모델 이름 - ex) efficientnet-b0')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
