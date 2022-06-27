# -*- coding: utf-8 -*-
# SyntaxError: Non-ASCII character '\xeb' 방지

from PIL import Image
import os,sys
from torchvision import transforms, datasets
from glob import glob
import argparse
import torch

# 라이브러리가 접근 가능하도록 path를 추가한다.
path = os.getcwd()  # 현재 위치 가져오기
sys.path.append('./EfficientNet-PyTorch')
# print(os.path.join(path,'EfficientNet-PyTorch'))
# sys.path.append(os.path.join(path,'EfficientNet-PyTorch'))
from efficientnet_pytorch import EfficientNet

def test(img, weights):  # 이미지와 학습된 모델을 인자로 받는다.

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # cpu or gpu 설정
    pretrain_model = 'efficientnet-b0'  # 사전 학습된 모델 이름

    # 학습된 모델 가져오기
    print(device)
    if device == 'cuda:0':
        model = EfficientNet.from_pretrained(pretrain_model, num_classes=3)
        model.load_state_dict(torch.load(weights))
    else:
        model = EfficientNet.from_pretrained(pretrain_model, num_classes=3)
        model.load_state_dict(torch.load(weights, map_location=device))

    # 모델의 계산 device 설정
    model.to(device)
    model.eval() # 평가모드이므로 역전파는 수행하지 않는다.
    
    # 이미지 전처리
    # IMG = Image.open(img).convert('RGB')

    inputs = Image.fromarray(img).convert('RGB')  # array를 PIL 형식으로 변경한다.
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    inputs = transform(inputs)  # 사이즈 변환 및 정규화
    inputs = inputs.reshape((1,) + inputs.shape)  # 3차원을 4차원으로 변경
    inputs = inputs.to(device)

    # 예측 -> 0 : disposable, 1 : no_recycle, 2 : recycle
    outputs = model(inputs)  # 0번째 인덱스는 disposable , 1번째 인덱스는 no_recycle, 2번째 인덱스는 recycle이다.
    _, preds = torch.max(outputs, 1)

    return preds

