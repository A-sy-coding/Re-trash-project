import shutil  # 파이썬에서 파일 이동할 때 필요한 라이브러리
import torch
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import time
import os, sys
import copy
import random
import argparse
from tqdm import tqdm
from glob import glob

from data_merge import file_merge  # 데이터 합치는 함수 불러오기
from data_split import train_val_split # 데이터를 훈련,검증 데이터로 나누기


# OSError: image file is truncated 해결방안
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 라이브러리가 접근 가능하도록 path를 추가한다.
sys.path.append('./EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet  # 경로 한번 확인
from torch.utils.tensorboard import SummaryWriter


# 디렉토리 만드는 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# aihub에 있는 이미지들을 원하는 경로에 복사하는 함수
def copy_file(img_list, img_destination):
  for f in img_list:
    shutil.copy(f, img_destination)


# 이미지 전처리 함수 구현 --> resize, class생성 등등

# folder_path에는 클래스를 생성하고 하는 폴더들의 최상위 위치를 설정한다.
def preprocessing(train_folder_path, val_folder_path, batch_size):
    # 폴더 안의 디렉토리들의 이름으로 클래스 생성 및 transformers를 이용해 데이터 전처리
    train_dataset = datasets.ImageFolder(train_folder_path,
                                         transforms.Compose([
                                             transforms.Resize((224, 224)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]), ]))

    val_dataset = datasets.ImageFolder(val_folder_path,
                                       transforms.Compose([
                                           transforms.Resize((224, 224)), transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]), ]))

    ## data loader 선언
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])
    print('batch_size : %d,  tvt : %d / %d ' % (batch_size, batch_num['train'], batch_num['valid']))

    return dataloaders


# epoch 돌리는 함수 구현
def model_train(model, dataloader, criterion, optimizer, scheduler, device, num_epochs, name):
    # 학습시간 구하기
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    test_pred = []

    #writer = SummaryWriter('./runs/train/')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # train과 valid일 때를 구분한다.
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # train일 때는 학습하도록 설정
            else:
                model.eval()  # valid일 때는 평가하도록 설정

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # 지정한 배치 사이즈만큼 데이터를 계속 가져오기
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # 훈련 범위 지정
                with torch.set_grad_enabled(phase == 'train'):

                    # 순전파
                    outputs = model(inputs)

                    # preds = (outputs<0.5).float()
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            # tensorboard 그리기
            #writer.add_graph('epoch loss', epoch_loss, epoch)
            #writer.add_graph('epoch acc', epoch_acc, epoch)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # valid 데이터를 가지고 bset model copy하기
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    end = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # 모델 저장하기
    createFolder('./save_model')  # 모델을 저장할 폴더 생성
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './save_model/{}.pt'.format(name))
    print('model saved')

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

# 훈련시키는 함수 구현
def train(dataloader, model_name, num_epochs, name):

  image_size = EfficientNet.get_image_size(model_name)
  print('pre-trained efficientnet의 이미지 크기 : ',image_size)

  # model_name = 'efficientnet-b0'  # b5
  model = EfficientNet.from_pretrained(model_name, num_classes=3) # 일회용, 재활용, 재활용 불가능 3개의 클래스 분

  # 사용 device 지정
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer_ft = optim.SGD(model.parameters(), lr = 0.05, momentum=0.9, weight_decay=1e-4)

  lmbda = lambda epoch: 0.98739
  exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

  # 모델 학습
  model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = model_train(model, dataloader, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs, name)
  
  return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


# base_dir는 aihub 데이터 셋이 있는 폴더의 경로
# model_name은 사용할 모델 이름
# name은 학습시킨 weight를 저장할 이름 설정

def run(batch, model_name, epochs, name, path, path1, setting):

    # 폴더들이 세팅되어 있으면 바로 dataloader로 이동하고, 폴더들이 세팅되어있지 않으면 세팅시킨 후 dataloader한다.
    if setting:
        # 저장할 디렉토리 생성
        createFolder('./train_img/disposable')
        createFolder('./train_img/recycle')
        createFolder('./train_img/no_recycle')

        createFolder('./val_img/disposable')
        createFolder('./val_img/recycle')
        createFolder('./val_img/no_recycle')

        # 훈련데이터와 검증데이터 이미지들의 경로 리스트
        disposable, recycle, no_recycle = file_merge(path, path1) # 각 클래스 이미지 경로 가져오기
        train_disposable, val_disposable = train_val_split(disposable)  # 일회용 데이터 분할
        train_recycle, val_recycle = train_val_split(recycle)  # 재활용 데이터 분할
        train_norecycle, val_norecycle = train_val_split(no_recycle) # 재활용 불가능 데이터 분할
        
        
        img_lists = [train_disposable, val_disposable, train_recycle, val_recycle, train_norecycle, val_norecycle]
        destinations = ['./train_img/disposable', './val_img/disposable', './train_img/recycle', './val_img/recycle','./train_img/no_recycle' , './val_img/no_recycle']  # 저장 위치

        # 위에서 만든 폴더로 이미지들이 복사되어 이동한다.
        print('----------------')
        print('image copy...')
        copy_start = time.time()
        for img_list, destination in tqdm(zip(img_lists, destinations)):
            #print(img_list, destination)
            copy_file(img_list, destination)  # 이미지 전체 경로들 리스트와 이미지 저장 경로
        copy_finish = time.time() - copy_start
        print('image copy complete in {:.0f}m {:.0f}s'.format(copy_finish // 60, copy_finish % 60))
        print('----------------')

    # 이미지 전처리 함수 사용
    dataloader = preprocessing('./train_img', './val_img', batch)

    # 모델 학습
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train(dataloader, model_name, epochs, name)
    print('best epochs : ', best_idx)
    print('best acc : ', best_acc)
    print('train_loss : {} , train_acc : {} , valid_loss : {} , valid_acc : {}'.format(train_loss, train_acc, valid_loss, valid_acc))

def parse_opt():
    parser = argparse.ArgumentParser(description="efficientnet을 이용한 disposable, recycle, no_recycle 데이터 학습...")
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--model_name', type=str, required=True, help='pre-trained 모델 이름 - ex) efficientnet-b0')
    parser.add_argument('--epochs', type=int, required=True, help='epoch 설정')
    parser.add_argument('--name', type=str, required=True, help='학습시킨 모델을 저장할 때 지정할 파일이름')
    parser.add_argument('--path', type=str, required=True, help='이미지 파일이 존재하는 경로-1')
    parser.add_argument('--path1', type=str, required=True, help='이미지 파일이 존재하는 경로-2')
    # parser.add_argument('--setting', type=str2bool, default = "True" , help='전처리를 하기 전 이미지 파일 폴더 세팅 확인 - 세팅 되어 있으면 False로 변경')
    parser.add_argument('--setting', action='store_true',  help='전처리를 하기 전 이미지 파일 폴더 세팅 확인 - 세팅 되어 있으면 인자 무시')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
