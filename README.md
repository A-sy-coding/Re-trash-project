# Webcam을 이용한 pet/can 분류 및 pet 재활용 여부 분류 

</br>

* **os환경**

	* ubuntu 18.04 
	
	* Window 64bit

--------------

> model을 학습할 때에는 ubuntu 18.04 환경을 이용한다.

> Webcam을 활용해 test 할 때에는 Window 환경을 이용한다.

--------------

</br>

## 1. venv 가상환경 만들기

</br>

> **먼저, venv를 이용하여 가상환경을 생성한다.**

```
> python -m venv env
```

> **가상환경 활성화**

```
> source ./env/bin/activate   # ubuntu 가상환경 접속 방법

> ./env/Scripts/activate.bat     # Window 가상환경 접속 방법
```

</br>

## 2. GitHub 에서 파일 clone 하기

</br>

> Git에서 master branch에 있는 파일들을 clone 해온다.

```
(env) > git clone https://github.com/A-sy-coding/Re-trash-project.git
```

</br>

## 3. 필요한 패키지 설치

</br>

```
(env) > ./default_setting.sh  # EfficientNet 파일 clone 및 requirements.txt 파일 실행
(env) > ./ubuntu_setting.sh  # torch gpu를 사용하면 실행해야 된다.
```

	* 이때 sh파일이 실행 권한이 없다면 sudo chmod 755 [file_name].sh를 통해 실행권한을 준다.

</br>

## 4. 모델 학습하기

</br>

> train.py 파일을 실행하여 EfficientNet 모델을 학습한다.

```
python train.py --batch 64 --model_name "efficientnet-b0" --epochs 5 --name "epochs5" --path '../aihub_pet_can_img/Validation/pet_img' --path1 '../aihub_pet_can_img/Training/pet_img/일회용' --setting
```

	* --batch : 한번에 데이터를 처리할 개수 
	* --model_name : pre-trained된 모델 이름
	* --epochs : 반복 횟수
	* --name : 학습을 완료한 모델을 저장할 이름 설정
	* --path : 일회용 이미지가 있는 폴더 경로 --> 실제 사용할 때는 경로를 수정해야 한다.
	* --path1 : 추가 일회용 이미지가 있는 폴더 경로 --> 실제 사용할 때는 경로를 수정해야 한다.
	* --setting : 이미지 파일들이 사전 준비되어 있으면 생략하고 사전 준비되어 있지 않으면 입력한다.

</br>


## 5. Webcam을 활용하여 pet 재활용 여부 분류( 선택 사항)

> webcam.py파일은 학습시킨 모델을 활용하여 Webcam에서 보여지는 pet 이미지가 재활용이 가능한지를 화면에 출력해주는 역할을 수행한다.

> server에서는 카메라가 존재하지 않기 때문에 Webcam을 활용할 때에는 로컬환경에서 수행하도록 한다.

```
(env) > python webcam.py --weights "./save_model/epochs_5.pt"
```

	* --weights : 학습시킨 모델 - pt파일

</br>


## 6. Yolov5 + EfficientNet을 활용해 Webcam 인식

> Yolov5 + EfficientNet 모델을 합쳐서 Webcam이 객체를 인식하면 해당 객체가 pet/can인지 분류하고, 추가적으로 pet의 경우에는 재활용이 가능한지 판단하도록 구현하였다.

> yolov5/detect.py 파일을 실행시켜 pet/can 분류뿐만 아니라 pet 재활용 여부도 판단할 수 있게 된다.

> server에서는 카메라가 존재하지 않기 때문에 Webcam 사용시 로컬환경에서 실행시켜야 한다.

> Yolov5 모델은 이미 학습되어 있는 가중치 파일이 존재한다.

```
(env) python detect.py --source 0 --weights [학습된 yolo pt파일] --conf 0.5 --save-crop
```

	* --source : 0이면 webcam을 의미한다.
	* --weights : 학습된 yolov5 pt 파일을 의미한다.
	* --conf : 객체를 인식할 때의 확률 threshold 설정
	* --save-crop : EfficientNet을 적용하기 위해서 설정해야될 옵션


