import cv2
import argparse
from predict import test

# 이미지에 텍스트를 출력하는 함수
def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color=(255, 0, 0)
    text_color_bg=(0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

def run(weights):
    cap = cv2.VideoCapture(0) # 웹캠 연결
    class_names = ['disposable', 'no_recycle', 'recycle']
    
    while(True):
        ret, cam = cap.read()
    
        if ret:
            preds = test(cam, weights)  # 예측한 값 저장
            x, y = 30, 50
            text = '{}'.format(class_names[preds])  # text 저장
    
            # 이미지의 (x, y)에 텍스트 출력
            draw_text(cam, text, x, y)
            cv2.imshow('camera', cam)
    
            if cv2.waitKey(1) & 0xFF == 27:  # esc를 누르면 종료
                break
    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser(description="webcam을 이용한 재활용 pet 분류...")
    parser.add_argument('--weights', type=str, required=True, help='학습시킨 모델pt 파일')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
