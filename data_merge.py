# -*- coding: utf-8 -*-

import os
from glob import glob

def file_merge(path, path1):  # 이미지 파일이 있는 경로

  file = glob(os.path.join(path, '*'))  #  일회용, 기타,  페트병
  
  files = [glob(os.path.join(folder_path,'*')) for folder_path in file]
  
  disposable = [glob(os.path.join(p,'*')) for p in files[0]] # 일회용
  disposable_list = [img  for file in disposable for img in file] # 일회용 이미지들
  
  file1 = glob(os.path.join(path1, '*')) # 일회용 페트병 추가 수집
  disposable_plus = [img for file in file1 for img in glob(os.path.join(file, '*'))]
  
  disposable_final = disposable_list + disposable_plus # 일회용 페트병 전체 리스트
  
  # pet_label data 출력
  norecycle = glob(os.path.join('label', 'label', '*'))
  norecycle_list = [img for file in norecycle for img in glob(os.path.join(file, '*'))]
  
  # pet_unlabel data 출력
  recycle = glob(os.path.join('label', 'unlabel', '*'))
  recycle_list = [img for file in recycle for img in glob(os.path.join(file, '*'))]
  
  # pet_unlabel 추가data 출력
  recycle_plus = glob(os.path.join('label','unlabel_plus', '*'))
  recycle_plus_list = [img for file in recycle_plus for img in glob(os.path.join(file,'*'))]
  
  recycle_final = recycle_list + recycle_plus_list # pet_unlabel 최종 data list
   
  
  return disposable_final, recycle_final, norecycle_list

