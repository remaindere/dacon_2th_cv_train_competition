# dacon_2th_cv_train_competition

제 2회 컴퓨터 비전 학습 경진대회 5등 수상  

개요   
```
이미지 속에 들어있는 알파벳과 들어있지 않은 알파벳을 분류하는 multi-label classification Task
```

원시적인 EDA 수행  
![image](https://user-images.githubusercontent.com/48322490/122669878-353ab080-d1fa-11eb-8bc3-3f59eadd2b82.png)
```
전체적으로 이미지에 노이즈가 많이 끼어 있고, "노이즈" 같은 다른 글자가 이미지에 있다
```

EfficientNet B7(pretrained) 사용 및 dacon 에서 제공한 baseline 코드 사용  

data는 DACON측 저작권 관련 이유로 첨부하지 않았습니다.  

reference :
dacon baseline codes https://dacon.io/competitions/official/235697/codeshare/2353?dtype=recent  
efficient-pytorch https://github.com/lukemelas/EfficientNet-PyTorch
   
