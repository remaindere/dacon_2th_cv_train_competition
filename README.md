# dacon_2th_cv_train_competition

제 2회 컴퓨터 비전 학습 경진대회 5등 수상  

## 개요   

이미지 속에 들어있는 알파벳과 들어있지 않은 알파벳을 분류하는 multi-label classification Task 입니다.


## DIRTY MNIST Data 간단한 EDA 수행  
![image](https://user-images.githubusercontent.com/48322490/122669878-353ab080-d1fa-11eb-8bc3-3f59eadd2b82.png)
  
1. 이미지에 노이즈가 많이 끼어 있음  
2. "노이즈" 같은 다른 글자가 이미지에 있음  
3. 데이터의 수 충분  
4. 알파벳의 분포 균일  
  
## 모델  
  
ResNext, Densenet, Mobilenet, Efficientnet 등의 모델 실험 진행  
최종 모델 제출시 EfficientNet B7(pretrained) 사용  
activation 함수 Swish(SiLU) 사용  
Multilabel Classification이므로, 최종 Activation에 Sigmoid 가 적용됨   
Probability가 0.5 이상일시 해당 class가 존재한다고 추론   

## Augmentation  
  
Tensorvision을 이용한 다양한 Augmentation 실험 (Flip, Rotation, GaussianNoise, RandomAffine etc..)  
RandomRotation을 제외하면 모두 성능 저하를 보임  

## Train  
  
```
criterion = torch.nn.BCELoss()  
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)  
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
kfold = KFold(n_splits=5, shuffle=True, random_state=42) 
```
개발 환경은 Colab 이었으며, 한 계정에 3개 정도의 세션을 사용하여 각 폴드를 동시에 학습시켜 시간을 크게 절감할수 있었음  
  
## Inference  
  
Soft Voting Emsemble, 각각의 모델에서 예측한 Prediction Listf를 concatenate  
이후 평균값을 계산한 후, 0.5 초과일 시 1로 추론  
pandas를 이용하여 csv 파일로 저장  
DACON측에 최종 제출할 때 Docker Container 첨부하여 제출  

### data는 DACON측 저작권 관련 이유로 첨부하지 않았습니다!

## Contributors

:floppy_disk:[송광원(remaindere)](https://github.com/remaindere) | :spades:[허재섭(shjas94)](https://github.com/shjas94)

## reference
[dacon baseline codes](https://dacon.io/competitions/official/235697/codeshare/2353?dtype=recent)  
[efficient-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
   
