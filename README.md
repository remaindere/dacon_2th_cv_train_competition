# dacon_2th_cv_train_competition

제 2회 컴퓨터 비전 학습 경진대회 **5등 수상**  

## 개요   

이미지 속에 들어있는 알파벳과 들어있지 않은 알파벳을 분류하는 **multi-label classification Task** 입니다.

## 코드 사용

*dacon_2th_CV_dirtymnist_5th_place_Ensembled.ipynb* 파일을 Google Drive에 저장하신 후 Colab 환경에서 실행하시면 됩니다.  


## DIRTY MNIST Data, EDA  
![image](https://user-images.githubusercontent.com/48322490/122669878-353ab080-d1fa-11eb-8bc3-3f59eadd2b82.png)
  
1. 이미지에 노이즈가 많이 끼어 있음  
2. "노이즈" 같은 다른 글자가 이미지에 있음  
3. 데이터의 수 충분  
4. 알파벳의 분포 균일  
  
## 모델  
  
1. ResNext, Densenet, Efficientnet 등의 모델 실험 진행  
2. 최종 모델 제출시 EfficientNet B7(pretrained) 사용  
3. activation 함수 Swish(SiLU) 사용  
4. Multilabel Classification이므로, 최종 Activation에 Sigmoid 가 적용됨   
5. Probability가 0.5 이상일시 해당 class가 존재한다고 추론   

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
개발 환경 Colab 사용, 한 계정에 3개 정도의 세션을 사용하여 각 폴드를 동시에 학습시켜 시간을 크게 절감할수 있었음  
  
## Inference  
  
Soft Voting Emsemble, 각각의 모델에서 예측한 Prediction Listf를 concatenate  
이후 평균값을 계산한 후, 0.5 초과일 시 1로 추론  
pandas를 이용하여 csv 파일로 저장  
DACON 측에 최종 제출할 때 DockerFile 첨부하여 제출  

## 주요 이슈 및 해결  
  
1. AI 대회에 첫 참여해 보는 참이라, Pytorch 및 scikit-learn 등의 다양한 library 사용이 어려웠습니다.  
  팀 내 그룹 스터디를 진행하였고, 제공된 DACON Baseline Code 리딩 중 이해가 가지 않는 부분들을 공유하여 공부하였습니다.  
  
2. 경험 부족으로 인해 기본적인 성능 향상 전략 수립이 어려웠습니다.  
  Kaggle에서 [관련 Competition](https://www.kaggle.com/c/digit-recognizer/overview)의 Discussion을 참고하였습니다.   
  성능 향상을 위한 모델 선택의 방향성을 정하기 위하여 PapersWithCode에서 [해당 Task](https://paperswithcode.com/task/image-classification)의 SOTA 모델을 참고하였습니다.   


### data는 저작권 관련 이유로 첨부하지 않았습니다!  


## Contributors
:floppy_disk:**[송광원(remaindere)](https://github.com/remaindere)** | :spades:**[허재섭(shjas94)](https://github.com/shjas94)**

## eference
**[dacon baseline codes](https://dacon.io/competitions/official/235697/codeshare/2353?dtype=recent)**  
**[efficient-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)**  
   
