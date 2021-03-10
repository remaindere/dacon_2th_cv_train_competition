import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet

device = torch.device('cpu')

class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}

to_tensor = T.Compose([
                        ToTensor()
                    ])

augmentations = T.Compose([
                           T.ToPILImage(),
                           T.RandomRotation(40),
                           T.ToTensor(),
                           ])

class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 dir_path,
                 meta_df,
                 transforms=to_tensor,
                 augmentations=None):
        self.dir_path = dir_path 
        self.meta_df = meta_df 
        self.transforms = transforms
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, index):
        image = cv2.imread(self.dir_path +\
                           str(self.meta_df.iloc[index,0]).zfill(5) + '.png',
                           cv2.IMREAD_GRAYSCALE)
        image = (image/255).astype('float')[..., np.newaxis]

        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)

        if self.augmentations:
            sample["image"] = self.augmentations(sample["image"])

        return sample


if __name__ == "__main__":
    # input 폴더에서 사용자가 저장한 sample_submission.csv를 load
    sample_submission = pd.read_csv('./input/sample_submission.csv')
    # 추론할 데이터는 submit/data/ 폴더에 들어있다고 가정하고 코드를 작성
    test_dataset = DatasetMNIST("./data/test_dirty_mnist_2nd/", sample_submission)

    # input 폴더에서 사용자가 저장한 sample_submission.csv를 load
    prediction_df = pd.read_csv('./input/sample_submission.csv')
    
    ##########################################################################
    # model1의 output (허재섭-HJS)
    batch_size = 32
    test_data_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False
    )

    path1 = "./input/"

    class MultiLabelEfficientnet(nn.Module):
        def __init__(self):
            super(MultiLabelEfficientnet, self).__init__()
            self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
            self.drop = nn.Dropout(p=0.2)
            self.FC = nn.Linear(1000, 26)

        def forward(self, x):
            x = F.silu(self.conv2d(x))

            x = F.silu(self.efficientnet(x))

            x = self.drop(x)
            x = torch.sigmoid(self.FC(x))
            return x

    best_models1 = []
    best_models1.append(torch.load(path1 + 'efficientnetb7_1_0.1780.pth', map_location=torch.device('cpu')))
    best_models1.append(torch.load(path1 + '2_efficientnetb7_0.1672.pth', map_location=torch.device('cpu')))
    best_models1.append(torch.load(path1 + '3_efficientnetb7_0.1581.pth', map_location=torch.device('cpu')))
    best_models1.append(torch.load(path1 + 'efficientnetb7_4_0.1610.pth', map_location=torch.device('cpu')))
    best_models1.append(torch.load(path1 + 'efficientnetb7_5_0.1645.pth', map_location=torch.device('cpu')))

    predictions_list1 = []

    # 5개의 fold마다 가장 좋은 모델을 이용하여 예측
    for model in tqdm(best_models1):
        # 0으로 채워진 array 생성
        prediction_array1 = np.zeros([prediction_df.shape[0],
                                    prediction_df.shape[1] -1])
        print(test_data_loader)
        for idx, sample in enumerate(test_data_loader):
            with torch.no_grad():
                # 추론
                model.eval()
                images = sample['image']
                images = images.to(device)
                probs  = model(images)
                probs = probs.cpu().detach().numpy()
                preds = (probs > 0.5)

                # 예측 결과를 
                # prediction_array에 입력
                batch_index = batch_size * idx
                prediction_array1[batch_index: batch_index + images.shape[0],:]\
                            = preds.astype(int)
                         
        # 채널을 하나 추가하여 list에 append
        predictions_list1.append(prediction_array1[...,np.newaxis])
    # 각 fold 별 결과 concat
    predictions_array1 = np.concatenate(predictions_list1, axis = 2)

    # model2의 output (송광원-잉돌)
    batch_size = 128
    test_data_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False
    )

    path2 = "./input/"

    # swish
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    # A memory-efficient implementation of Swish function
    class SwishImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, i):
            result = i * torch.sigmoid(i)
            ctx.save_for_backward(i)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            i = ctx.saved_tensors[0]
            sigmoid_i = torch.sigmoid(i)
            return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

    class MemoryEfficientSwish(nn.Module):
        def forward(self, x):
            return SwishImplementation.apply(x)
    class MultiLabelEfficientnet(nn.Module):
        def __init__(self):
            super(MultiLabelEfficientnet, self).__init__()
            self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
            self._swish = MemoryEfficientSwish()
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
            self.fc = nn.Linear(1000, 26)

        def forward(self, x):
            x = self._swish(self.conv2d(x))

            x = self._swish(self.efficientnet(x))

            x = torch.sigmoid(self.fc(x))
            return x

    best_models2 = []
    best_models2.append(torch.load(path2 + '1_Effnetb7_3th_0.2018_epoch_27.pth', map_location=torch.device('cpu')))
    best_models2.append(torch.load(path2 + '2_Effnetb7_3th_0.2010_epoch_27.pth', map_location=torch.device('cpu')))
    best_models2.append(torch.load(path2 + '3_Effnetb7_3th_0.1920_epoch_26.pth', map_location=torch.device('cpu')))
    best_models2.append(torch.load(path2 + '4_Effnetb7_3th_0.1954_epoch_28.pth', map_location=torch.device('cpu')))
    best_models2.append(torch.load(path2 + '5_Effnetb7_3th_0.1909_epoch_26.pth', map_location=torch.device('cpu')))

    predictions_list2 = []

    # 5개의 fold마다 가장 좋은 모델을 이용하여 예측
    for model in tqdm(best_models2):
        # 0으로 채워진 array 생성
        prediction_array2 = np.zeros([prediction_df.shape[0],
                                    prediction_df.shape[1] -1])
        for idx, sample in enumerate(test_data_loader) :
            with torch.no_grad():
                # 추론
                model.eval()
                images = sample['image']
                images = images.to(device)
                probs  = model(images)
                probs = probs.cpu().detach().numpy()
                preds = (probs > 0.5)

                # 예측 결과를 
                # prediction_array에 입력
                batch_index = batch_size * idx
                prediction_array2[batch_index: batch_index + images.shape[0],:]\
                            = preds.astype(int)
                            
        # 채널을 하나 추가하여 list에 append
        predictions_list2.append(prediction_array2[...,np.newaxis])

    # 각 fold 별 결과 concat
    predictions_array2 = np.concatenate(predictions_list2, axis = 2)

    ################################################################################
    # 각 모델별 결과 concat
    predictions_array_final = np.concatenate([predictions_array1, predictions_array2], axis=2)
    predictions_mean = predictions_array_final.mean(axis = 2)
    # 평균 값이 0.5보다 클 경우 1 작으면 0
    predictions_mean = (predictions_mean > 0.5) * 1
    
    # input 폴더에서 사용자가 저장한 sample_submission.csv를 load
    # 다른 형태의 데이터가 필요하다면 저장가능
    sample_submission = pd.read_csv("./input/sample_submission.csv")
    sample_submission.iloc[:,1:] = predictions_mean
    
    # 추론한 결과는 submit/output/ 폴더에 submission.csv라는 이름으로 저장
    sample_submission.to_csv("./output/submission.csv", index = False)
