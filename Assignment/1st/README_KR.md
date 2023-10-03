# Assignment 1 - Practice AlexNet

### 개요

1. PyTorch tutorial 을 참고하여 나만의 AlexNet 코드 작성하기
2. Fashion MNIST 데이터 혹은 다른 데이터를 사용하여 분류 수행하기


----

### DataSet

* DataSet : Fashion MNIST
* 특징 : 운동화, 셔츠, 샌들과 같은 작은 이미지 모음
* 개수 : 70,000
* 28x28 grayscale image
* Label : 10개 구분

|Label |Class|
|-----|------|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

-------

### Process
1. 필요 라이브러리 가져오기
``` python
import os # 파이썬을 이용해 파일을 복사하거나 디렉터리를 생성하고 특정 디렉터리 내의 파일 목록을 구하고자 할 때 사용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision # torchvision package : 컴퓨터 비전을 위한 유명 데이터셋, 모델 아키텍처, 이미지 변형등을 포함
import torch.nn as nn # nn : neural netwroks (define class) attribute를 활용해 state를 저장하고 활용
import torch.optim as optim # 최적화 알고리즘
import torch.nn.functional as F # (define function) 인스턴스화 시킬 필요없이 사용 가능
from PIL import Image
from torchvision import transforms, datasets # transforms : 데이터를 조작하고 학습에 적합하게 만듦.
from torch.utils.data import Dataset, DataLoader
# dataset : 샘플과 정답(label)을 저장
# DataLoader : Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감싼다.
```

2. GPU 사용하도록 설정
``` python
epochs = 10 # 훈련 반복수
batch_size = 512 # 배치 크기

device = ("cuda" if torch.cuda.is_available() else "cpu") # device 정의
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # 총 10개의 클래스

print(torch.__version__)
print(device)

# 결과
# 2.0.1+cu118
# cuda
```

3. 데이터 셋 준비
  * Fashion MNIST 데이터 셋 사용
  * 인터넷에서 다운로드
``` python
#데이터 셋 준비
transform = transforms.Compose([
    transforms.Resize(227), # Compose : transforms 리스트 구성
    # 227x227 : input image(in alexnet) but fashionMNIST's input image : 28x28
    transforms.ToTensor()]) # ToTensor : PIL image or numpy.ndarray를 tensor로 바꿈

training_data = datasets.FashionMNIST(
    root="data", # data가 저장될 경로(path)
    train=True, # training dataset
    download=True, # 인터넷으로부터 데이터 다운
    transform=transform # feature 및 label 변환(transformation) 지정
)

validation_data = datasets.FashionMNIST(
    root="data",
    train=False, # test dataset
    download=True,
    transform=transform
)
```
``` python
#Data Loader
# (class) DataLoader(dataset, batch_size, shuffle, ...)
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True)
```

4. 모델 생성
``` python
class AlexNet_Fashion_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out
```
``` python
model = AlexNet_Fashion_MNIST().to(device) # to()로 모델에 gpu 사용
criterion = F.nll_loss
optimizer = optim.Adam(model.parameters()) # model(신경망) 파라미터를 optimizer에 전달해줄 때 nn.Module의 parameters() 메소드를 사용
```
5. Train/Test 함수 작성
 * Train
``` python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # enumberate() : 인덱스와 원소로 이루어진 튜플(tuple)을 만들어줌
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 항상 backpropagation 하기전에 미분(gradient)을 zero로 만들어주고 시작해야 한다.
        output = model(data)
        loss = criterion(output, target) # criterion = loss_fn
        loss.backward() # Computes the gradient of current tensor w.r.t. graph leaves
        optimizer.step() # step() : 파라미터를 업데이트함
        if (batch_idx) % 100 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```
* Test
``` python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        #print test_loss & accuracy
        test_loss /= len(test_loader.dataset)  # -> mean
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('='*50)
```

6. 학습

``` python
for epoch in range(1, epochs+1):
    train(model, device, training_loader, optimizer, epoch)
    test(model, device, validation_loader)

```

7. 학습 결과
```
Test set: Average loss: 0.3113, Accuracy: 9048/10000 (90%)
```
-------

### Takeaways
* PyTorch 와 coLab 모두 처음 사용해보며 사용법을 익힐 수 있었다.
* AlexNet 틀 안에서 레이어들의 채널 수 등을 다르게 설정하면서 값이 천차만별로 변하는 것을 확인할 수 있었다.
* Cross Entropy loss 를 사용하려고 했으나, PyTorch 에서 사용에 어려움을 겪었다. 이로 인해 Negative Log-likelihood(NLL) Loss 를 적용하였다.
* GPU 사용으로 설정해주지 않아 처음에 모델 Train 단계에서 1번의 Epoch 수행 과정이 30분 이상 소요 되었다. GPU로 설정을 변경한 뒤 2분 내외로 속도가 개선되는 것을 확인할 수 있었다.

-------

### Appendix
* https://www.kaggle.com/code/tiiktak/fashion-mnist-with-alexnet-in-pytorch-92-accuracy
* Negative log likelihood
  * 입력값 X와 parameter θ가 주어졌을 때 정답 Y가 나타낼 확률, 즉 likelihood P(Y|X;θ)를 최대화하는 θ가 우리가 찾고 싶은 결과
  * 학습데이터의 각각의 likelihood를 log scale로 바꾸어도 argmax의 결과는 바뀌지 않으므로, likelihood의 곱을 최대로 만드는 θ와 log likelihood의 기대값을 최대로 하는 θ는 같다
