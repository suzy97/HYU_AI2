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
2. GPU 사용하도록 설정함 - PyTorch 의 특징
3. 데이터 셋 준비
  * Fashion MNIST 데이터 셋 사용
  * 인터넷에서 다운로드
4. 모델 생성
5. Train/Test 함수 작성
6. 학습

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
