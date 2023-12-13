# Transformer - Multi30K 예제 (독일어 - 영어)

## 들어가기 전

본 문서는 Transformer에 대한 기본 지식을 가진 딥러닝 초보자를 위한 문서입니다.

이 프로젝트는 "Attention Is All You Need" 논문을 기반으로 한 트랜스포머 모델의 기본 구현을 다루며, Google Colab에서 실행됩니다. 이 튜토리얼은 독일어에서 영어로 번역하는 모델을 구축하고 훈련시키는 과정을 보여줍니다.

## Transformer 란?

- 2017년 구글에서 소개된 <Attention is All You Need> 논문에서 처음 등장했습니다.
- Attention Mechanism만을 활용하여 크고 제한적인 데이터를 효과적으로 처리할 수 있는 기술입니다.
- RNN의 단점을 극복하고, GPT, BERT 등의 기계번역 분야 발전에 계기가 되었습니다.
- Encoder와 Decoder 파트로 구성되며, 각 파트는 Self-Attention 레이어와 Neural Network로 구성됩니다.

## 데이터 셋 소개

- Multi30k 데이터 셋을 사용합니다.
- 약 30,000개의 독일어와 영어 문장 쌍을 포함하며, 이미지 설명 번역 작업에 주로 사용됩니다.
- torchtext를 통해 다운로드하거나, 다른 방법으로 Google Drive에 업로드하여 사용합니다.
- Dataset 구성: Training (29,000개), Validation (1014개), Test (1000개)

## 전처리 과정

1. spacy 라이브러리를 이용한 영어와 독일어 문장의 토큰화 (Tokenization)
2. 독일어와 영어 토큰화 함수 정의 및 torchtext의 Field 라이브러리 사용
3. 최소 2회 이상 등장한 단어로 독일어와 영어 사전 구축

    ```python
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)
    ```

## 인코더

- Multi Head Attention 구조와 Feed Forward Neural Network 사용
- 문장의 빈 공간을 채우는 <pad> 토큰은 mask 처리
- Positional Encoding 값 구하여 문장의 임베딩과 더함

## 디코더

- 입력과 출력의 차원이 동일한 구조
- Masked Multi Head Attention 레이어에서 Encoder의 출력값을 Attention 처리

## 모델 학습

- Hidden Layer의 Dimension: 256, Head 개수: 8, 내부 Dimension: 512
- Dropout: 0.1, Optimizer: Adam, Learning Rate: 0.0005
- Epoch: 10회, 1 Epoch 당 약 15초 소요

    ```python
    Epoch: 01 | Time: 0m 17s
    Train Loss: 4.221 | Train PPL: 68.073
    Validation Loss: 3.052 | Validation PPL: 21.164
    ...
    ```

## 검증

- 소스 문장과 타겟 문장을 출력하고, 모델 출력 결과를 비교
- 임의로 10번째의 문장을 가져와 비교

    ```python
    example_idx = 10

    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']

    print(f'소스 문장: {src}')
    print(f'타겟 문장: {trg}')
    ...
    ```

- Attention 시각화를 통해 각 Head 별 단어의 가중치 확인
- - Attention 시각화는 모델의 각 Head에서 단어에 주어진 가중치를 시각적으로 나타내며, 핵심 단어의 중요도를 파악하는 데 도움을 줍니다. 예를 들어, "Young", "Man", "Skateboard"와 같은 단어들이 높은 attention score를 받은 것을 확인할 수 있습니다.

    ![Attention Visualization](https://example.com/attention_visualization.png)

- 결과적으로, 이 트랜스포머 모델은 독일어 문장을 영어로 효과적으로 번역하는 것을 보여줍니다. 이는 기계 번역 분야에서의 중요한 진보를 나타냅니다.

## 결론

이 튜토리얼을 통해 Transformer 모델의 기본 구조와 작동 방식에 대한 이해를 높일 수 있었습니다. Multi30k 데이터셋을 활용한 이번 프로젝트는 독일어에서 영어로의 번역 과정을 통해 모델의 성능을 시험하고, 실제 언어 처리 작업에 Transformer를 어떻게 적용할 수 있는지를 보여줍니다. 

이 문서는 Transformer에 대한 기본적인 이해를 바탕으로, 실제 모델 구현과 훈련 과정을 따라가며 실습해 볼 수 있는 기회를 제공합니다. 딥러닝과 기계 번역에 관심 있는 이들에게 유익한 참고 자료가 될 것입니다.

---

본 튜토리얼은 "Attention Is All You Need" 논문을 기반으로 하며, Google Colab에서 실행되도록 구성되었습니다.

**참고 문헌**
- Vaswani, A., et al. (2017). Attention is All You Need. 
- Multi30k Dataset: [https://github.com/multi30k/dataset](https://github.com/multi30k/dataset)

---

본 문서는 교육 및 학습 목적으로 작성되었습니다. 
