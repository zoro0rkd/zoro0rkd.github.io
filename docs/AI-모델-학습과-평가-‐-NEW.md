## 1. AI 모델 학습 개요
### 학습의 정의
AI 모델 학습이란, 머신러닝 모델이 주어진 데이터 셋을 통해 작업 수행에 필요한 패턴과 상관관계를 학습하여 성능을 최적화하는 과정입니다. 즉, AI 모델이 데이터에서 특정 패턴을 인식하고 결과를 예측하거나 결정을 내릴 수 있도록 모델의 내부 매개변수(가중치와 편향 등)를 조정하는 것을 말합니다.

모델 학습 과정은 다음과 같은 특징을 갖습니다:

- 학습 데이터는 모델이 실제 문제를 해결할 때 유사한 상황을 경험할 수 있도록 관련성이 있어야 합니다.
- 모델은 주어진 데이터를 통해 수학적 함수의 계수를 최적화하여 더 정확한 출력을 생성하도록 조정됩니다.
- 학습 결과로 얻어진 내부 매개변수는 모델이 이전에 본 적 없는 새로운 데이터에서도 예측을 수행할 수 있는 '지식'을 나타냅니다.
- 학습은 지도학습, 비지도학습, 강화학습 등 다양한 방법으로 이루어질 수 있으며, 복잡한 신경망을 포함하는 딥러닝 모델도 이 과정을 거칩니다.
- AI 모델 학습은 단순한 예측 모델부터 생성형 AI까지 다양한 형태의 AI 시스템에서 핵심적인 역할을 합니다.

한편, AI 학습은 기존 머신러닝 범위를 넘어서 자율적으로 적응하고 추론하며 의사 결정을 내릴 수 있는 시스템을 만드는 더 넓은 개념을 포함하기도 합니다.

요약하면, AI 모델 학습은 데이터 기반으로 모델을 '가르치고' 최적화하여 미래에 유사한 문제에서 올바른 결과를 도출하도록 만드는 과정을 의미합니다. 이는 AI 모델 성능을 결정하는 가장 중요한 단계입니다. 

출처
[1] 모델 학습이란 무엇인가요? https://www.ibm.com/kr-ko/think/topics/model-training
[2] AI 모델이란? | 용어 해설 https://www.hpe.com/kr/ko/what-is/ai-models.html
[3] AI 모델 훈련: 그것이 무엇이고 어떻게 작동하는가 https://www.mendix.com/ko/blog/ai-model-training/
[4] AI 모델(AI Model)의 이해 https://www.databricks.com/kr/glossary/ai-models
[5] AI 학습이란 무엇인가요? https://www.lenovo.com/kr/ko/glossary/ai-learning/
[6] AI 모델이란 무엇인가요? https://www.ibm.com/kr-ko/think/topics/ai-model
[7] AI 모델 훈련이란 무엇이고 왜 중요한 것인가요? https://www.oracle.com/kr/artificial-intelligence/ai-model-training/
[8] AI 모델이란? https://cloud.google.com/discover/what-is-an-ai-model?hl=ko
[9] [All Around AI 6편] 생성형 AI의 개념과 모델 - SK하이닉스 뉴스룸 https://news.skhynix.co.kr/all-around-ai-6/
[10] [AI] 딥러닝 모델 학습과정 이해하기 - EveryDay.DevUp - 티스토리 https://everyday-devup.tistory.com/200

### 지도학습(Supervised Learning)
지도학습(Supervised Learning)은 머신러닝의 주요 학습 방법 중 하나로, 입력 데이터와 그에 대응하는 정답(레이블)이 함께 제공되는 학습 데이터를 사용하여 모델을 학습시키는 과정입니다. 

#### 지도학습의 정의
- 주어진 입력값(features)과 그에 해당하는 출력값(label)의 쌍으로 이루어진 데이터셋을 통해 모델이 입력과 출력 간의 관계를 학습함
- 목표는 학습 데이터로부터 함수 혹은 규칙을 찾아내어, 새로운 입력 데이터에 대해 정확한 출력을 예측하는 것

#### 지도학습의 특징
- 정답(label)이 명시된 데이터를 기반으로 학습하기 때문에 모델의 학습 정확도가 높음
- 레이블링된 데이터가 필요하며, 이 데이터 준비 과정이 비용과 시간이 많이 들 수 있음
- 학습된 모델은 새로운 데이터에 대해 분류(Classification)나 회귀(Regression) 문제를 해결할 수 있음
  - **분류**: 입력 데이터가 특정 클래스(예: 고양이, 개, 새 등) 중 어느 것에 속하는지 예측
  - **회귀**: 연속적인 값(예: 주택가격, 온도 등)을 예측

#### 지도학습 과정 예시
- 10,000장 새 이미지 중 8,000장에 새 종류 레이블을 붙여 학습 데이터로 사용
- 모델이 이 데이터를 통해 새의 특징을 학습하면 나머지 2,000장에 대해 새 종류를 예측할 수 있음

#### 주요 지도학습 알고리즘 예
- 나이브 베이즈 분류기 (Naïve Bayes)
- 서포트 벡터 머신 (SVM)
- 결정 트리 및 랜덤 포레스트
- 신경망(Neural Networks)
- 로지스틱 회귀(Logistic Regression) 등

지도학습은 명확한 정답이 있는 문제에서 매우 효과적이며, 일상적인 AI 응용 분야의 상당 부분에서 핵심적인 학습 방법으로 사용됩니다. 대표적인 응용으로는 스팸 메일 분류, 이미지 인식, 음성 인식, 금융 데이터 기반 예측 등이 있습니다.

출처
[1] 지도 학습(Supervised Learning)이란 무엇인가? - Appier https://www.appier.com/ko-kr/blog/what-is-supervised-learning
[2] 지도 학습이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/supervised-learning
[3] 지도 학습 (Supervised Learning) - 도리의 디지털라이프 https://blog.skby.net/%EC%A7%80%EB%8F%84-%ED%95%99%EC%8A%B5-supervised-learning/
[4] 지도학습이란? | Supervised Learning(Regression, Classification) https://standing-o.github.io/posts/supervised-learning/
[5] 지도학습(Supervised learning)에 대해 알아보자 - CAI - 티스토리 https://kjh-ai-blog.tistory.com/3
[6] supervised learning (지도학습) - 인공지능(AI) & 머신러닝(ML) 사전 https://wikidocs.net/120172
[7] 지도학습 - 주홍색 코딩 https://kwonkai.tistory.com/154
[8] [인공지능] 지도학습, 비지도학습, 강화학습 - 삶은 확률의 구름 https://ebbnflow.tistory.com/165
[9] 지도 학습이란 무엇인가요? | Oracle 대한민국 https://www.oracle.com/kr/artificial-intelligence/machine-learning/supervised-learning/

---

### [비지도학습(Unsupervised Learning)](https://github.com/zoro0rkd/ai_study/wiki/AI-모델-아키텍처-설계-%E2%80%90-NEW#7-비지도-학습unsupervised-learning-구조)

링크 참고

---

### 강화학습(Reinforcement Learning)
강화학습(Reinforcement Learning, RL)은 머신러닝의 한 분야로, 에이전트(학습 주체)가 환경과 상호작용하면서 시행착오를 통해 최적의 행동을 학습하는 방법입니다.

#### 강화학습의 정의
- 에이전트가 현재 상태(State)를 인식하고, 가능한 여러 행동(Action) 중에서 하나를 선택함
- 선택한 행동에 대해 환경으로부터 보상(Reward)을 받음
- 에이전트는 이 보상을 최대화하기 위해 앞으로 어떤 행동을 선택해야 할지 학습함
- 명시적인 정답(Label) 없이도 보상 신호를 통해 스스로 학습하는 점이 특징

#### 강화학습의 주요 구성 요소
- **에이전트(Agent):** 학습을 수행하는 주체
- **환경(Environment):** 에이전트가 상호작용하는 대상
- **상태(State):** 에이전트가 처한 현재 상황
- **행동(Action):** 에이전트가 선택할 수 있는 동작
- **보상(Reward):** 행동에 대한 평가 신호로, 에이전트가 목표 달성 정도를 판단하는 기준
- **정책(Policy):** 특정 상태에서 어떤 행동을 할지 결정하는 전략
- **가치함수(Value Function):** 특정 상태 또는 상태-행동 쌍의 기대 보상을 평가

#### 강화학습의 특징
- 시행착오(Trial and Error)를 기반으로 학습하며 경험을 통해 성능을 향상시킴
- 환경에 대한 명확한 모델이 없어도 학습 가능
- 인간의 학습 방식 중 하나인 '시행착오 학습'과 유사
- 미래의 누적 보상을 최대화하는 행동을 학습하는 것이 목적
- 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추어야 함 (새로운 행동 시도 vs 이미 아는 최적 행동 수행)

#### 예시
- 알파고가 바둑 대국에서 자기 자신과 반복적으로 대국하며 전략을 개선하는 과정
- 게임 AI가 플레이하며 승리 전략을 터득하는 것

강화학습은 복잡한 의사결정 문제나 동적 환경에서 최적의 행동을 배우는 데 매우 효과적인 방법입니다. 최근 딥러닝과 결합해 많은 분야에서 혁신적인 성과를 내고 있습니다.

출처
[1] 강화학습 개요 - Introduction of Reinforcement Learning https://skidrow6122.tistory.com/3
[2] 강화 학습 - 위키백과, 우리 모두의 백과사전 https://ko.wikipedia.org/wiki/%EA%B0%95%ED%99%94_%ED%95%99%EC%8A%B5
[3] 강화 학습이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/reinforcement-learning
[4] 강화학습이란? - MATLAB & Simulink - 매스웍스 https://kr.mathworks.com/discovery/reinforcement-learning.html
[5] 머신러닝의 꽃, 강화학습 - 브런치 https://brunch.co.kr/@namujini/22
[6] 강화학습이란? - velog https://velog.io/@dorthy/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5%EC%9D%B4%EB%9E%80
[7] 강화학습 개념부터 Deep Q Networks까지, 10분만에 훑어보기 https://jeinalog.tistory.com/20
[8] 딥러닝 (8) - [RL1] 강화학습(Reinforcement Learning)이란? https://davinci-ai.tistory.com/31
[9] Q: 강화 학습이란 무엇인가요? - AWS https://aws.amazon.com/ko/what-is/reinforcement-learning/
[10] 강화학습 - 나무위키 https://namu.wiki/w/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5

#### 대표적인 강화학습 알고리즘

1. **Q-러닝 (Q-Learning)**
   - 가치 기반 강화학습 알고리즘으로, 에이전트가 상태-행동 쌍의 가치를 학습하여 최적의 행동 정책을 찾음
   - 오프폴리시(off-policy) 방식으로, 현재 정책이 아닌 최적 정책에 맞게 학습

2. **SARSA (State-Action-Reward-State-Action)**
   - 온폴리시(on-policy) 방식의 가치 기반 알고리즘으로, 에이전트가 따르는 정책에 따라 학습
   - Q-러닝보다 더 안정적으로 학습하지만 최적 정책 탐색이 느릴 수 있음

3. **DQN (Deep Q-Network)**
   - Q-러닝에 딥러닝을 결합한 알고리즘
   - 신경망을 이용해 Q-값을 근사하여 고차원 입력 데이터나 복잡한 환경에서 효과적임

4. **REINFORCE**
   - 정책 기반(Policy-Based) 강화학습 기법
   - 직접 정책을 학습하며 확률적 정책을 사용해 행동 선택

5. **A2C, A3C (Advantage Actor-Critic, Asynchronous Advantage Actor-Critic)**
   - 액터-크리틱(actor-critic) 구조를 이용하는 정책 기반 알고리즘
   - 액터는 정책을, 크리틱은 가치 함수를 동시에 학습

6. **SAC (Soft Actor-Critic)**
   - 최대 엔트로피 강화학습 기법으로 안정적이며 샘플 효율성이 높음
   - 연속적인 행동 공간에 적합

7. **모델 기반 강화학습 (Model-Based RL)**
   - 환경 모델을 학습하여 시뮬레이션 기반으로 최적 행동을 탐색
   - 예: Dyna-Q, MBPO 등

이들 알고리즘은 다양한 환경과 문제 유형에 맞게 선택되고, 심층 강화학습(Deep RL) 기술과 결합되며 자율주행, 게임 AI, 로보틱스 등 다양한 분야에서 활발히 활용됩니다.

출처
[1] 강화학습 알고리즘의 종류(분류) - DACON https://dacon.io/forum/406104
[2] [인공지능] 강화학습 기법 - 종류와 해당 알고리즘 정리 - 매석의 메모장 https://maeseok.tistory.com/135
[3] 강화학습 알고리즘 종류와 특징 - 건강 생활 - 티스토리 https://healthlife88.tistory.com/entry/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%A7%95
[4] RL 알고리즘의 종류 : Model-Free vs Model-Based - deeep - 티스토리 https://dalpo0814.tistory.com/52
[5] 4. 강화 학습(Reinforcement Learning) 알고리즘 - 내 삶속 AI - 위키독스 https://wikidocs.net/236199
[6] [강화학습] REINFORCE 알고리즘 : 개념 및 수식 - HIGHQUAL https://mengu.tistory.com/136
[7] 강화학습 - 나무위키 https://namu.wiki/w/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5
[8] 강화학습(Reinforcement Learning)이란 무엇일까? - 인공지능 연구소 https://minsuksung-ai.tistory.com/13
[9] 적절한 머신러닝 알고리즘을 선택하는 방법 | 블로그 - 모두의연구소 https://modulabs.co.kr/blog/choosing-right-machine-learning-algorithm

---

### Self-supervised learning
**Self-Supervised Learning(자기지도학습)**입니다.
- **Self-Supervised Learning**은 **라벨이 없는 대규모 데이터**에서 스스로 학습 신호를 만들어내어, 별도의 수작업 라벨 없이도 효과적으로 모델을 학습시키는 AI 방법입니다[1][6][8][9].
- 이 방식은 **표현 사전학습(Pretraining)** 단계에서 라벨 없는 데이터를 활용해 인코더를 학습하고, 이후 **소량의 라벨 데이터**로 미세조정(Fine-tuning)하여 특정 과제를 수행합니다[1][4][9].
- 대표적으로 InfoNCE 등 **contrastive(대비) 손실**을 활용하며, 음성·이미지·자연어 등 다양한 분야에서 **라벨링 비용 절감**과 **전이 학습** 효과로 주목받고 있습니다[1][4][9].


출처
[1] Self-Supervised Learning: 최소한의 라벨로 학습하는 AI 방법 - AI꿀정보 https://lifestyleimformation.tistory.com/entry/Self-Supervised-Learning-%EC%B5%9C%EC%86%8C%ED%95%9C%EC%9D%98-%EB%9D%BC%EB%B2%A8%EB%A1%9C-%ED%95%99%EC%8A%B5%ED%95%98%EB%8A%94-AI-%EB%B0%A9%EB%B2%95
[2] 2-3-3 딥러닝의 학습 방법 - AI와 클라우드컴퓨팅 입문 - 위키독스 https://wikidocs.net/240099
[3] 머신러닝이란 무엇인가요? https://channel.io/ko/blog/articles/what-is-machine-learning-6f0de76a
[4] 대형 사전훈련 모델의 파인튜닝을 통한 강건한 한국어 음성인식 모델 ... https://www.eksss.org/archive/view_article?pid=pss-15-3-75
[5] [논문리뷰] Robust Speech Recognition via Large-Scale Weak ... https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/whisper/
[6] 자기 지도 학습이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/self-supervised-learning
[7] 수동 어노테이션의 한계를 극복하는 기술 4가지 -오토레이블링, 자기 ... https://ahha.ai/2023/12/19/autolabeling/
[8] 자연어 처리 13강 - Self Supervised Learning 1 - 공대생 도전 일지 https://yoonschallenge.tistory.com/584
[9] 자기지도학습(Self-supervised Learning) - C's Shelter - 티스토리 https://gnuhcjh.tistory.com/49

#### Self-Supervised Learning(SSL) 방법의 두 단계 상세 설명

##### 1. 대규모 라벨 없는 데이터로 표현 사전학습 (Pretraining with Unlabeled Data)

- **목적:**  
  라벨이 없는 대규모 데이터(예: 음성, 이미지, 텍스트 등)에서 유용한 특징(Feature) 표현을 자동으로 학습합니다.
- **학습 방식:**  
  - 입력 데이터에서 인위적으로 학습 신호를 생성합니다.
  - 예시: 같은 음성 클립에서 두 가지 서로 다른 변환(증강)을 적용해 파형 쌍 $$(x, x')$$를 만듭니다.
  - 인코더(Encoder)는 이 두 파형의 표현이 서로 가깝도록 학습합니다.
- **대표적 손실 함수:**  
  - **InfoNCE**: 대표적인 contrastive(대비) 손실로, 같은 클립 쌍은 유사하게, 다른 클립은 멀어지도록 만듭니다.
  - **MSE(Mean Squared Error)** 등 유사도 기반 손실도 사용됩니다.
- **장점:**  
  - 라벨이 없는 데이터만으로도 강력한 표현을 학습할 수 있어, 대규모 수작업 라벨링 비용이 절감됩니다.
  - 다양한 변환(Noise, Time Stretch, Masking 등)을 통해 모델이 견고한 특징을 익힙니다.

##### 2. 소량 라벨 데이터로 미세조정 (Fine-tuning with Labeled Data)

- **목적:**  
  사전학습된 인코더의 파라미터를 이어받아, 소량의 라벨이 달린 데이터로 실제 과제를 수행하도록 미세조정합니다.
- **학습 방식:**  
  - 인코더 가중치(θ)를 초기값으로 사용하고, 라벨 데이터(예: (x, y))로 분류기 등 다운스트림 모델을 학습합니다.
  - 이때는 정답 라벨이 필요합니다.
- **대표적 손실 함수:**  
  - **CE(Cross Entropy)**: 분류 문제에서 널리 사용.
  - **CTC(Connectionist Temporal Classification)**: 음성 인식 등 시퀀스 라벨링에서 사용.
- **장점:**  
  - 소량의 라벨 데이터만으로도 높은 성능을 달성할 수 있습니다.
  - 사전학습된 표현 덕분에 데이터 효율성이 크게 향상됩니다.

#### 요약 표

| 단계 | 사용 데이터 | 학습 목표 | 주요 손실 함수 | 특징 |
|------|-------------|-----------|---------------|------|
| 표현 사전학습 | 라벨 없는 대규모 데이터 | 유용한 특징 표현 학습 | InfoNCE, MSE 등 | 라벨 불필요, 데이터 다양성 활용 |
| 과제 미세조정 | 소량 라벨 데이터 | 실제 과제 성능 극대화 | CE, CTC 등 | 라벨 필요, 사전학습 인코더 사용 |

#### 실제 예시

- **음성:**  
  - 사전학습: 수십만 시간의 라벨 없는 음성으로 wav2vec, HuBERT 등 모델 학습  
  - 미세조정: 수천 개 라벨 음성으로 음소/단어 인식기 학습
- **이미지:**  
  - 사전학습: 이미지 일부 가리기(Masking) 등으로 특징 추출  
  - 미세조정: 소량 라벨 이미지로 분류기 학습

이처럼 SSL은 대규모 비라벨 데이터에서 표현을 먼저 익히고, 소량 라벨 데이터로 실제 과제에 맞게 모델을 미세조정하는 방식입니다.


## 2. 학습 데이터 준비
### 데이터 분할 (Train / Validation / Test)

머신러닝 및 AI 모델 학습에서 데이터 분할은 데이터를 학습과 평가에 적절히 사용하는 중요한 단계입니다. 전체 데이터를 여러 집합으로 나누어 각각의 역할을 분명히 함으로써 모델 성능을 정확하게 평가하고, 과적합(overfitting)을 방지합니다.

#### 분할 구성
- **Train Set (학습용 데이터)**  
  모델이 패턴과 관계를 학습하는 데 사용하는 데이터입니다. 이 데이터로 모델의 파라미터가 조정됩니다.
  
- **Validation Set (검증용 데이터)**  
  학습 중간에 모델의 성능을 확인하고, 하이퍼파라미터 조정이나 모델 선택 등에 사용되는 데이터입니다. 학습에는 직접적으로 사용되지 않지만 모델 성능을 빠르게 피드백받는 데 활용됩니다.
  
- **Test Set (평가용 데이터)**  
  학습과 검증이 완료된 후, 최종적으로 모델이 얼마나 잘 작동하는지 객관적으로 평가하는 데 사용하는 데이터입니다. 이 데이터는 절대 학습이나 검증에 사용하지 않으며, 한 번만 평가에 사용됩니다.

#### 일반적인 분할 비율
- 보통 전체 데이터셋을 **Train : Validation : Test = 6 : 2 : 2** 또는 60% : 20% : 20% 정도로 나누는 경우가 많습니다.
- Train 데이터 내에서 Validation 데이터를 분할하는 경우도 있으며, 데이터가 적을 땐 교차검증(K-fold cross-validation) 기법을 활용해 데이터 효율을 높이기도 합니다.

#### 분할의 중요성
- Train과 Test만 분할할 경우, 모델 성능을 한 번만 확인할 수 있고, 테스트 결과를 토대로 모델을 수정하면 과적합 위험이 커집니다.
- Validation 데이터를 두면 모델 성능을 중간 점검 하면서 최적화할 수 있어 진짜 일반화 성능을 더 잘 측정할 수 있습니다.
- Test 데이터는 학습 및 검증에 전혀 관여하지 않아야, 실제 환경에서의 성능을 객관적으로 평가할 수 있습니다.

#### 요약
1. **Train**: 모델 학습에 사용  
2. **Validation**: 학습 중 하이퍼파라미터 튜닝 및 중간 평가용  
3. **Test**: 학습 완료 후 최종 성능 평가용  

이처럼 데이터 분할은 머신러닝 모델의 성능을 신뢰성 있게 평가하고 개선하는 데 필수적인 절차입니다.

출처
[1] Train, Validation, and Test Set - 포자랩스의 기술 블로그 https://pozalabs.github.io/Dataset_Splitting/
[2] 데이터셋 나누기와 모델 검증 - velog https://velog.io/@minjung00/%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EB%82%98%EB%88%84%EA%B8%B0%EC%99%80-%EB%AA%A8%EB%8D%B8-%EA%B2%80%EC%A6%9D
[3] [Machine Learning][머신러닝] 데이터셋 나누기와 교차검증 https://ysyblog.tistory.com/69
[4] [머신러닝&딥러닝] Train / Validation / Test 의 차이 - JoJo's Study Blog https://wkddmswh99.tistory.com/10
[5] validation set이란? test set과의 차이점과 사용 방법 https://for-my-wealthy-life.tistory.com/19
[6] 데이터셋 나누기 & 교차 검증 - 코딩하는 오리 - 티스토리 https://cori.tistory.com/162
[7] Sklearn 익히기 - train/test data set 분리 및 Cross Validation https://libertegrace.tistory.com/entry/Sklearn-%EC%9D%B5%ED%9E%88%EA%B8%B0-trainvaltest-data-set-%EB%B6%84%EB%A6%AC-%EB%B0%8F-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC
[8] [데이터 전처리] 훈련 및 테스트 데이터 분할 - Smalldata Lab https://smalldatalab.tistory.com/21

---

### 교차검증(Cross Validation)

교차 검증(Cross Validation)은 머신러닝에서 모델의 성능을 더 정확하고 일반화되게 평가하기 위해 데이터를 여러 번 나누어 학습과 검증을 반복하는 방법입니다. 단순히 한 번만 학습 데이터와 검증 데이터로 나누는 것이 아니라, 여러 번 나누어 각각의 경우에 대해 모델을 학습시키고 평가하여 평균 성능을 구합니다[1][2][4].

#### 왜 교차 검증을 사용하는가?

- **과적합(Overfitting) 방지**: 모델이 특정 데이터셋에만 잘 맞는 현상을 줄이고, 다양한 데이터 분할에서의 성능을 확인할 수 있습니다[3][2].
- **일반화 성능 평가**: 여러 번 나누어 평가함으로써, 전체 데이터에 대한 모델의 일반화 능력을 더 정확히 측정할 수 있습니다[1][2].
- **데이터 부족 시 효과적**: 데이터가 적을 때도 모든 데이터를 학습과 검증에 최대한 활용할 수 있습니다[1][2].

#### 대표적인 교차 검증 방법

| 방법명              | 설명                                                                                   |
|---------------------|----------------------------------------------------------------------------------------|
| K-Fold 교차 검증    | 데이터를 K개의 폴드로 나누고, 각 폴드가 한 번씩 검증 데이터셋이 되어 K번 반복 평가[1][2][4][9]. |
| Stratified K-Fold   | K-Fold와 같지만, 각 폴드에 클래스 비율이 동일하게 유지되도록 분할(불균형 데이터에 효과적)[1][2]. |
| Leave-One-Out (LOOCV)| 데이터 한 개를 검증 데이터로, 나머지를 학습 데이터로 하여 N번(데이터 개수만큼) 반복 평가[2].    |
| Nested CV           | 바깥쪽과 안쪽에 각각 교차 검증을 적용하여 하이퍼파라미터 튜닝과 모델 평가를 동시에 진행[5].    |

#### K-Fold 교차 검증 예시

1. 데이터를 K개(예: 5개)로 나눕니다.
2. 각 폴드가 한 번씩 검증 데이터가 되고, 나머지 폴드는 학습 데이터가 됩니다.
3. 이 과정을 K번 반복하여 K개의 평가 결과를 얻고, 평균을 내어 최종 성능을 평가합니다[1][2][4][10].

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
model = DecisionTreeClassifier()
scores = cross_val_score(model, iris.data, iris.target, cv=5)
print('교차 검증별 점수:', scores)
print('평균 점수:', scores.mean())
```

#### 장점과 단점

| 장점                                         | 단점                                    |
|----------------------------------------------|-----------------------------------------|
| 데이터셋 전체를 학습/검증에 모두 활용 가능    | 반복 학습으로 인해 시간 소요가 큼        |
| 과적합/과소적합 탐지 및 일반화 성능 향상      | 데이터 분할 방법에 따라 성능 편차 발생   |
| 데이터가 적을 때도 효과적으로 평가 가능        | 시계열 데이터 등 순서가 중요한 경우 주의 |

#### 계층별(층화) K-Fold 교차 검증

- 데이터 클래스(레이블) 비율이 불균형할 때 각 폴드에 클래스 비율이 고르게 분포되도록 나누는 방식입니다.
- 예를 들어, 사기 거래가 전체의 5%라면 각 폴드에도 사기 거래가 5%씩 포함되도록 분할합니다[1][2].

#### 요약

- 교차 검증은 데이터를 여러 번 나누어 모델을 학습·평가하여, 성능을 평균적으로 평가하는 방법입니다.
- 대표적으로 K-Fold, Stratified K-Fold, LOOCV, Nested CV 등이 있습니다.
- 과적합 방지, 일반화 성능 평가, 데이터 부족 상황에서 효과적이지만, 반복 학습으로 시간이 더 걸릴 수 있습니다[1][2][4].

출처
[1] [Machine learning] 쉽게 설명하는 Cross Validation 교차검증 https://huidea.tistory.com/30
[2] [머신러닝] 교차검증 (Cross Validation) - velog https://velog.io/@soo_oo/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-Cross-Validation
[3] 교차 검증(Cross Validation) : 네이버 블로그 https://blog.naver.com/ckdgus1433/221599517834
[4] [기계학습] 교차검증(Cross Validation) https://gsbang.tistory.com/entry/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9DCross-Validation
[5] Nested cross validation - IBOK - 티스토리 https://bo-10000.tistory.com/85
[6] [Recommend] K-Fold Cross Validation(CV, 교차 검증)의 개념 - velog https://velog.io/@recoder/Cross-Validation
[7] [머신러닝] 크로스 밸리데이션(cross validation, 교차 검증)의 개념, 의미 https://losskatsu.github.io/machine-learning/cross-validation/
[8] [바람돌이/머신러닝] 교차검증(CV), Cross Validation, K-fold ... https://blog.naver.com/PostView.nhn?blogId=winddori2002&logNo=221850530979
[9] k-겹 교차 검증 - 인코덤, 생물정보 전문위키 https://incodom.kr/k-%EA%B2%B9_%EA%B5%90%EC%B0%A8_%EA%B2%80%EC%A6%9D
[10] [ML/DL] 교차 검증(Cross Validation) - K-Fold ... - Sonstory - 티스토리 https://sonstory.tistory.com/29

### [데이터 불균형 처리](https://github.com/zoro0rkd/ai_study/wiki/AI-모델-튜닝-%E2%80%90-NEW#3-클래스-불균형class-imbalanced-문제-해결)
- 데이터 불균형이란 특정 클래스(주로 소수 클래스)의 샘플 수가 매우 적어 학습 시 모델이 다수 클래스에 치우쳐 성능 저하를 일으키는 문제를 의미합니다.
- 주요 처리 방법으로는  
  - **오버샘플링(Over-sampling)**: 소수 클래스 데이터를 인위적으로 복제하거나 생성(SMOTE 같은 합성 샘플 생성 기법 포함)하여 균형 맞추기  
  - **언더샘플링(Under-sampling)**: 다수 클래스 데이터를 줄여 소수 클래스와 균형 맞추기 (정보 손실 가능성 주의)  
  - **앙상블 기법**: 여러 모델의 예측을 통합해 불균형 영향을 완화 (예: 랜덤 포레스트, 부스팅)  
  - **비용 민감 학습(Cost-sensitive Learning)**: 소수 클래스 오류에 더 큰 가중치를 부여하여 학습 시 반영  
  - **데이터 수집 확대**: 추가 데이터 확보로 클래스 불균형 완화
- 적절한 평가지표(재현율, 정밀도, F1 점수 등)를 함께 사용해야 실질적 성능 향상을 확인할 수 있음[1][4][7][8].

### [데이터 증강](https://github.com/zoro0rkd/ai_study/wiki/데이터-증강-%E2%80%90-NEW)과 [전처리](https://github.com/zoro0rkd/ai_study/wiki/데이터-정제-%E2%80%90-NEW)의 학습 영향
- 데이터 증강(Data Augmentation)은 기존 학습 데이터를 변형(회전, 자르기, 잡음 추가 등)하여 다양한 학습 사례를 만들어내는 기법으로, 모델 일반화 성능 개선에 필수적입니다.
- 특히 이미지는 물론 텍스트, 음성 등 다양한 데이터 유형에 맞춘 증강 기법들이 발전하고 있음  
- 전처리(Preprocessing)는 노이즈 제거, 이상치 처리, 정규화, 토큰화 같은 데이터의 품질을 향상시키는 작업으로, 데이터 품질이 높을수록 학습 성능 향상 및 안정성 확보 가능  
- 잘 설계된 증강과 전처리 과정은 데이터 부족 및 불균형 문제를 완화하고, 과적합 방지, 학습 효율 개선에 기여함

요약하면, 데이터 불균형 처리와 적절한 증강 및 전처리는 모델이 다양한 상황에서 안정적으로 작동하도록 하는데 필수적이며, 최종 성능 향상을 위한 중요한 초석입니다.

출처
[1] 불균형 데이터 (imbalanced data) 처리를 위한 샘플링 기법 https://casa-de-feel.tistory.com/15
[2] [논문]머신러닝을 위한 불균형 데이터 처리 방법 https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201900937327481
[3] 머신러닝 데이터 세트의 불균형 클래스와 싸우기 위한 8가지 ... https://www.nepirity.com/blog/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
[4] 불균형 데이터(Imbalanced Data) 머신러닝 Classification ... https://datasciencediary.tistory.com/entry/%EB%B6%88%EA%B7%A0%ED%98%95-%EB%8D%B0%EC%9D%B4%ED%84%B0-Imbalanced-Data-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Classification-%EB%AC%B8%EC%A0%9C%EC%A0%90-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95
[5] 머신러닝을 위한 불균형 데이터 처리 방법 : 샘플링을 위주로 https://koreascience.kr/article/JAKO201900937327481.pdf
[6] 불균형 데이터(imbalanced data) https://raziel.oopy.io/12a6fa0e-30de-80cd-9a8a-d05782acc94b
[7] 불균형 데이터 처리, 언더 샘플링, 오버 샘플링 https://eewjddms.tistory.com/87
[8] 머신러닝에서의 핵심 전략들 Imbalanced Dataset - 데이터 AI 벌집 https://datasciencebeehive.tistory.com/76
[9] 머신러닝을 위한 불균형 데이터 처리 방법 : 샘플링을 위주로 https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002527922
[10] 불균형 데이터(imbalanced data)란 무엇이고, 무엇이 문제인가? https://rfriend.tistory.com/773

## 3. 모델 학습 과정
### 초기화 (Weight Initialization)
신경망에서 **Weight Initialization(가중치 초기화)**는 각 층의 가중치를 학습 시작 전에 어떤 값으로 정할지 결정하는 과정입니다. 주요 목적은 학습의 효율성을 높이고, 기울기 소실이나 기울기 폭주 같은 문제를 방지하는 데 있습니다.

#### 왜 가중치 초기화가 중요한가?

- 모든 가중치를 **0**이나 동일한 값으로 초기화하면, 각 뉴런이 똑같이 학습되어 대칭성이 깨지지 않아 모델이 무의미해집니다.
- 가중치가 너무 크거나 작게 설정되면, 역전파를 할 때 기울기가 0에 가까워지거나(기울기 소실), 폭발적으로 커져서 학습이 불안정해질 수 있습니다.

#### 대표적인 가중치 초기화 방법과 예시

##### 1. **균등 분포/정규 분포 무작위 초기화**
- 각 가중치를 `-a`에서 `a` 사이(균등 분포) 또는 평균 0, 표준편차 σ(정규 분포)로 랜덤하게 설정합니다.
- 예시:
  - **Uniform:** $$ w \sim U(-0.05, 0.05) $$
  - **Normal:** $$ w \sim N(0, 0.01^2) $$
- 장점: 간단하고 빠름.
- 단점: 네트워크가 깊어질수록 기울기 문제가 생길 수 있음.

##### 2. **Xavier 초기화 (Glorot Initialization)**
- 주로 **Sigmoid, Tanh** 활성화 함수에서 사용.
- 각 층의 입력 개수($$n_{in}$$)와 출력 개수($$n_{out}$$)에 따라 분포의 범위를 조정하여, 신호가 각 층에서 적절히 전달되도록 함.
- 예시:
  - **균등분포:** $$ w \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right) $$
  - **정규분포:** $$ w \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right) $$

##### 3. **He 초기화 (Kaiming Initialization)**
- **ReLU** 계열 활성화 함수 사용할 때 적합.
- 입력 노드 개수($$n_{in}$$)만 고려해서 더 큰 분산을 사용, ReLU가 많은 값을 0으로 만들 수 있기 때문.
- 예시:
  - **정규분포:** $$ w \sim N\left(0, \frac{2}{n_{in}}\right) $$
  - **균등분포:** $$ w \sim U\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right) $$

##### 4. **Bias(바이어스) 초기화**
- 바이어스는 보통 0으로 초기화. 특별한 경우를 제외하면 문제 발생이 적음.

#### 핵심 정리

- **가중치 초기화**는 신경망 학습의 첫걸음이며, 성능과 수렴 속도에 직접적인 영향을 줍니다.
- 네트워크 구조와 활성화 함수에 맞는 초기화 방식을 선택해야 효과적입니다.
- 실제로는 딥러닝 프레임워크(파이토치, 텐서플로우 등)에서 함수로 쉽게 적용할 수 있습니다.

잘 설계된 가중치 초기화가 학습의 시작과 성능을 좌우한다는 점이 가장 중요합니다.

참고 : https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/


### 순전파(Forward Propagation)
- 순전파는 신경망에 입력값이 들어와 각 층을 거쳐 출력층까지 전달되는 과정입니다.
- 입력 데이터에 가중치(weight)를 곱하고 편향(bias)을 더한 뒤 활성화 함수(activation function)를 거치면서 다음 층으로 신호를 전달합니다.
- 이 과정에서 모델이 예측하는 출력값이 계산됩니다.
- 순전파는 입력에서 출력으로의 데이터 흐름이며, 모델의 예측을 생성하는 본질적인 단계입니다.

### 활성 함수(Activation Function)
#### Step Function을 활성화 함수로 사용하지 않는 이유

##### 1. 미분 불가능성과 기울기 소실 문제
- 딥러닝 학습은 **경사 하강법(gradient descent)**과 **역전파(backpropagation)**에 기반함.
- 역전파를 위해선 활성화 함수가 **미분 가능**해야 함.
- 하지만 **Heaviside step function**은:
  - \( x = 0 \)에서 **미분 불가능**
  - 그 외의 구간에서는 도함수가 **0** → **기울기 소실(vanishing gradient)** 문제 발생
- 따라서 가중치를 효과적으로 **업데이트할 수 없음** → 학습이 불가능

##### 2. 이산적인 출력으로 인한 최적화 어려움
- 신경망은 **출력이 실제값에 가까워지도록** 가중치와 편향을 조정함.
- 이를 위해선 **작은 가중치 변화**가 출력에 **연속적인 영향을 줘야** 함.
- 그러나 step 함수는 출력이 **0 또는 1**로 이산적(discrete)이므로:
  - 입력 변화에 따른 출력 변화가 **불연속적**
  - **경사 기반 최적화**(gradient-based optimization)에 부적합

##### 3. ReLU와의 비교: 부분 미분 가능성
- **ReLU(Rectified Linear Unit)**도 \( x = 0 \)에서 미분 불가능하지만:
  - 대부분 구간에서 **미분 가능**
  - **부분 도함수(subderivative)** 등을 통해 학습 가능
- 반면, step 함수는 거의 모든 구간에서 도함수가 0 → 실질적 학습 불가

---

##### ✅ 결론
> **Step function은 출력이 이산적이고 도함수가 대부분 0이므로, 딥러닝에서의 학습에 적합하지 않음.**  
> 따라서 **ReLU, sigmoid, tanh** 등 **연속적이며 미분 가능한 함수**들이 활성화 함수로 사용됨.

#### 주요 Activation Function(활성화 함수) 장단점 비교표

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ZafDv3VUm60Eh10OeJu1vw.png)


| 함수명          | 수식/출력 범위        | 장점                                                                 | 단점                                                                  | 주요 사용처/특징                         |
|----------------|----------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|------------------------------------------|
| **Step**       | 0 또는 1             | 구현이 간단, 이진 분류에 직관적                                      | 미분 불가(학습 불가), 다중 클래스 불가, gradient=0으로 역전파 불가      | 고전적 퍼셉트론, 실전에서는 거의 사용 안 함 |
| **Linear**     | 실수 전체             | 회귀 문제에 적합, 출력 해석 용이                                     | 비선형성 없음, 레이어 쌓아도 단일 선형함수와 동일, 학습력 제한         | 회귀 문제 출력층, 은닉층에는 사용 안 함    |
| **Sigmoid**    | (0, 1)               | 출력값을 확률처럼 해석 가능, 이진 분류 출력층                        | gradient vanishing(기울기 소실), 출력 분포가 0에 치우침, 느린 수렴      | 이진 분류 출력층, 은닉층에는 잘 안 씀      |
| **Tanh**       | (-1, 1)              | 출력 평균이 0에 가까워 학습 빠름, Sigmoid보다 gradient vanishing 덜함 | 여전히 gradient vanishing 있음, 출력 saturate 구간에서 학습 느림        | RNN 등 일부 구조 은닉층                   |
| **ReLU**       | [0, ∞)               | 계산 단순, 빠른 학습, gradient vanishing 적음, deep network에 효과적   | 입력<0에서 gradient=0(죽은 뉴런 문제), 음수 입력 무시                  | CNN, MLP 등 대부분의 은닉층               |
| **Leaky ReLU** | x<0: αx, x≥0: x      | ReLU의 죽은 뉴런 문제 완화, 음수 영역도 gradient 존재                | α값 선정 필요, 여전히 완벽하지 않음                                   | ReLU 대체, 죽은 뉴런 방지 목적             |
| **Softmax**    | (0, 1), 합=1         | 다중 클래스 확률 분포 출력, 각 클래스 확률 해석                      | gradient vanishing, 출력값이 0/1에 가까워질수록 학습 어려움            | 다중 클래스 분류 출력층                   |
| **Swish/GELU** | 비선형, 실수 전체     | ReLU보다 부드러운 비선형성, deep network에서 성능 우수                | 계산 복잡, 최신 네트워크에서 주로 사용                                 | 최신 deep network, BERT 등                |

---

#### 참고 및 요약

- **ReLU**: 현재 가장 널리 사용되는 은닉층 활성화 함수로, 계산이 간단하고 gradient vanishing 문제가 적어 deep learning에 최적[1][4][5][6].
- **Sigmoid, Tanh**: 과거에는 많이 썼으나, gradient vanishing 문제가 심해 최근에는 출력층에만 사용[1][4][5].
- **Leaky ReLU, Swish, GELU 등**: ReLU의 단점을 보완하거나 deep network에서 성능을 높이기 위해 고안된 함수들[5][6].
- **Softmax**: 다중 클래스 분류의 출력층에서 표준적으로 사용[1][5].
- **Linear**: 회귀 문제의 출력층에만 사용, 은닉층에는 사용하지 않음[1][5].

> "ReLU is the top choice as it is simpler, faster, much lower run time, better convergence performance and does not suffer from vanishing gradient issues"[5].

> "Sigmoid/Logistic and Tanh functions should not be used in hidden layers as they make the model more susceptible to problems during training (due to vanishing gradients)"[1].

---

#### 상황별 권장 함수

- **은닉층**: ReLU 계열(Leaky ReLU, PReLU 등), 일부 RNN은 Tanh/Sigmoid
- **출력층**: 회귀(Linear), 이진분류(Sigmoid), 다중분류(Softmax), 멀티라벨(Sigmoid)

---

출처
[1] Activation Functions in Neural Networks [12 Types & Use Cases] https://www.v7labs.com/blog/neural-networks-activation-functions
[2] [PyTorch] PyTorch가 제공하는 Activation function(활성화함수) 정리 https://sanghyu.tistory.com/182
[3] [PDF] Review and Comparison of Commonly Used Activation Functions for ... https://arxiv.org/pdf/2010.09458.pdf
[4] Why is Relu considered superior compared to Tanh or sigmoid? https://www.reddit.com/r/learnmachinelearning/comments/ua6n6s/why_is_relu_considered_superior_compared_to_tanh/
[5] What is an activation function? What are the different types of ... https://aiml.com/what-is-an-activation-function-what-are-the-different-types-of-activation-functions-discuss-their-pros-and-cons/
[6] 활성화 함수(activation function) 종류와 정리 - PGNV 계단 - 티스토리 https://pgnv.tistory.com/17
[7] Activation functions in neural networks [Updated 2024] https://www.superannotate.com/blog/activation-functions-in-neural-networks
[8] Top 10 Activation Function's Advantages & Disadvantages - LinkedIn https://www.linkedin.com/pulse/top-10-activation-functions-advantages-disadvantages-dash
[9] Activation Functions: Advantages & Disadvantages - YouTube https://www.youtube.com/watch?v=uomZz2pckOA
[10] 7 Common Nonlinear Activation Functions (Advantage and ... - Kaggle https://www.kaggle.com/getting-started/157710

### 손실함수 (Loss Function)

손실함수는 머신러닝 및 딥러닝 모델 학습 과정에서, 모델이 예측한 값과 실제 정답 사이의 차이를 수치적으로 평가하는 함수입니다. 이 값이 작다는 것은 모델의 예측이 실제 값에 가깝다는 것을 의미하며, 이 함수를 최소화하는 방향으로 모델을 학습합니다. 손실함수는 최적화 과정에서 경사 하강법과 같은 알고리즘이 가중치 조정을 수행할 때 매우 중요한 역할을 합니다.

#### 대표적인 손실함수 정리

| 손실함수명                | 적용 문제 유형         | 정의 및 설명                                                                                   | 특징                                                 |
|--------------------------|-----------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------|
| Mean Squared Error (MSE) | 회귀                  | 예측값과 실제값 차이의 제곱을 평균한 값                                                    | 큰 오차에 더 큰 페널티를 부여, 이상치에 민감          |
| Mean Absolute Error (MAE) | 회귀                  | 예측값과 실제값 절대 차이의 평균                                                           | 이상치에 덜 민감, 더 안정적인 성능 제공               |
| Binary Cross Entropy      | 이진 분류              | 예측 확률과 실제 이진 클래스 간의 차이를 측정 (로그 손실)                                  | 확률 기반 손실, 이진 분류에 적합                       |
| Categorical Cross Entropy | 다중 클래스 분류       | 다중 클래스에 대해 예측 확률 분포와 실제 분포 간의 차이를 측정                            | softmax 출력 및 one-hot 라벨 대응에 적합              |
| Huber Loss                | 회귀                  | MSE와 MAE의 장점을 결합, 작은 오차는 제곱, 큰 오차는 절대값으로 처리                   | 이상치에 강인하며 안정적인 학습 가능                   |
| Hinge Loss                | 분류 (SVM 등)          | 올바른 클래스 점수와 최대 마진을 고려하는 손실 함수                                      | 마진 기반 분류에 적합, 서포트 벡터 머신에서 주로 사용 |

손실 함수는 문제 유형과 데이터 특성에 따라 적합한 함수를 선택하는 것이 매우 중요하며, 이를 통해 모델의 학습 방향성과 최종 성능이 크게 좌우됩니다.

출처
[1] 딥러닝 개발자라면 꼭 알아야 할 손실 함수 의 개념과 종류 https://modulabs.co.kr/blog/machine_learning_loss_function
[2] 손실함수 (Loss Function) | 블로그 - 모두의연구소 https://modulabs.co.kr/blog/loss-function-machinelearning
[3] 손실 함수란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/loss-function
[4] [Machine Learning] 손실 함수 (loss function) https://insighted-h.tistory.com/7
[5] 손실함수의 이해와 종류/파이썬으로 구현까지 https://my-inote.tistory.com/164
[6] loss function 종류 - velog https://velog.io/@jh_ds/loss-function-%EC%A2%85%EB%A5%98
[7] 손실 함수(Loss Function) 개념 및 종류 - learningflix - 티스토리 https://learningflix.tistory.com/128
[8] 10. 손실 함수(Loss Function) - 소설처럼 읽는 딥러닝 - 위키독스 https://wikidocs.net/277027
[9] [Theory] 손실함수(Loss function)의 통계적 분석 https://hyewonleess.github.io/theory/loss-function/
[10] [딥러닝] 손실함수 (loss function) 종류 및 간단 정리 (feat. keras ... https://didu-story.tistory.com/27


### 역전파 (Backpropagation)
- 역전파는 모델의 예측값과 실제값 사이의 오차(loss)를 기반으로 가중치를 업데이트하는 과정입니다.
- 출력층에서부터 입력층 방향으로 오차를 거꾸로 전파하며, 각 가중치가 손실 함수에 어느 정도 기여하는지 미분(기울기)를 계산합니다.
- 이때 연쇄법칙(chain rule)을 활용해 미분값을 효율적으로 구하며, 이를 통해 가중치를 학습률(learning rate)에 따라 조정합니다.
- 역전파는 신경망을 학습시키는 핵심 알고리즘으로, 손실을 최소화하도록 모델을 최적화합니다.

요약하면, 순전파는 데이터가 입력에서 출력으로 전달되어 예측을 계산하는 과정이고, 역전파는 예측 오차를 거꾸로 전파하여 가중치를 조정하는 학습 과정입니다. 두 과정이 반복되며 신경망은 점점 더 정확한 예측을 할 수 있도록 학습됩니다.

출처
[1] 순전파 & 역전파 - velog https://velog.io/@tnsida315/%EC%88%9C%EC%A0%84%ED%8C%8C-%EC%97%AD%EC%A0%84%ED%8C%8C
[2] 07-05 역전파(BackPropagation) 이해하기 - 위키독스 https://wikidocs.net/37406
[3] [딥러닝개론] 순전파와 역전파 - Jini Dev https://yujindevv.tistory.com/7
[4] 신경망 (3) - 역전파 알고리즘(BackPropagation algorithm) https://yhyun225.tistory.com/22
[5] 역전파란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/backpropagation
[6] 딥러닝에 대한 이해와 순전파, 역전파 직접 구현 https://coco0414.tistory.com/44

### 최적화(Optimization)
#### SGD, Adam, RMSprop
![](https://statusneo.com/wp-content/uploads/2023/09/Credit-Analytics-Vidya.jpg)

#### Batch, Mini-Batch, Stochastic Gradient Descent 비교표

| 구분                | Batch Gradient Descent (배치)         | Mini-Batch Gradient Descent (미니배치)                | Stochastic Gradient Descent (확률적/SGD)   |
|---------------------|--------------------------------------|-----------------------------------------------------|------------------------------------------|
| **Batch Size**      | 전체 데이터셋                        | 1 < 미니배치 크기 < 전체 데이터셋                      | 1                                        |
| **업데이트 빈도**   | 에포크마다 1회 (전체 데이터로 1회)    | 미니배치마다 1회 (여러 번)                             | 샘플마다 1회 (가장 빈번)                  |
| **속도**            | 느림                                 | 빠름 (GPU 등 벡터화 활용 가능)                         | 빠름 (업데이트는 빠르나, 전체적으로 느릴 수 있음) |
| **노이즈/진동**     | 거의 없음 (매끄러운 경로)             | 중간 (적당한 진동, 노이즈는 일부 있음)                  | 큼 (진동 심함, 경로가 불규칙)             |
| **메모리 사용량**   | 큼                                   | 중간                                                  | 적음                                     |
| **장점**            | 정확한 gradient, 수렴 경로 안정적     | 속도와 안정성의 균형, 하드웨어 최적화, 실무에서 주로 사용 | 빠른 업데이트, local minima 탈출 가능성    |
| **단점**            | 느린 학습, 대용량 데이터에 부적합     | 하이퍼파라미터(배치 크기) 조정 필요, local minima에 갇힐 수 있음 | 노이즈 큼, 수렴 경로 불안정, 벡터화 어려움 |
| **사용 예시**       | 소규모 데이터, 이론적 분석            | 대부분의 딥러닝/머신러닝 실무                           | 온라인 학습, 실시간 데이터 처리           |

---

##### 추가 설명

- **Batch Gradient Descent**: 전체 데이터셋을 한 번에 사용해 gradient를 계산하고 파라미터를 업데이트합니다. 계산은 정확하지만, 데이터가 많아질수록 한 번의 업데이트에 시간이 오래 걸립니다[1][2][4][6].
- **Stochastic Gradient Descent (SGD)**: 매번 하나의 샘플만 사용해 파라미터를 업데이트합니다. 업데이트가 매우 빈번하고 빠르지만, 경로가 매우 불규칙하고 노이즈가 많아 최적점 근처에서 진동이 심할 수 있습니다[1][2][4][6].
- **Mini-Batch Gradient Descent**: 전체 데이터를 여러 개의 작은 배치로 나누어 각 배치마다 gradient를 계산해 파라미터를 업데이트합니다. 대부분의 딥러닝 프레임워크에서 기본적으로 사용하며, GPU 등 벡터 연산에 최적화되어 있고, 속도와 안정성의 균형이 좋아 실무에서 가장 많이 쓰입니다[1][2][3][4][6].

---

##### 요약

- **Batch**: 전체 데이터로 한 번에 업데이트 → 느리지만 안정적
- **SGD**: 샘플 하나로 업데이트 → 빠르지만 불안정
- **Mini-Batch**: 일부 샘플(수십~수백 개)로 업데이트 → 빠르고 안정적, 실무 표준

출처
[1] [Optimization Algorithms] Gradient Descent (1) Batch, Stochastic ... https://geniewishescometrue.tistory.com/entry/Gradient-Descent
[2] Batch vs Mini batch vs Stochastic - Woongjoon_AI - Choi Woongjoon https://woongjoonchoi.github.io/optimization/Batch-Size/
[3] Mini-batch Gradient Descent(미니배치 경사 하강법) - 정리 - 티스토리 https://better-tomorrow.tistory.com/entry/Mini-batch-Gradient-Descent%EB%AF%B8%EB%8B%88%EB%B0%B0%EC%B9%98-%EA%B2%BD%EC%82%AC-%ED%95%98%EA%B0%95%EB%B2%95
[4] Gradient Descent(경사하강법) 와 SGD( Stochastic ... - 매일매일 딥러닝 https://everyday-deeplearning.tistory.com/entry/SGD-Stochastic-Gradient-Descent-%ED%99%95%EB%A5%A0%EC%A0%81-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95
[5] Batch gradient descent versus stochastic gradient descent https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent
[6] Differences Between Gradient, Stochastic and Mini Batch Gradient ... https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch
[7] [ML]Gradient Descent 의 세 종류(Batch, Stochastic, Mini-Batch) - velog https://velog.io/@crosstar1228/MLGradient-Descent-%EC%9D%98-%EC%84%B8-%EC%A2%85%EB%A5%98Batch-Stochastic-Mini-Batch
[8] 경사하강법 Batch/Stochastic/Mini-Batch Gradient Descent (BGD ... https://ju-blog.tistory.com/60
[9] 딥러닝 용어정리, MGD(Mini-batch gradient descent), SGD ... - All about https://light-tree.tistory.com/133
[10] BGD: Batch Gradient Descent (배치 경사 하강법) - 위키독스 https://wikidocs.net/200934

### 학습률(Learning rate)
* **학습률(learning rate)은 경사하강법에서 손실 함수의 기울기(gradient)를 따라 파라미터를 얼마나 이동시킬지를 결정하는 값**
* 너무 작으면 학습 속도가 느려지고, 너무 크면 발산하거나 최적값에 도달하지 못함

#### **수식 이해**
딥러닝의 가중치 업데이트 수식:
```
w = w - η * ∇L(w)
```
- w: 모델의 가중치
- η (eta): **learning rate**
- ∇L(w): 손실 함수 L에 대한 가중치의 기울기

#### learning Rate 의 영향
* 너무 작을 때: 학습 속도가 매우 느리며, local minima에 갇힐 수 있음
* 너무 클 때: 손실 값이 발산하거나 최적점을 지나쳐 계속 진동하여 수렴 실패

#### 관련 기법
* **고정 학습률 (Fixed LR)**
  * 일정한 값을 처음부터 끝까지 사용하는 방식
* **Learning Rate Decay (감쇠)**
  * 학습이 진행됨에 따라 학습률을 **점점 줄이는** 전략
  * 예: Step decay, Exponential decay, Cosine annealing
* **Warm-up**
  * 초기 학습률을 매우 작게 시작하고, 일정 epoch 동안 점차 증가
  * 안정적인 초기 학습에 도움
* **Adaptive Methods**
  * 학습률을 자동으로 조정하는 옵티마이저
  * 예: Adam, RMSprop, Adagrad 등
* **Cyclical Learning Rate**
  * 일정 주기로 학습률을 증가·감소시키며 지역 최적점 탈출 유도

[Learning rate scheduler 정리](https://gaussian37.github.io/dl-pytorch-lr_scheduler/)

### 정규화(Normalization)
#### Normalization이란?
Normalization(정규화)은 **뉴럴 네트워크의 학습과정에서 데이터의 분포를 일정하게 맞추는 과정**입니다. 입력 값들의 스케일을 비슷하게 맞추어서 학습의 안정성, 속도를 높이고, 기울기 소실/폭주 문제를 완화하며 일반화 성능도 향상됩니다[1][2].

#### 일반적인 목적과 효과
- 학습 속도 향상 및 안정화
- 내부 공변량 변화(Internal Covariate Shift) 완화
- 큰/작은 값에 의한 학습률 저하 방지
- 각 feature(특징)의 중요도 균형화

#### 대표적인 Normalization 종류
아래 표는 신경망에서 자주 쓰이는 LayerNorm, BatchNorm 외 주요 Normalization 기법을 정리한 것입니다.

<img width="960" height="252" alt="1*gat8a-TUnopoYN_veGEi0w" src="https://github.com/user-attachments/assets/1a653d45-9886-4868-9ebe-eeea6a45a55f" />

| 구분               | 정규화 방식                       | 정규화 축                       | Batch 크기 의존성 | 적용 위치                         | 대표 활용 분야                 | 특징 및 장단점                                                                                     |
|--------------------|-----------------------------------|----------------------------------|-------------------|------------------------------------|-------------------------------|----------------------------------------------------------------------------------------------------|
| **BatchNorm**      | (x-평균)/표준편차                  | Batch 차원(전체 샘플별 feature별)| O                 | Layer와 Activation 사이            | CNN, 일반 MLP                 | 빠른 수렴, 대형 Batch 필요, 추론 시 moving avg 사용[3][4][5]                                        |
| **LayerNorm**      | (x-평균)/표준편차                  | Feature 차원(샘플별 전체 feature)| X                 | 주로 입력/출력(Transformer 등)     | Transformer, RNN               | Batch 크기 무관, 소규모 Batch, NLP/시계열, 실시간 추론 적합[6][7][4][8]                            |
| **InstanceNorm**   | (x-평균)/표준편차                  | 각 샘플, 각 채널 별               | X                 | Conv layer 출력 등                 | 스타일 트랜스퍼, 이미지         | 이미지별 채널 정규화, 스타일 변화에 유리                                                           |
| **GroupNorm**      | (x-평균)/표준편차                  | 채널을 여러 그룹으로 나눠 정규화  | X                 | Conv layer 출력 등                 | 컴퓨터 비전(CNN)               | 소규모-대규모 Batch 모두 유연, Group 수 조절로 Batch/LayerNorm 절충[4]                             |
| **RMSNorm**        | RMS(제곱평균근)로만 정규화          | Feature 차원(평균 미사용)         | X                 | Transformer 등                    | 대규모 LLM(언어모델)            | 계산 비용 낮고, Mean subtraction 없음(최근 논문에서 성능비교)[9]                                   |

#### Batch Normalization (BatchNorm) 요약

- 미니배치 단위로 각 feature(채널)의 평균과 분산을 계산, 정규화 후 학습 가능 파라미터로 scale/shift를 적용.
- 학습 시 batch 통계 사용, 추론 시엔 전체 데이터 통계 적용.
- **장점**: 빠른 수렴, 일반화 성능 향상, 깊은 네트워크에 효과적[3][4][10].
- **단점**: Batch size에 민감, 순환/트랜스포머 모델/실시간 처리엔 적합하지 않음.

#### Layer Normalization (LayerNorm) 요약

- 각 샘플 별 feature들을 정규화(즉, 한 샘플 벡터 전체의 평균, 분산 활용).
- **Batch size의 영향 없이** 사용 가능해 NLP, RNN, Transformer 등에 적합.
- **장점**: 작은 batch, 시퀀스 모델 등에서 우수, 실시간에 적합[6][7][8].
- **단점**: CNN에서 BatchNorm만큼 효과적이지 않을 때도 있음.

#### 참고: 그 외 Normalization

- **InstanceNorm**: 주로 이미지 생성에 사용, 스타일 변화에 강건.
- **GroupNorm**: Group 단위로 정규화하여 Batch, LayerNorm의 단점을 보완.
- **RMSNorm**: Mean을 빼지 않고 RMS(Root Mean Square)만 사용해 계산 효율적[9].

#### 요약
- **Normalization은 신경망 학습의 핵심 도구**로, 데이터의 분포를 통제하여 학습을 빠르고 안정적으로 만든다.
- **BatchNorm**은 대형 Batch에서 효과적이며, **LayerNorm**은 batch size와 무관하여 Transformer/RNN 등에서 주로 쓰인다.
- 모델 구조 및 목적에 따라서 적합한 Normalization 기법을 선택해야 한다[2][4][7].

**참고 문헌:**
내용은 최신 논문, PyTorch 공식문서, 학습 사이트 등 다양한 신뢰성 있는 자료 기반으로 작성되었습니다[1][2][6][7][3][4][8][10].

출처
[1] Normalizing Inputs for an Artificial Neural Network https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
[2] Using Normalization Layers to Improve Deep Learning ... https://www.machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/
[3] Batch Normalization (BatchNorm) Explained | Ultralytics https://www.ultralytics.com/glossary/batch-normalization
[4] Build Better Deep Learning Models with Batch and Layer ... - Pinecone https://www.pinecone.io/learn/batch-layer-normalization/
[5] 8.5. Batch Normalization - Dive into Deep Learning http://d2l.ai/chapter_convolutional-modern/batch-norm.html
[6] LayerNorm — PyTorch 2.7 documentation https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
[7] Batch vs Layer Normalization - Zilliz Learn https://zilliz.com/learn/layer-vs-batch-normalization-unlocking-efficiency-in-neural-networks
[8] What is Layer Normalization? - GeeksforGeeks https://www.geeksforgeeks.org/deep-learning/what-is-layer-normalization/
[9] Re-Introducing LayerNorm: Geometric Meaning, Irreversibility and a ... https://arxiv.org/html/2409.12951v1
[10] Batch normalization - Wikipedia https://en.wikipedia.org/wiki/Batch_normalization
[11] Normalization (machine learning) https://en.wikipedia.org/wiki/Normalization_(machine_learning)
[12] Normalization Techniques in Training DNNs https://arxiv.org/pdf/2009.12836.pdf
[13] Normalizing data for better Neural Network performance https://www.youtube.com/watch?v=jL4cs5EZuO4
[14] Normalization effects on deep neural networks https://www.aimsciences.org/article/doi/10.3934/fods.2023004
[15] Batch Normalization VS Layer Normalization - 주홍색 코딩 - 티스토리 https://kwonkai.tistory.com/144
[16] Neural Network 적용 전에 Input data를 Normalize 해야 하는 이유 https://goodtogreate.tistory.com/entry/Neural-Network-%EC%A0%81%EC%9A%A9-%EC%A0%84%EC%97%90-Input-data%EB%A5%BC-Normalize-%ED%95%B4%EC%95%BC-%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0
[17] Batch Norm Explained Visually - How it works, and why neural ... https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739/
[18] Why do transformers use layer norm instead of batch norm? https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
[19] Should I normalize all data prior feeding the neural network ... https://stats.stackexchange.com/questions/458579/should-i-normalize-all-data-prior-feeding-the-neural-network-models
[20] LayerNorm 과 BatchNorm 의 차이 - Embedded World - 티스토리 https://docon.tistory.com/22

### Regularization

머신러닝에서 **Regularization**는 모델이 학습 데이터에 과도하게 적합하여 새로운 데이터에 대해 성능이 떨어지는 과적합(overfitting)을 방지하기 위해 모델의 복잡도를 제한하거나 패널티를 주는 기법입니다. 대표적인 규제 방법에는 **Dropout**과 **L1/L2 정규화**가 있습니다.

***

#### Dropout

- **개념**: 학습 과정에서 신경망의 일부 뉴런을 임의로 선택하여 일시적으로 비활성화(즉, 출력을 0으로 만듦)하는 방법입니다.
- **목적**: 뉴런 간 공동 적응(co-adaptation)을 방지하고, 다양한 뉴런 조합에서 학습하게 함으로써 모델이 보다 일반화할 수 있도록 돕습니다.
- **작동 방식**: 학습 시 각 층의 뉴런을 확률적으로 드롭(꺼짐)하며, 테스트 시에는 모든 뉴런을 활성화하고 출력에 확률값을 반영해 보정합니다.
- **효과**: 과적합 억제와 모델 일반화 성능 개선에 매우 효과적이며, 복잡한 신경망에서 자주 사용됩니다.

***

#### L1/L2 정규화

- **개념**: 손실 함수(loss function)에 모델 파라미터(가중치)에 대한 페널티 항을 추가하는 기법으로, 가중치가 너무 커지지 않도록 조절하여 과적합을 방지합니다.

- **L1 정규화 (Lasso)**
  - 가중치의 절대값 합을 페널티로 추가합니다.
  - 특징: 일부 가중치를 정확히 0으로 만들어 희소한(sparse) 모델을 생성해 중요 특성만 선택하는 효과가 있습니다.
  - 장점: 특성 선택(feature selection)에 적합하며, 이상치(outlier)에 대해 더 강건함(robust).
  - 단점: 수학적으로 0에서 미분이 불가능해 최적화에 주의가 필요하며, 규제 강도가 약할 수 있음.

- **L2 정규화 (Ridge)**
  - 가중치의 제곱합을 페널티로 추가합니다.
  - 특징: 모든 가중치를 균등하게 작게 유지하려 하며, 오버핏팅을 방지합니다.
  - 장점: 미분 가능하고 계산이 안정적이며, 대부분의 경우 L1보다 더 좋은 예측 성능을 냅니다.
  - 단점: 가중치를 완전히 0으로 만들지 않으므로 희소성은 제공하지 않음.
  - L2 정규화는 weight decay(가중치 감소)와 같은 효과를 가지며, 큰 가중치에 더 큰 패널티를 줍니다.

***

이 두 Regularization 기법은 종종 혼합하여 사용되기도 하며(예: Elastic Net), 모델의 성능과 일반화 능력을 크게 향상시키는 데 중요한 역할을 합니다. Dropout은 주로 딥러닝 모델에서 비선형 신경망의 과적합 방지에 활용되고, L1/L2 정규화는 전통적인 머신러닝과 딥러닝 모두 널리 적용됩니다.

출처
[1] [최적화] 가중치 규제 L1, L2 Regularization의 의미, 차이점 (Lasso ... https://seongyun-dev.tistory.com/52
[2] L1 Regularization & L2 Regularization - Everything - 티스토리 https://hyebiness.tistory.com/11
[3] L1 정규화, L2 정규화 https://esj205.oopy.io/4b321662-5d02-4559-8677-7e974cf080a8
[4] L1, L2 Norm, Loss, Regularization? - 생각정리 - 티스토리 https://junklee.tistory.com/29
[5] L1 & L2 loss/regularization - Seongkyun Han's blog https://seongkyun.github.io/study/2019/04/18/l1_l2/
[6] L1 Loss & L2 Loss - 효과는 굉장했다! - 티스토리 https://thflgg133.tistory.com/231
[7] Regularization(정규화/규제화) 기법 - Ridge(L2 norm) / LASSO(L1 ... https://bigdaheta.tistory.com/104
[8] [기술면접] L1, L2 regularization, Ridge와 Lasso의 차이점 (201023) https://huidea.tistory.com/154
[9] L1, L2 Regularization에 대하여 - Doby's Lab - 티스토리 https://draw-code-boy.tistory.com/502
[10] [CS231n] 2강. L1 & L2 distance - 룰루랄라 효니루 - 티스토리 https://bookandmed.tistory.com/27

### [Gradient clipping](https://sanghyu.tistory.com/87)
* Gradient Clipping은 기울기 폭주(gradient explosion) 현상을 방지하기 위한 기법
* 딥러닝 모델, 특히 RNN 또는 심층 네트워크 학습 시, 역전파 중 기울기가 지나치게 커지면 가중치가 발산하거나 NaN이 되는 문제가 발생
* Gradient Clipping은 특정 임계값을 초과하는 기울기의 크기를 **잘라내거나 재조정(clip)**하여 안정적인 학습 도움
* 대표적인 방식: norm 기준으로 클리핑 (torch.nn.utils.clip_grad_norm_ 등)

## 4. 학습 모니터링 및 개선

### 손실 및 정확도 곡선 모니터링
- 모델 학습 동안 손실 함수 값(loss)과 정확도(accuracy)를 에포크(epoch)별로 추적, 시각화하는 기법입니다.
- 손실 곡선이 지속적으로 감소하고 정확도 곡선이 증가하면 학습이 잘 진행되고 있는 신호입니다.
- 곡선의 변화를 통해 학습 속도, 과적합(overfitting) 여부, 학습 정체 상태 등을 파악할 수 있습니다.
- 대표적인 도구로 TensorBoard, Weights & Biases 등이 있으며, 실시간 모니터링 및 다양한 메트릭 시각화가 가능합니다.

### 조기 종료(Early Stopping)
- 검증(validation) 데이터에서 성능 개선이 일정 에포크 이상 없을 경우 학습을 미리 종료하는 기법
- 과적합을 방지하며, 불필요한 학습 시간을 줄이고 자원 효율성을 개선합니다.
- 구현 방식은 주로 검증 손실(validation loss) 또는 검증 정확도(validation accuracy)를 기준으로 하며, 이 값이 개선되지 않으면 정지 신호를 보냅니다.

### 자동화된 하이퍼파라미터 탐색 (AutoML)
- 모델 성능에 큰 영향을 미치는 하이퍼파라미터(학습률, 배치 크기, 네트워크 크기 등)를 자동으로 탐색, 최적화하는 기술
- 그리드 서치(Grid Search), 랜덤 서치(Random Search), 베이지안 최적화(Bayesian Optimization), 진화 알고리즘, 강화학습 등 다양한 방법이 활용됨
- AutoML은 사람이 직접 조정하던 많은 반복적이고 시간 소모적인 작업을 자동화하여, 효율적이고 최적화된 모델을 빠르게 찾을 수 있도록 지원
- 대표적인 프레임워크로는 Google AutoML, Microsoft Azure AutoML, open-source인 Auto-sklearn, Optuna 등이 있음

이처럼 학습 모니터링과 조기 종료, 그리고 AutoML은 모델 학습을 안정적으로 유지하고 최적 성능을 추구하기 위한 핵심 기술들입니다.

출처
[1] Azure Machine Learning 모니터링 https://learn.microsoft.com/ko-kr/azure/machine-learning/monitor-azure-machine-learning?view=azureml-api-2
[2] 모니터링을 통한 머신러닝 모델 정확도 유지 https://dev.to/yoon/moniteoringeul-tonghan-meosinreoning-model-jeonghwagdo-yuji-1k86
[3] 프로덕션 ML 시스템: 파이프라인 모니터링 | Machine Learning https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring?hl=ko
[4] 내 딥러닝 모델(학습)이 지나치게 느릴 때 점검해야할 사항들 https://heesunpark26.tistory.com/15
[5] 딥러닝 파이토치 교과서: 2.2.7 훈련 과정 모니터링 https://thebook.io/080289/0053/
[6] 배포 후 컴퓨터 비전 모델 유지 관리 https://docs.ultralytics.com/ko/guides/model-monitoring-and-maintenance/
[7] 봇 감지를 위한 머신 러닝 모델 모니터링 https://blog.cloudflare.com/ko-kr/monitoring-machine-learning-models-for-bot-detection/
[8] 모델 성능 모니터링을 통해 풀스택 옵저버빌리티를 머신 러닝 ... https://newrelic.com/kr/blog/nerdlog/ml-model-performance-monitoring
[9] 머신러닝 학습 모니터링 데이터 버저닝을 위한 clearml https://www.youtube.com/watch?v=pIXxdFBdSEM


### PFLOPS-days 절감 전략 선택을 위한 기본 개념 정리

- **PFLOPS-days**는 대규모 AI/머신러닝 트레이닝에서 사용하는 연산량 단위로, 1초에 1펫타플롭스(= $$10^{15}$$ FLOPS)의 계산 속도를 하루(86,400초) 동안 수행했을 때의 총 연산량을 의미합니다.
- 예) “3,000 PFLOPS-days”는 1 PFLOPS의 컴퓨팅 자원으로 3,000일, 혹은 3,000 PFLOPS의 클러스터에서 1일간 작업한 총 연산량입니다[1][2][3].

#### 총학습 비용(PFLOPS-days)에 영향을 주는 요소

| 요소                | 설명                                                                                                                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **모델 크기**        | 파라미터·레이어 수가 클수록 연산량 폭증                                                                                                          |
| **데이터셋 규모**     | 학습에 투입하는 데이터 양(토큰 수, 샘플 수)이 많을수록 연산 필요량 증가                                                                                  |
| **학습 반복(epoch)** | 반복 학습 횟수가 늘수록 전연산량(PFLOPS-days) 증가                                                                                         |
| **하드웨어 성능 및 효율** | 실제 GPU/TPU의 성능, 활용률(이론치 대비 실제 연산 효율; 통신, 자원 미활용 등으로 효율이 감소)                                                         |
| **알고리즘과 최적화** | 연산 최소화 및 메모리 최적화가 이뤄질수록 총 연산 코스트 절감                                                                                    |

#### 대표적 PFLOPS-days 절감 전략

##### 1. 하드웨어 및 인프라 효율화

- **최적화된 인스턴스 타입/구성**: 자체 하드웨어 외, GPU/TPU 선택, 클라우드의 최적 비용-성능 인스턴스 활용.
- **병렬화·분산 학습**: 여러 장비를 효율적으로 동원하고, 네트워크/데이터 병목 최소화[4][5][6].
- **클러스터 자동 확장/축소**: 필요 작업만 자원을 할당(오토스케일링); 유휴 리소스 감소.
- **스팟/저가형 인스턴스 활용**: 클라우드에서 스팟(Preemptible/Spot) 인스턴스를 사용하여 단가를 크게 낮춤[7].

##### 2. 모델 구조 및 학습 알고리즘 개선

- **경량화 구조 설계**: 모델 파라미터 수를 줄이는 경량 모델, 효율적 트랜스포머, Separable Convolution 등 연산량이 적은 구조 도입[8].
- **모델 프루닝, 양자화**: 불필요한 파라미터 제거, 저정밀 연산(half-precision/8bit 연산 등) 도입.
- **Transfer/Adapter Learning**: 사전학습(pre-trained) 모델을 전이학습 및 어댑터 등으로 활용, “from scratch” 전체 학습 비용 감소[5].

##### 3. 학습 프로세스 최적화

- **데이터 최적화**: 품질 높은 데이터 선별, 불필요 데이터 제거, 데이터 중복 배제.
- **효율적 Hyperparameter 탐색**: 서치 공간 축소, 초매개변수 자동 튜닝(HPO) 등으로 불필요한 반복 시도 절감[7].
- **조기 종료(Early Stopping)**: 성능 개선 효과가 없으면 학습 종료하여 불필요한 연산 방지.
- **체계적 자원 할당 및 모니터링**: 자원 낭비 방지, 실패 Job의 자동 종료 설정[6][5][7].

##### 4. Managed/AutoML 서비스 활용

- **클라우드 관리형 학습 서비스**: 인프라 자동 관리(AutoML, Vertex AI, Azure ML 등)로 오버프로비저닝·비효율 해소[5][6].
- **리소스 예약·할당 자동화**: 사용/비사용 시점에 자원 자동 할당-해제[6].

#### 주요 개념 요약

- **PFLOPS-days**: 대규모 AI 학습의 자원소비량/비용 척도 단위(연산량=속도×시간).
- **절감 전략**: 하드웨어 효율화, 모델·알고리즘 개선, 프로세스/데이터 최적화, 관리형 서비스 활용 등.
- **실전 적용**: 필요한 비용 추산 → 병목/비효율 요인 파악 → 조합 전략 수립[4][5][7].

이 개념과 전략을 기반으로 실제 AI/ML 프로젝트에서 PFLOPS-days 기반 총학습 비용을 합리적으로 절감할 수 있습니다.

출처
[1] What are petaFLOPS (PFLOPS)? - IONOS https://www.ionos.com/digitalguide/server/know-how/pflops/
[2] Computation used to train notable AI systems, by affiliation of ... https://ourworldindata.org/grapher/artificial-intelligence-training-computation-by-researcher-affiliation
[3] AI and compute https://openai.com/index/ai-and-compute/
[4] Transformer training costs | Continuum Labs https://training.continuumlabs.ai/infrastructure/data-and-memory/transformer-training-costs
[5] AI and ML perspective: Cost optimization https://cloud.google.com/architecture/framework/perspectives/ai-ml/cost-optimization
[6] Manage and optimize Azure Machine Learning costs https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2
[7] Cost optimization - Machine Learning Best Practices for ... https://docs.aws.amazon.com/whitepapers/latest/ml-best-practices-public-sector-organizations/cost-optimization.html
[8] How to Optimize a Deep Learning Model for faster Inference? https://www.thinkautonomous.ai/blog/deep-learning-optimization/
[9] [D] The cost of training GPT-3 : r/MachineLearning - Reddit https://www.reddit.com/r/MachineLearning/comments/hwfjej/d_the_cost_of_training_gpt3/
[10] Petaflops per second-days - DeepLearning.AI https://community.deeplearning.ai/t/petaflops-per-second-days/365974
[11] Estimating PaLM's training cost https://blog.heim.xyz/palm-training-cost/
[12] Disaggregating Power in Data Centers - Vicor Corporation https://www.vicorpower.com/resource-library/articles/high-performance-computing/disaggregating-power-in-data-centers
[13] What is the cost of training large language models? - CUDO Compute https://www.cudocompute.com/blog/what-is-the-cost-of-training-large-language-models
[14] 8 ways to reduce cycle time with robots: Step-by-step guide https://standardbots.com/blog/reduce-cycle-time-guide
[15] FLOPS (Floating Point Operations Per Second) - Klu.ai https://klu.ai/glossary/flops
[16] Trends in the Dollar Training Cost of Machine Learning Systems https://epoch.ai/blog/trends-in-the-dollar-training-cost-of-machine-learning-systems
[17] 10 strategies for cycle time reduction https://about.gitlab.com/blog/strategies-to-reduce-cycle-times/
[18] DeepSeek V3 and the actual cost of training frontier AI models https://www.interconnects.ai/p/deepseek-v3-and-the-actual-cost-of
[19] Saving cost on your machine learning training and inference ... https://pages.awscloud.com/EMEA-ML-Cost-Optimization.html
[20] Flops - Lark https://www.larksuite.com/en_us/topics/ai-glossary/flops

## 5. 모델 평가 개요
- 평가 지표의 필요성
### 분류(Classification) 지표
  - 정확도(Accuracy)
  - 정밀도(Precision)
  - 재현율(Recall)
  - F1-score
  - ROC-AUC, PR-AUC

| 지표        | 정의 및 수식                                                                                  | 의미 및 특징                                                                                  |
|-------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Accuracy (정확도)   | $$\frac{TP + TN}{TP + TN + FP + FN}$$                                                   | 전체 예측 중에서 맞게 예측한 비율. 클래스 불균형이 심할 때는 신뢰도가 떨어질 수 있음[1][2].          |
| Precision (정밀도)  | $$\frac{TP}{TP + FP}$$                                                                 | 양성으로 예측한 것 중 실제로 양성인 비율. False Positive(오탐)를 줄이는 데 중요[3][4].               |
| Recall (재현율)     | $$\frac{TP}{TP + FN}$$                                                                 | 실제 양성 중에서 모델이 맞게 예측한 비율. False Negative(누락)를 줄이는 데 중요[3][5][2].            |
| F1-Score     | $$2 \times \frac{Precision \times Recall}{Precision + Recall}$$                              | 정밀도와 재현율의 조화평균. 두 지표의 균형이 필요할 때 사용. 1에 가까울수록 성능이 우수[6].           |
| ROC AUC      | ROC 곡선 아래 면적 (0~1)                                                                     | 임계값 변화에 따른 분류 성능을 종합적으로 평가. 1에 가까울수록 분류 성능 우수. 클래스 구분력 지표[7].  |

**용어 설명**
- TP: True Positive (실제 양성, 예측도 양성)
- TN: True Negative (실제 음성, 예측도 음성)
- FP: False Positive (실제 음성, 예측은 양성)
- FN: False Negative (실제 양성, 예측은 음성)

**요약**
- **Accuracy**: 전체 예측 중 정답 비율. 클래스 불균형에 취약[1][2].
- **Precision**: 양성 예측 중 실제 양성 비율. 오탐이 중요한 분야에 적합[3][4].
- **Recall**: 실제 양성 중 맞춘 비율. 누락이 중요한 분야에 적합[3][5][2].
- **F1-Score**: Precision과 Recall의 조화평균(harmonic mean)으로, 두 지표를 균형 있게 반영하여 클래스 불균형 상황에서도 모델 성능을 종합적으로 평가할 수 있다.[6].
- **ROC AUC**: 임계값 변화 전반의 분류 성능. 1에 가까울수록 좋음[7].

**참고: roc-auc-precision-and-recall-visually-explained**
[roc-auc-precision-and-recall-visually-explained](https://paulvanderlaken.com/2019/08/16/roc-auc-precision-and-recall-visually-explained/)

**참고: Confusion Matrix 계산기**
[Confusion Matrix Calculator](https://www.omnicalculator.com/statistics/confusion-matrix)

출처
[1] Accuracy vs. precision vs. recall in machine learning - Evidently AI https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall
[2] Accuracy vs. Precision vs. Recall in Machine Learning - Encord https://encord.com/blog/classification-metrics-accuracy-precision-recall/
[3] Classification: Accuracy, recall, precision, and related metrics https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
[4] Precision - Logicballs https://logicballs.com/glossary/precision/
[5] Recall - Logicballs https://logicballs.com/glossary/recall/
[6] An Introduction to the F1 Score in Machine Learning - Wandb https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-the-F1-Score-in-Machine-Learning--Vmlldzo2OTY0Mzg1
[7] What is Receiver Operating Characteristic Area Under Curve (ROC-AUC)? https://klu.ai/glossary/roc-auc
[8] Accuracy - Cognilytica https://www.cognilytica.com/glossary/accuracy/
[9] Accuracy (error rate) Definition - DeepAI https://deepai.org/machine-learning-glossary-and-terms/accuracy-error-rate
[10] What Is Accuracy In Machine Learning | Robots.net https://robots.net/fintech/what-is-accuracy-in-machine-learning/

**평가지표 활용 예**
| 상황                 | 중시 지표              | 이유                      |
| ------------------ | ------------------ | ----------------------- |
| 암 진단, 범죄 탐지, 공항 보안 | **Recall**         | 놓치면 큰 피해 (FN ↓)         |
| 스팸 분류, 광고 추천       | **Precision**      | 잘못 긍정하면 사용자가 불쾌 (FP ↓)  |
| 불균형 클래스 문제         | **F1 Score**       | Precision과 Recall 모두 중요 |
| 전체 모델 평가           | **ROC Curve, AUC** | 임계값 변화에 대한 전체 성향 시각화    |

## 5. 모델 평가 개요

### 회귀(Regression) 지표

회귀 문제에서는 실제 연속적인 값을 예측하므로, 예측값과 실제값 간의 오차를 기반으로 모델 성능을 평가합니다. 대표적인 지표들은 다음과 같습니다.

| 지표  | 정의 및 설명 | 특징 |
|-------|------------|-------|
| MSE (Mean Squared Error, 평균 제곱 오차) | 실제값과 예측값 차이를 제곱한 뒤 평균한 값<br> $$\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y_i})^2$$ | 큰 오차에 더 큰 패널티를 줌으로써, 큰 예측 오차가 중요시됨. 이상치에 민감함. 값이 작을수록 좋음. |
| RMSE (Root Mean Squared Error, 평균 제곱근 오차) | MSE의 제곱근을 취해 오차 단위를 실제 데이터 단위에 맞춤<br> $$\text{RMSE} = \sqrt{\text{MSE}}$$ | MSE보다 해석이 직관적이며, 큰 오차에 민감. 값이 작을수록 좋음. |
| MAE (Mean Absolute Error, 평균 절대 오차) | 실제값과 예측값 차이의 절댓값 평균<br> $$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y_i}|$$ | 이상치에 덜 민감하며, 해석이 직관적. 값이 작을수록 좋음. |
| $$R^2$$ (결정계수) | 모델이 실제 데이터를 얼마나 잘 설명하는지 나타내는 지표로, 1에 가까울수록 좋은 모델<br> $$ R^2 = 1 - \frac{\sum (y_i - \hat{y_i})^2}{\sum (y_i - \bar{y})^2} $$ | 음수가 될 수도 있으나, 일반적으로 0~1 사이 값을 가지며 클수록 설명력이 높음. |

***

### 랭킹(Ranking) 지표

랭킹 문제는 검색, 추천시스템 등에서 결과 순서의 질을 평가할 때 사용되는 지표로, 대표적인 것들은 다음과 같습니다.

| 지표  | 정의 및 설명 | 특징 |
|-------|------------|-------|
| NDCG (Normalized Discounted Cumulative Gain) | 순위가 높을수록 더 큰 가중치를 주어 정답과 예측 랭킹의 품질을 평가하는 지표 | 랭킹의 위치와 정답의 중요도 모두 고려, 0~1 사이 값 |
| MRR (Mean Reciprocal Rank) | 정답이 처음 나타나는 위치의 역수를 평균한 값 | 첫 번째 정답에 초점, 값이 클수록 좋은 성능 |

랭킹 지표는 결과 순서가 중요한 문제에서 단순 정확도보다 더 의미 있는 평가를 제공합니다.

***

이처럼 회귀와 랭킹 문제는 목적에 맞는 적절한 지표로 모델 성능을 평가하고 해석해야 합니다.

출처
[1] 회귀 모델 성능 평가 지표(MAE, MSE, RMSE, MAPE 등) - Note https://white-joy.tistory.com/10
[2] 회귀모델 평가지표 - R2 score 결정계수, MAE, MSE, RMSE, MAPE ... https://sy-log.tistory.com/entry/%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C-R2-score-%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98-MAE-MSE-RMSE-MAPE-MPE
[3] [ML] 머신러닝 평가지표 - 회귀 모델 MSE, RMSE, MAE - DataPilots https://datapilots.tistory.com/42
[4] 회귀 평가지표 - MAE, MSE, R² - 퇴근 후 study with me - 티스토리 https://for-my-wealthy-life.tistory.com/68
[5] 회귀 모델 성능평가지표 : MAE, MSE, RMSE, R2 Score https://emjayahn.github.io/2022/02/10/Regression-Score/
[6] 머신러닝 - 17. 회귀 평가 지표 - 귀퉁이 서재 - 티스토리 https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-17-%ED%9A%8C%EA%B7%80-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C
[7] 회귀 모델 평가 지표 정리 - 하미's 블로그 https://carpe08.tistory.com/501
[8] [D+25][ML] 지도학습 - 회귀 분석과 평가지표(MSE, RMSE, R²) https://soojung624.tistory.com/26
[9] 회귀분석(Regression)의 모델 평가 알아보기 - 브런치 https://brunch.co.kr/@26dbf56c3e594db/67
[10] [TIL] 다중선형회귀와 평가지표 - velog https://velog.io/@woooa/TIL-%EB%8B%A4%EC%A4%91%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EC%99%80-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C


### 코사인 유사도

#### 코사인 유사도의 정의

코사인 유사도(Cosine Similarity)는 두 벡터 간의 방향이 얼마나 유사한지를 측정하는 지표로, 벡터의 크기보다는 두 벡터가 이루는 각도(코사인 값)를 기반으로 유사도를 계산합니다. 값의 범위는 -1에서 1 사이이며, 1에 가까울수록 두 벡터가 같은 방향, 0에 가까울수록 두 벡터가 직각, -1은 완전히 반대 방향임을 의미합니다.

#### 코사인 유사도 계산식

코사인 유사도는 다음의 수식으로 계산됩니다.

$$
\text{cosine similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}
$$

여기서  
- $$\mathbf{A} \cdot \mathbf{B}$$: 벡터 A와 B의 내적  
- $$\|\mathbf{A}\|$$: 벡터 A의 크기(노름)  
- $$\|\mathbf{B}\|$$: 벡터 B의 크기(노름)

#### 코사인 유사도 이용 사례

- **문서 유사도 측정**: 자연어 처리(NLP) 분야에서 두 개의 문서가 얼마나 유사한지 평가할 때 문서별 단어 벡터(또는 TF-IDF 벡터) 간의 코사인 유사도를 활용.
- **추천 시스템**: 사용자와 아이템의 선호 벡터 간 유사도 계산을 통해 맞춤형 추천 제공.
- **클러스터링**: 유사도가 높은 데이터 포인트끼리 군집화하는 데 사용.

#### 코사인 유사도와 비슷한 다른 개념

자연어처리(NLP)에서 텍스트의 유사성을 수치화할 때 널리 쓰이는 여러 **유사도 평가 방법**을 아래 표로 정리합니다.

| **유사도 지표**      | **계산 방식 및 특징**                                                                                         | **주요 활용 예**                          | **참고** |
|-------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------|------|
| 코사인 유사도         | 두 벡터의 내적을 각각의 노름(크기)으로 나눈 값, -1~1 범위, **방향성**을 중시함. 크기에 영향을 받지 않음[1][3][7].               | 문서/문장 임베딩 비교, 추천 시스템, 검색 엔진 | 가장 널리 사용됨 |
| 유클리드 거리(유사도)      | 두 벡터 사이의 **직선 거리**(L2 norm)를 측정, 작을수록 유사, **절대적 크기** 반영. 단, 데이터가 표준화되어 있지 않으면 주의 필요[2][5].                  | 이미지, 사운드 데이터 마이닝, 임베딩 비교      | 거리 기반 방식 |
| 맨허튼(맨해탄) 거리(유사도) | 각 차원의 **절대값 차이 합**(L1 norm), 작을수록 유사, **좌표(차원) 단위 절대 거리** 측정[2][5].                                     | 문장/단어 임베딩 비교, 특이값 탐지           | 거리 기반 방식 |
| 자카드 유사도             | **집합**으로 변환 후, 교집합 크기 / 합집합 크기. 0~1 범위, **단어의 중복 여부**만 반영(순서 무시)[2][5][6].                   | 짧은 텍스트, 이벤트/행동 기록 등 집합형 데이터 | 이진/집합 데이터에 적합 |
| 피어슨 상관계수           | 두 변수 간 **선형 상관관계**(가장 유사: +1, 반대 방향: -1, 무관계: 0), **표준화된 민감도** 측정, 평균, 분산을 고려함.                    | 신호/시계열 데이터, 선형 회귀 텍스트 실험 등  | 통계적 유사도 |
| n-gram 기반 유사도        | n개의 연속 문자(혹은 단어)를 묶어, 겹치는 n-gram의 비율로 유사도 산출. **국부적 패턴**에 강점[4].                                       | 표절 탐지, 음성 인식, 맞춤법 검사             | 문자열/음성 데이터 |

- **코사인 유사도**: 문장 임베딩이나 문서 벡터화 후 방향성을 기반으로 유사도 산출, **가장 대표적**임[1][3][9].
- **유클리드/맨해탄 거리**: 벡터 간 거리(패턴, 위치) 차이에 주목, 데이터 정규화가 중요.
- **자카드 유사도**: 순서, 빈도, 문법적 구조를 무시하고 집합의 교집합/합집합 비율로 산출, **짧은 텍스트/이진 데이터**에 적합[2][5][6].
- **피어슨 상관계수**: 두 벡터의 **선형 관계** 측정, 사회과학/과학적 방법론에서도 활용.
- **n-gram 유사도**: 문자열/음성의 **국소적 패턴** 일치를 중시, 띄어쓰기/맞춤법 오류 탐지 등에 효과적[4].

실제 NLP에서는 **코사인 유사도**와 **임베딩 벡터화**가 가장 보편적이며, **자카드 유사도**는 집합 기준 텍스트 간 중복 검출 등에, **n-gram**은 표절/음성 패턴 매칭 등에 적합합니다. **거리 기반** 방법(유클리드, 맨해탄)은 차원 축소나 비정형 데이터 탐색 시, **피어슨**은 통계적 분석 시 활용됩니다.

출처
[1] [python] 자연어처리(NLP) - 텍스트 유사도 https://wonhwa.tistory.com/26
[2] 3장 자연어 처리 개요(3) : 텍스트 유사도 https://coshin.tistory.com/31
[3] 05-01 코사인 유사도(Cosine Similarity) - 위키독스 https://wikidocs.net/24603
[4] [자연어처리] 중복 검출을 위한 텍스트 유사도 측정 - velog https://velog.io/@ykang5/%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95
[5] 자연어 처리 개요_텍스트 유사도 - 데이터 한 그릇 - 티스토리 https://kurt7191.tistory.com/117
[6] [ NLP ] 텍스트 유사도 ( 자카드 유사도, 코사인 유사도 ) - 야누쓰 https://yanoo.tistory.com/21
[7] [자연어처리 입문] 4. 벡터의 유사도(Vector Similarity) https://codong.tistory.com/35
[8] 유사도 측정법 (Similarity Measure) - 도리의 디지털라이프 https://blog.skby.net/%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95%EB%B2%95-similarity-measure/
[9] OpenAI 임베딩으로 유사한 문장 찾기 실습 - velog https://velog.io/@cha-suyeon/OpenAI-%EC%9E%84%EB%B2%A0%EB%94%A9%EC%9C%BC%EB%A1%9C-%EC%9C%A0%EC%82%AC%ED%95%9C-%EB%AC%B8%EC%9E%A5-%EC%B0%BE%EA%B8%B0-%EC%8B%A4%EC%8A%B5

## 6. 모델 평가 심화

### 혼동 행렬(Confusion Matrix) 해석
혼동 행렬은 분류 모델 성능을 평가하는 데 많이 사용되는 도구로, 모델의 예측값과 실제값 간 관계를 표로 나타냅니다. 보통 다음 4가지 요소로 구성됩니다:

- **True Positive (TP)**: 실제 Positive를 정확히 Positive로 예측
- **True Negative (TN)**: 실제 Negative를 정확히 Negative로 예측
- **False Positive (FP)**: 실제 Negative를 잘못 Positive로 예측 (오류 유형 1)
- **False Negative (FN)**: 실제 Positive를 잘못 Negative로 예측 (오류 유형 2)

이 행렬을 바탕으로 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수 등 다양한 지표를 계산하며, 특히 불균형 데이터 상황에서 단순 정확도보다 더 의미 있는 성능 파악이 가능합니다. 다중 클래스 분류 문제도 각 클래스마다 혼동 행렬을 작성하여 어떤 클래스 간 혼동이 많은지 시각적으로 분석할 수 있습니다.

***

### 샘플 효율성(Sample Efficiency)
샘플 효율성은 모델이 주어진 데이터 샘플을 얼마나 효율적으로 활용하여 높은 성능을 낼 수 있는지를 나타냅니다. 적은 데이터에서도 빠르게 학습하여 효율적으로 일반화하는 능력을 의미합니다. 특히 강화학습이나 저자원 환경에서 중요한 평가 요소로 활용됩니다. 샘플 효율성이 높은 모델은 데이터 및 계산 비용을 줄이면서도 좋은 결과를 이끌어냅니다.

***

### OOD(Out-of-Distribution) 데이터 평가
OOD 평가는 모델이 훈련 데이터 분포를 벗어난 새로운 데이터에 대해 얼마나 잘 대응하는지를 확인하는 과정입니다. 실제 활용 환경에서는 훈련 데이터와 분포가 다른 데이터가 자주 발생하기 때문에, 모델이 이러한 OOD 데이터를 인지하고 올바른 판단을 할 수 있어야 신뢰할 수 있습니다. OOD 탐지 기법으로는 Softmax 확률 분석, ODIN, Outlier Exposure 등이 있습니다.

***

### LLM·멀티모달 모델 평가 방법론
대형 언어 모델(LLM)과 텍스트, 이미지, 음성 등 복수의 입력 모달리티를 처리하는 멀티모달 모델의 평가는 다음과 같은 방법을 사용합니다:

- 작업별 벤치마크 데이터셋 활용 (예: Visual Question Answering, 이미지 캡션 등)
- 마이크로/Macro F1, 정확도 등 다양한 다면적 지표 도입
- 모달별 세밀한 평가를 통해 단일 모달 동작뿐 아니라 모달 간 상호작용 및 통합 능력 분석
- 사용자 피드백과 실제 사용 사례를 반영한 정성적 평가 병행

이런 평가 방법을 통해 복잡한 모델이 다양한 환경과 태스크에서 안정적이고 일관된 성능을 발휘하는지 검증할 수 있습니다.

출처
[1] 혼동행렬(confusion matrix) - 의미를 이해하는 통계학과 데이터 분석 https://diseny.tistory.com/entry/%ED%98%BC%EB%8F%99%ED%96%89%EB%A0%ACconfusion-matrix
[2] 혼동 행렬(Confusion Matrix) 해석 - 하미's 블로그 https://carpe08.tistory.com/500
[3] Confusion Matrix로 분류모델 성능평가 지표(precision, recall, f1 ... https://kyull-it.tistory.com/99
[4] [파이썬 sklearn] 오차행렬(혼동행렬, confusion matrix) 공부하기 - 선비 https://spine-sunbi.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-sklearn-%EC%98%A4%EC%B0%A8%ED%96%89%EB%A0%AC%ED%98%BC%EB%8F%99%ED%96%89%EB%A0%AC-confusion-matrix-%EA%B3%B5%EB%B6%80%ED%95%98%EA%B8%B0-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-%EC%9D%B4%ED%95%B41
[5] [ML] 혼동 행렬(Confusion Matrix) / TP/TN/FP/FN - 홍시의 씽크탱크 https://kimhongsi.tistory.com/entry/ML-%ED%98%BC%EB%8F%99-%ED%96%89%EB%A0%ACConfusion-Matrix-TPTNFPFN
[6] 머신러닝 분류 - 혼동 행렬(Confusion matrix) - 세상탐험대 블로그 https://skillmemory.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%84%EB%A5%98-%ED%98%BC%EB%8F%99-%ED%96%89%EB%A0%ACConfusion-matrix
[7] 혼동행렬 (Confusion Matrix) - 지식덤프 http://www.jidum.com/jidums/view.do?jidumId=1212
[8] 혼동 행렬이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/confusion-matrix


## 7. 모델 성능 향상 기법
### 전이학습(Transfer Learning)

전이학습은 이미 어떤 작업(Task)에 대해 학습된 모델이나 네트워크가 가진 지식을 새로운, 관련되었거나 유사한 작업에 재활용하여 학습 효율성과 성능을 개선하는 기술입니다. 예를 들어 대량의 이미지 데이터인 ImageNet으로 미리 학습된 CNN 모델을 가져와 고양이와 개를 분류하는 작은 데이터셋에 적용하는 것입니다. 이 경우 초기 레이어들은 이미지의 기본 특성(모서리, 패턴 등)을 추출하는 역할을 하므로 이 지식을 그대로 활용하고, 후반부 레이어만 새롭게 학습하거나 미세 조정하여 고양이 vs 개 문제에 맞게 최적화합니다.

- **장점**
  - 대량 데이터와 긴 학습 시간을 절약할 수 있음
  - 작은 데이터셋으로도 좋은 성능 달성 가능
  - 사전 학습된 모델의 특징 추출 능력을 활용해 오버피팅 방지에 유리
- **구성 요소**
  - 소스 도메인/태스크: 최초 학습된 도메인과 작업
  - 타겟 도메인/태스크: 새로운 도메인과 작업
- **전이학습 방법**
  - 고정된 피처 추출기 사용 후 분류기만 재학습
  - 미세 조정(fine-tuning)으로 전체 또는 일부 네트워크 파라미터 수정

사전학습된 모델을 재활용하므로, 전이학습은 사전학습(pre-training)과 미세 조정(fine-tuning)을 아우르는 개념으로도 볼 수 있습니다.

***

### 파인튜닝(Fine-tuning)

파인튜닝은 전이학습 중 사전학습된 모델을 특정한 작은 데이터셋이나 특화된 태스크에 맞추어 추가로 학습시키는 과정입니다. 주로 전체 모델 또는 일부 계층의 가중치를 업데이트하여 모델이 새로운 데이터 환경이나 요구사항에 적합하도록 조정합니다.

- **특징**
  - 보통 낮은 학습률을 사용해 기존 가중치의 안정성을 유지하면서 점진적 적응
  - 과적합 위험 감소를 위해 일부 레이어만 훈련하거나 드롭아웃, 조기 종료 같은 기법을 병행
  - 특정 도메인(의료, 법률 등)이나 목적에 특화된 모델 개발에 강점

***

### LoRA, Prefix Tuning 등 경량 학습 기법

- **LoRA (Low-Rank Adaptation)**
  - 대체로 크고 복잡한 LLM에 적용
  - 전체 모델 가중치를 업데이트하는 대신, 가중치 행렬을 저차원 행렬들의 곱으로 분해하고 이들 중 일부만 학습함
  - 저장 공간과 계산량 크게 감소시키며 효율적인 미세 조정 가능
  - 기존 사전학습 모델을 그대로 두면서 추가 학습만 수행, 확장성 좋음

- **Prefix Tuning**
  - 프롬프트 기반 미세 조정 기법
  - 모델 입력 앞에 학습 가능한 "프리픽스"(특수 벡터 시퀀스)를 추가
  - 모델 파라미터는 고정한 채 프리픽스 벡터만 업데이트
  - 빠르고 자원 효율적이며, 대형 모델에 적합

이들 경량 학습 기법은 급격한 모델 크기 증가와 이를 학습시키는 비용 부담 문제를 해결하며, 최근 대형 언어 모델 적용에 필수적인 기술로 주목받고 있습니다.

출처
[1] 전이 학습이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/transfer-learning
[2] 전이학습(Transfer Learning)이란? - 데이콘 https://dacon.io/forum/405988
[3] 딥러닝 기초 - 전이 학습 이해하기 - velog https://velog.io/@tjdtnsu/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EC%B4%88-%EC%A0%84%EC%9D%B4-%ED%95%99%EC%8A%B5-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
[4] 전이 학습(Transfer learning)이란? 정의, 사용 방법, AI 구축 | appen 에펜 https://kr.appen.com/blog/transfer-learning/
[5] 전이 학습(Transfer Learning) 개념 및 활용 - learningflix - 티스토리 https://learningflix.tistory.com/138
[6] 05화 전이학습(Transfer Learning)이란? - 브런치 https://brunch.co.kr/@harryban0917/283
[7] 전이학습: 사전 훈련된 모델 활용 전략 - 재능넷 https://www.jaenung.net/tree/14200
[8] 파이썬을 활용한 딥러닝 전이학습 - 위키북스 https://wikibook.co.kr/transfer-learning/
[9] 전이 학습을 위한 기반 모델 결정 방법 및 그 방법을 지원하는 장치 https://patents.google.com/patent/KR102439606B1/ko
[10] 21.03.04. 딥러닝 - - 모도리는 공부중 - 티스토리 https://studying-modory.tistory.com/entry/210304-%EB%94%A5%EB%9F%AC%EB%8B%9D

### [앙상블(Ensemble) 기법](https://github.com/zoro0rkd/ai_study/wiki/AI-모델-아키텍처-설계-%E2%80%90-NEW#9-앙상블-구조)

### 경량화 기법

AI 모델 경량화 기법 중 **Pruning(가지치기)**은 모델의 불필요한 매개변수를 제거하여 크기와 계산 복잡도를 줄이는 기술입니다. 이 기법은 **EfficientNet**, **MobileNet**, **SqueezeNet** 등 모바일 친화적 아키텍처에서 핵심적으로 활용되며, 다음과 같은 방식으로 적용됩니다.

---

#### Pruning의 작동 원리
- **Unstructured Pruning**: 개별 가중치를 무작위로 제거하는 방식으로, 세밀한 압축이 가능하지만 하드웨어 가속에 비효율적[1][8].
- **Structured Pruning**: 뉴런/필터 단위로 제거하여 하드웨어 호환성을 높이며, MobileNet 등에서 채널 단위 절삭에 활용[3][8].
- **Lottery Ticket Hypothesis**: 초기 가중치를 유지한 서브네트워크가 원본 모델과 유사한 성능을 보인다는 이론으로, 반복적 가지치기 기반[1][7].

| 기법         | 특징                          | 적용 예시               |
|--------------|-------------------------------|-------------------------|
| Unstructured | 95% 가중치 제거 가능          | VGG, ResNet[6]         |
| Structured   | GPU 가속 최적화               | MobileNet, EfficientNet[5][9] |

---

#### 주요 모델별 적용 사례
##### 1. SqueezeNet
- **1x1 컨볼루션**으로 기본 구조를 경량화한 후, **Deep Compression**(가지치기 + 6비트 양자화) 적용 시 **510x 크기 감소** 달성[6].
- 원본 AlexNet 대비 50x 작은 4.8MB 모델로 동등한 정확도 유지[6].

##### 2. MobileNet
- **Depthwise Separable Convolution**으로 매개변수 90% 절감[5].
- AutoML 기반 **AMC(Automated Model Compression)**로 레이어별 최적 압축 정책 탐색[9]:
  - GPU(Titan Xp)에서 1.53x, 픽셀 폰에서 1.95x 추론 속도 향상.

##### 3. EfficientNet
- **Compound Scaling**(깊이/너비/해상도 통합 최적화)으로 구조 효율화[9].
- 강화학습 기반 자동 압축(AMC)으로 리소스 제약 조건 하 최적 서브네트워크 탐색[9].

---

Pruning은 단독으로도 효과적이지만 **양자화**, **지식 증류**와 결합할 때 최대 효율을 발휘합니다. 모바일 디바이스의 실시간 AI 적용을 위해 MobileNet/EfficientNet은 구조 설계 단계부터 압축을 고려하며, SqueezeNet은 사후 압축의 극단적 사례로 자리잡았습니다[5][6][9].

출처
[1] 모델 경량화 1 - Pruning(가지치기) https://blogik.netlify.app/boostcamp/u_stage/45_pruning/
[2] Model Pruning: Keeping the Essentials - Unify AI https://unify.ai/blog/compression-pruning
[3] Pruning (artificial neural network) - Wikipedia https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)
[4] Edge AI: Evaluation of Model Compression Techniques for ... - arXiv https://arxiv.org/html/2409.02134v1
[5] Deep Learning on Mobile Devices: Strategies for Model ... - Zetic.ai https://zetic.ai/ko/blog/deep-learning-on-mobile-devices-strategies-for-model-compression-and-optimization
[6] [PDF] SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER ... https://openreview.net/pdf?id=S1xh5sYgx
[7] 딥러닝 모델 경량화를 위한 Pruning 기본 개념 정리 - Seanpark https://seanpark11.tistory.com/190
[8] AI Model Compression Techniques - Sogeti Labs https://labs.sogeti.com/ai-model-compression-techniques/
[9] [PDF] AutoML for Model Compression and Acceleration on Mobile Devices https://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf
[10] 모델 경량화 방법 - 인공지능 공부 - 티스토리 https://ai0-0jiyun.tistory.com/5
[11] 경량화 기법 정리: Pruning, Quantization, Knowledge Distillation - velog https://velog.io/@mmodestaa/%EA%B2%BD%EB%9F%89%ED%99%94-%EA%B8%B0%EB%B2%95-%EC%A0%95%EB%A6%AC-Pruning-Quantization-Knowledge-Distillation
[12] [최적화] 모델 경량화 , AutoML , Pruning , Knowledge Distillation ... https://amber-chaeeunk.tistory.com/110
[13] 딥러닝 모델 최적화 방법: 모델 경량화와 모델 추론 속도 가속화 https://blog-ko.superb-ai.com/how-to-optimize-deep-learning-models/
[14] [2412.02328] Efficient Model Compression Techniques with FishLeg https://arxiv.org/abs/2412.02328
[15] A Comparative Study of Preprocessing and Model Compression ... https://www.mdpi.com/1424-8220/24/4/1149
[16] [PDF] Lecture 9: Models Compression Techniques - GitHub Pages https://harvard-iacs.github.io/2023-AC215/assets/lectures/lecture9/05_model2_compression_techniques.pdf
[17] [PDF] Efficient Model Compression Techniques with FishLeg - OpenReview https://openreview.net/pdf?id=0PnN3hKYL7
[18] Model Compression - an overview | ScienceDirect Topics https://www.sciencedirect.com/topics/computer-science/model-compression
[19] [PDF] What is the State of Neural Network Pruning? - arXiv https://arxiv.org/pdf/2003.03033.pdf
[20] Neural Network Pruning - Nathan Hubens https://nathanhubens.github.io/posts/deep%20learning/2020/05/22/pruning.html

## 8. 모델 학습과 평가의 한계 및 고려사항
### Bias & Variance
![스크린샷 2025-07-07 오후 1 48 02](https://github.com/user-attachments/assets/0dd50ce8-07e2-433b-ae0f-a6fe44a4dd70)

AI를 학습하면 4가지 케이스 중 하나.

#### Bias & Variance tradeoff
![스크린샷 2025-07-07 오후 1 53 08](https://github.com/user-attachments/assets/e41164be-bee2-4278-a8bd-c616a9e63680)

![스크린샷 2025-07-07 오후 1 53 24](https://github.com/user-attachments/assets/694b5679-5487-440a-8c8e-bcfdf3e353f8)

**Bias-Variance tradeoff(편향-분산 트레이드오프)**는 머신러닝과 통계학에서 모델의 성능과 일반화 능력을 이해하는 데 핵심적인 개념입니다. 이는 **모델의 복잡성에 따라 발생하는 두 가지 주요 오차(편향, 분산)** 사이의 균형을 의미하며, 이 두 오차를 동시에 최소화하는 것이 불가능하다는 딜레마를 설명합니다[1][3][5].
### 왜 Tradeoff(트레이드오프)인가?

- **편향과 분산은 반비례 관계**에 있습니다.  
  모델이 단순할수록(복잡성이 낮을수록) 편향은 커지고 분산은 작아집니다.  
  반대로 모델이 복잡할수록 편향은 줄어들지만 분산이 커집니다[1][8].

- **이상적인 모델**은 편향과 분산이 모두 낮아야 하지만, 실제로는 둘을 동시에 최소화할 수 없습니다.  
  따라서 **적절한 균형점(Optimal Point)**을 찾는 것이 중요합니다[1][5].

#### 수식과 시각적 이해

- **예측 오차(Mean Squared Error, MSE)는 다음과 같이 분해할 수 있습니다:**  
  $$
  \text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
  $$
  여기서 Irreducible Error(줄일 수 없는 오차)는 데이터 자체의 노이즈 등 모델로는 줄일 수 없는 부분입니다[4][5].

- **그래프 상에서**  
  - 모델 복잡성이 증가할수록 편향은 감소, 분산은 증가  
  - 테스트 에러는 처음에는 감소하다가(편향 감소 효과), 이후 다시 증가(분산 증가 효과)  
  - **최적점**은 테스트 에러가 최소가 되는 지점, 즉 적절한 복잡성을 가진 모델입니다[5].

#### 실전에서의 활용

- **Underfitting(과소적합)**: 높은 편향, 낮은 분산 → 모델이 너무 단순  
- **Overfitting(과대적합)**: 낮은 편향, 높은 분산 → 모델이 너무 복잡  
- **좋은 모델**: 적절한 편향과 분산의 균형을 이루어, 훈련 데이터뿐 아니라 새로운 데이터(테스트 데이터)에서도 좋은 성능을 보임[5][7].

**요약**  
Bias-Variance tradeoff는 머신러닝 모델이 훈련 데이터와 새로운 데이터 모두에서 잘 작동하도록 하기 위해, 모델의 복잡성과 두 오차(편향, 분산) 사이의 균형을 어떻게 맞출 것인지를 설명하는 핵심 원리입니다[1][3][5].

출처
[1] Bias-Variance trade-off - 편향-분산 트레이드 https://velog.io/@lolhi/Bias-Variance-trade-off
[2] 편향-분산 트레이드오프 (Bias-Variance Tradeoff)와 L2 규제 ... https://untitledtblog.tistory.com/143
[3] 편향-분산 트레이드오프 https://ko.wikipedia.org/wiki/%ED%8E%B8%ED%96%A5-%EB%B6%84%EC%82%B0_%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%93%9C%EC%98%A4%ED%94%84
[4] 쉽게 이해해보는 bias-variance tradeoff - 건빵의 블로그 https://bywords.tistory.com/entry/%EB%B2%88%EC%97%AD-%EC%9C%A0%EC%B9%98%EC%9B%90%EC%83%9D%EB%8F%84-%EC%9D%B4%ED%95%B4%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-biasvariance-tradeoff
[5] 머신러닝 - Bias and Variance trade-off - 태호의 개발 블로그 https://dailytaeho.tistory.com/79
[6] 편향-분산 상충관계 - 위키피디아 https://translate.google.com/translate?u=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBias%25E2%2580%2593variance_tradeoff&sl=en&tl=ko&client=srp
[7] [인사이드 머신러닝] Bias-Variance Trade-Off https://velog.io/@cleansky/%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Bias-Variance-Trade-Off
[8] Bias-Variance Trade Off 이란? - Attention, Please!!! - 티스토리 https://g3lu.tistory.com/9

### 과적합(Overfitting) 및 과소적합(Underfitting)
* 언더피팅(Underfitting)과 오버피팅(Overfitting)은 딥러닝과 머신러닝에서 모델 성능에 영향을 미치는 대표적인 두 가지 학습 문제
* 모델의 학습 부족과 과잉 학습을 의미하며, 서로 반대 개념

| 구분        | 언더피팅 (Underfitting)                | 오버피팅 (Overfitting)                                           |
| --------- | ---------------------------------- | ------------------------------------------------------------ |
| 🔍 정의     | 모델이 충분히 학습하지 못해 성능이 낮은 상태<br/>모델이 **너무 단순**해서 훈련 데이터도 제대로 학습 못함  | 모델이 훈련 데이터에 과하게 맞춰져, 일반화가 안 되는 상태<br/>모델이 **너무 복잡**해서 훈련 데이터에 과하게 맞춤                               |
| 🎯 원인     | 모델이 너무 단순함<br>훈련 부족<br>입력 정보 부족    | 모델이 너무 복잡함<br>에폭 과다<br>노이즈까지 학습                              |
| 📊 특징     | 훈련 정확도 낮음<br>검증 정확도 낮음             | 훈련 정확도 매우 높음<br>검증 정확도 낮음                                    |
| 📈 그래프 형태 | 훈련/검증 오류 모두 높음 (→ 모델이 무지함)         | 훈련 오류는 낮지만, 검증 오류는 높음 (→ 모델이 외움)                             |
| 🛠 해결책    | 모델 복잡도 증가<br>더 많이 학습<br>더 많은 특징 사용 | 정규화 사용 (Dropout, L2 등)<br>단순한 모델 사용<br>조기 종료(Early Stopping) |

#### 간단한 예: 고양이 vs 강아지 구분
*  언더피팅:
  * 모델이 너무 단순해서 고양이 귀와 강아지 귀의 차이를 구분 못함 
  * 모두 그냥 "동물"이라고 예측 
* 오버피팅:
  * 모델이 훈련 데이터에 나온 특정 고양이 사진만 기억 
  * 새로운 고양이 사진이 나오면 예측 실패

#### 확인 방법: 학습 곡선(Learning Curve)
| 항목    | 훈련 정확도 | 검증 정확도   |
| ----- | ------ | -------- |
| 언더피팅  | 낮음     | 낮음       |
| 적절 학습 | 높음     | 높음       |
| 오버피팅  | 매우 높음  | 낮음 (떨어짐) |

* 참고
  * https://kh-kim.github.io/nlp_with_deep_learning_blog/docs/1-12-how-to-prevent-overfitting/02-overfitting/

### 평가 지표의 해석 한계
- **단일 지표의 한계:** 정확도, 정밀도, 재현율, F1 점수 등 각 평가지표는 특정 문제 유형이나 목표에 따라 강점과 약점이 존재합니다. 예를 들어, 정확도는 클래스가 불균형일 때 모델 성능을 과대평가할 수 있습니다. 정밀도와 재현율이 높다고 해도 현실에서는 한쪽만 높을 수 있고, F1 점수 역시 실제 응용 상황이나 비즈니스 목표를 반영하지 못할 수 있습니다[1][2].
- **다양성 미반영:** 기존 평가지표는 데이터 내의 서브그룹별 성능을 잘 포착하지 못하며, 신뢰도(Confidence), 불확실성(Entropy) 등 미묘한 요소를 간접적으로만 반영합니다.
- **주관성 및 적용 제한:** 생성형 LLM, 이미지 생성 등에서는 정답이 명확하지 않아서 BLEU, FID, 정확도 등 기존 지표의 단순한 수치만으로 평가가 어려우며, 평가자 개인의 판단, 맥락, 상황에 따라 결과가 달라질 수 있습니다[3][4][5].
- **모델 블랙박스 문제:** 성능 지표는 모델의 결정 과정을 완전히 설명하지 못하며, 특히 복잡한 딥러닝 모델의 경우 예측 근거를 해석하기 어려운 한계가 있습니다[6][7].

***

### 사회적·윤리적 영향 고려
- **공정성과 편향:** AI 시스템은 데이터에 내재된 편견을 재생산하거나 강화할 수 있으므로, 사회적 공정성(공평 배분, 차별 방지) 확보가 필수적입니다. 데이터셋의 다양성, 알고리즘의 투명성, 설명가능성(Explainability) 등도 중요한 윤리적 고려 사항입니다[8][7][9].
- **개인정보 보호:** 모델 학습 및 운영 과정에서 사용자의 개인 정보를 안전하게 보호하고, 관련 법/규정을 준수해야 합니다. GDPR 등 강화된 글로벌 기준이 점차 확대되고 있습니다.
- **책임성과 투명성:** AI 결정 과정의 투명성 확보와 그 결과에 대한 책임 부여가 중요합니다. 이는 신뢰성 확보와 함께, AI의 사회적 수용도를 높이는 데 필요합니다.
- **사회적 가치 및 영향:** 자동화로 인한 고용 변화, 신뢰성/안전성 문제, 불평등 심화 등 다양한 사회적 영향이 존재하며, 이에 대한 정책적 대응이나 교육, 기술 개발 방향의 설정이 필수적입니다.

***

결론적으로, 모델 평가 지표의 해석은 항상 그 한계를 인식하고, 다양한 사회적·윤리적 시각을 함께 반영하여, AI 기술이 더 안전하고 신뢰할 수 있으며 포용적인 방향으로 발전할 수 있도록 하는 것이 매우 중요합니다.

출처
[1] 머신러닝 성능 지표란? | 퓨어스토리지 - Pure Storage https://www.purestorage.com/kr/knowledge/machine-learning-performance-metrics.html
[2] 머신러닝 모델 평가 방법 (예: 정확도, F1 스코어) - LearnCodeEasy https://thebasics.tistory.com/105
[3] LLM 성능평가를 위한 지표들 - 슈퍼브 블로그 - Superb AI https://blog-ko.superb-ai.com/llm-evaluation-metrics/
[4] AI로 생성한 이미지는 어떻게 평가할까요? (기본편) https://techblog.lycorp.co.jp/ko/how-to-evaluate-ai-generated-images-1
[5] [LLM Evaluation] LLM 성능 평가 방법 - 가디의 tech 스터디 https://gagadi.tistory.com/58
[6] AI 기반 분석 프로젝트는 왜 실패하는가? AI 분석모델에 대한 오해와 진실 https://www.samsungsds.com/kr/insights/ai_analytics_model.html
[7] AI 윤리와 사회적 영향의 현주소 https://seo.goover.ai/report/202412/go-public-report-ko-63153ce6-da9e-40b5-ba98-e5876f4d696e-0-0.html
[8] 07 인공지능의 윤리와 사회적 영향 https://wikidocs.net/240300
[9] AI 시대의 윤리적 책임: 지속 가능한 발전을 위한 접근 방법 https://seo.goover.ai/report/202503/go-public-report-ko-f8a5db8a-7ffb-4ae8-ab1b-3f7744c19812-0-0.html
[10] LLM 성능, 어떻게 평가하는 것일까? (feat. lm-eval-harness) - DevOcean https://devocean.sk.com/blog/techBoardDetail.do?ID=166716&boardType=techBlog
