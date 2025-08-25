## 1. 모델 튜닝 개요
- 모델 튜닝의 정의와 필요성
    - 하이퍼파라미터 튜닝(Hyperparameter Tuning)은 머신러닝이나 딥러닝 모델의 성능을 최적화하기 위해 사전에 설정하는 매개변수를 조정하는 과정
    - 튜닝이 필요한 이유
        - 성능 최적화 : 과적합(overfitting)이나 언더피팅(underfitting)을 방지
        - 일반화 능력 향상 : 새로운 데이터에 잘 작동하도록 조정
        - 효율적인 학습 : 더 빠르게 수렴하고 계산 자원 절약
- 하이퍼파라미터(Hyperparameter)와 파라미터(Parameter) 차이

| 항목 | 파라미터(Parameter) | 하이퍼파라미터(Hyperparameter) |
| --- | --- | --- |
| 정의 |  모델이 학습을 통해 자동으로 결정하는 값 (e.g. 가중치, 편향) | 학습 전 수동으로 설정해야 하는 값 (e.g. learning rate, batch size) |
| 학습 여부 | 데이터 기반 학습됨 | 수동 또는 자동 탐색 필요 |
| 예시 | 신경망의 weight, bias | optimizer, dropout rate, layer 수, hidden unit 수 |

- **튜닝 대상 파라미터 우선순위 선정 기준**
    - 모델 성능에 민감한 순으로 조정
        - 학습률 (learning rate): 가장 큰 영향력
        - 모델 복잡도 관련: hidden units, depth, regularization
        - 데이터 관련: batch size, epoch


## 2. 하이퍼파라미터 최적화(HPO, Hyperparameter Optimization)
### 2.1 HPO 개요
- 정의와 목표
    - 정의: 최적 성능을 내는 하이퍼파라미터 조합을 자동으로 탐색하는 기법
    - 목표: 최소한의 계산 비용으로 성능을 극대화하는 하이퍼파라미터 조합 탐색
- 탐색 공간(Search Space) 설정
    - 튜닝할 하이퍼파라미터의 범위 및 타입을 정의
        - 범주형 (categorical): optimizer 종류
        - 정수형 (int): layer 수
        - 연속형 (float): learning rate
    - 예시
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)  # 최적 하이퍼파라미터 출력
```
- 목적 함수(Objective Function) 정의
    - 입력: 하이퍼파라미터 조합
    - 출력: 성능 평가 지표 (예: validation accuracy, F1 score)
    - 예시: val_loss = objective(params)

### 2.2 HPO 기법
| 기법                          | 개요                                                                 | 탐색 전략               | 장점                                                                 | 단점                                                                   | 활용 예 / 특징                            |
|-----------------------------|----------------------------------------------------------------------|------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------|
| Grid Search                 | 모든 가능한 하이퍼파라미터 조합을 체계적으로 탐색                     | 전수조사(Brute-force)   | 간단하고 구현 쉬움                                                  | 계산량이 폭발적으로 증가, 효율성 낮음                                      | 소규모 탐색 공간에서만 적합                |
| Random Search               | 랜덤하게 조합을 선택하여 탐색                                         | 무작위 탐색             | 고차원 공간에서 효율적, 불필요한 계산 줄임                             | 중요한 조합을 놓칠 수 있음                                               | 대규모 탐색 공간에서 더 효과적            |
| Bayesian Optimization       | 이전 결과를 기반으로 다음 탐색 지점을 예측                            | 확률 모델 기반 탐색     | 계산 효율 높음, 적은 반복으로 좋은 결과 가능                          | 초기 모델 설계 복잡, 병렬처리 어려움                                     | Gaussian Process, Tree-structured Parzen Estimator (TPE) 등 사용 |
| Evolutionary Algorithm      | 유전 알고리즘 기반으로 생존/돌연변이/교배를 통해 최적 해 탐색           | 진화 기반 최적화         | 비선형 공간에서도 작동, 전역 최적화에 강함                              | 수렴 속도 느림, 계산 비용 큼                                             | 강화학습, 복잡한 신경망 구조 탐색에 사용   |
| Hyperband / ASHA           | 성능 기반 조기 중단과 리소스 재할당을 통해 효율적으로 탐색             | Successive Halving 기반 | 빠른 탐색, 비효율적인 조합에 자원 낭비 적음                            | 초반 성능이 낮은 모델이 좋은 결과일 수 있음                              | 대규모 분산 환경에서 적합                 |
| Meta-Learning               | 이전 학습 결과를 활용해 새로운 HPO에 빠르게 적응                        | 경험 기반 메타 학습      | 빠른 적응 가능, 기존 데이터 활용                                     | 메타 데이터 구축 필요, 일반화 어려움                                     | AutoML, Task Transfer 기반 학습에 활용     |
| Population-Based Training (PBT) | 여러 모델을 동시에 훈련하고 잘 수행되는 모델로 파라미터 업데이트          | 진화 + 탐색 동시 수행    | 탐색과 학습 병렬 수행, 튜닝 자동화 가능                                | 리소스 많이 소모, 구현 복잡                                               | AlphaStar, LLM 미세조정 등에 활용          |

- **그리드 서치(Grid Search)**  
    - 모든 하이퍼파라미터 조합을 완전 탐색  
    - 단순, 직관적이지만, 계산 비용 매우 큼, 고차원에서는 비효율적  
    - 탐색 방식: 하이퍼파라미터의 가능한 값들을 격자 형태로 구성하고, 모든 조합을 실험  
    - 장점: 전수조사 방식으로 최적값을 반드시 찾을 수 있음  
    - 단점: 차원이 많아질수록 조합 수가 기하급수적으로 늘어나 계산 비효율 발생
- **랜덤 서치(Random Search)**  
    - 무작위 조합으로 탐색  
    - 효율적이고 빠름  
    - 탐색 방식: 각 하이퍼파라미터를 사전 정의된 분포에서 무작위로 샘플링하여 실험  
    - 장점: 탐색 자원을 중요한 파라미터에 집중할 확률이 높음  
    - 단점: 실험 결과의 일관성이 부족하며, 운에 따라 품질 편차 가능  
- **베이지안 최적화(Bayesian Optimization)**
    - 이전 탐색 결과를 바탕으로 다음 탐색 포인트를 탐색 vs 활용(Exploration vs Exploitation) 균형으로 선택 (대표 라이브러리: Optuna, Hyperopt)  
    - 효율적이지만 구현 복잡  
    - 탐색 방식: 가우시안 프로세스(GP), Tree-structured Parzen Estimator(TPE) 등으로 성능을 예측하는 확률 모델 구축 → 다음 실험 위치 예측  
    - 획득 함수(Acquisition Function): Expected Improvement, UCB 등  
    - 장점: 적은 실험으로도 좋은 성능 가능, 계산 효율성 우수  
    - 단점: 모델링 복잡도와 계산 비용이 존재, 범용성은 낮음
- [**진화 알고리즘(Evolutionary Algorithm)**](https://jeongchul.tistory.com/845)  
    - 하이퍼파라미터를 유전 연산자로 변형하면서 탐색, 개체군(Population) 기반 탐색  
    - 핵심 요소
      1. **개체 (Individual)**  
        - 하나의 하이퍼파라미터 조합(예: `learning_rate=0.01, batch_size=64`)
      2. **집단 (Population)**  
        - 여러 개체의 집합
        - 각 세대(generation)마다 전체 집단을 평가하고, 성능이 높은 개체를 중심으로 다음 세대를 생성
      3. **적합도 함수 (Fitness Function)**  
        - 개체의 성능을 평가하는 기준 (예: validation accuracy, validation loss)
        - 최적의 하이퍼파라미터를 찾기 위한 지표
      4. **진화 연산자**
        - Selection(선택): Fitness 기반 선택. 뛰어난 개체가 자손을 남길 확률이 높음
        - Crossover(교배): 두 개체의 하이퍼파라미터를 결합하여 새로운 조합 생성
        - Mutation(돌연변이): 일부 파라미터를 랜덤하게 변경하여 다양성 증가
    - 탐색 방식
      1. 초기 개체군 무작위 생성
      2. 각 개체에 대해 fitness 계산 > 종료 조건 만족 여부 확인
      3. 선택(selection), 교차(crossover), 돌연변이(mutation)를 통해 다음 세대 구성
    - 장점: 전역 최적 탐색에 유리하고, 복잡한 탐색 공간에서 유연함  
    - 단점: 수렴 속도 느리고, 많은 반복이 필요하며 계산 자원 소모 큼 
- [**Hyperband / ASHA (Asynchronous Successive Halving Algorithm)**](https://iyk2h.tistory.com/143)  
    - 자원을 적게 할당한 실험에서 성능이 좋은 조합만 다음 라운드로 진행  
    - 효율적인 조기 종료(Early Stopping)를 포함함
    - 관련 개념
        - Successive Halving (SH)
          - 1) 초기에는 많은 후보를 적은 자원(적은 epoch, 적은 sample 등)으로 학습
          - 2) 성능 좋은 일부만 더 많은 자원을 배분하여 계속 학습
          - 3) 최종적으로 가장 좋은 후보만 남김
        - Hyperband
          - 1) SH를 여러 번 실행하되, 각기 다른 초기 자원(epochs, budget) 설정을 병렬적으로 수행
          - 2) 탐색 공간을 넓게 커버하면서 자원 효율을 높임
        - ASHA (Asynchronous SH)
          - Successive Halving을 **비동기(asynchronous)** 방식으로 실행하여, 느린 실험 때문에 전체 프로세스가 지연되지 않도록 함
    - 탐색 방식
        1. 많은 하이퍼파라미터 후보를 무작위로 선택  
        2. 각 후보를 작은 자원(예: 1 epoch, 10% 데이터)으로 학습  
        3. 성능 상위 일부를 선택하여 자원을 점점 늘려가며 학습 (예: 3 epoch → 9 epoch → 27 epoch)  
        4. 나머지 후보들은 조기 종료(Early Stopping)  
        5. Hyperband: 여러 `bracket`을 병렬 실행 → 자원 분배 전략 다양화  
        6. ASHA: 개별 실험의 완료 여부와 관계없이, 준비된 후보부터 바로 다음 라운드로 승격 (비동기 실행)  
    - **장점**  
        - 자원 효율성 우수: 불필요한 후보에 자원 낭비 최소화  
        - Hyperband: 자원 분배를 여러 bracket으로 분산 → 다양한 자원 스케줄 실험 가능  
        - ASHA: 느린 후보가 전체 탐색을 지연시키지 않음 → 분산 환경에서 확장성 높음  
    - **단점**  
        - 조기 종료 때문에 학습 초반에 성능이 낮지만 후반에 좋아질 후보를 놓칠 수 있음  
        - Hyperband는 bracket 수와 자원 분배 비율 등 추가적인 설정 필요  
    - **활용 예시**  
        - 대규모 신경망 튜닝에서 GPU/TPU 클러스터 효율적으로 활용  
        - [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)에서 Hyperband와 ASHA 지원  
        - AutoML 시스템에서 기본 HPO 알고리즘으로 자주 사용됨   
- 메타러닝(Meta-Learning) 기반 HPO
    - 과거 유사한 문제에서의 튜닝 결과를 학습하여 새로운 문제에 적용  
    - 적은 탐색으로 성능 좋은 하이퍼파라미터 추천  
    - 탐색 방식: 유사 데이터셋이나 태스크에 대해 수집된 튜닝 기록을 학습 → 새로운 문제에서 사전 지식을 바탕으로 초기값 제안  
    - 장점: 빠른 초기 튜닝 가능, 실시간 적응성  
    - 단점: 메타 데이터셋이 충분히 있어야 하고, 새로운 도메인에는 일반화 어려움  
- [Population-Based Training(PBT)](https://rahites.tistory.com/354)
    - 학습 중간에 하이퍼파라미터를 동적으로 조정
    - 여러 개체(모델)를 동시에 학습시키고, 일정 간격마다 성능을 평가하여 **우수한 개체는 활용(Exploit), 성능이 낮은 개체는 탐색(Explore)**으로 전환  
    - 유전 알고리즘의 선택(selection), 돌연변이(mutation) 개념을 차용 
    - 분산 환경에서 매우 유용함
    - **탐색 방식 / Process**  
        1. 여러 모델(개체)을 동시에 다른 하이퍼파라미터 조합으로 학습 시작  
        2. 일정 주기마다 각 모델의 성능 평가  
        3. 성능이 좋은 모델의 **가중치와 하이퍼파라미터를 복제(Exploit)** → 다른 모델에 적용  
        4. 복제된 하이퍼파라미터에 **작은 변화(돌연변이, Explore)**를 주어 새로운 탐색 시도  
        5. 위 과정을 학습 종료까지 반복하여, 학습과 튜닝을 동시에 수행  
    - **활용 예시**  
        - DeepMind **AlphaStar**: 실시간 전략 게임 *스타크래프트 II* 에이전트 학습에 PBT 적용  
        - LLM 튜닝: LoRA/QLoRA 학습 중 Learning rate, Dropout 등을 실시간으로 조정  
        - 강화학습(RL): 환경 적응성을 높이고, 파라미터를 고정하지 않고 계속 개선 

### 2.3 HPO 실무 고려사항
- 계산 자원 관리
    - 분산/병렬 처리 필수
    - GPU/TPU 등 자원에 따른 튜닝 전략 변경 필요
- 조기 종료(Early Stopping) 전략
    - 성능 개선이 없거나 오버피팅이 감지되면 실험 중단
    - ASHA, Hyperband에서 핵심 전략
- 분산/병렬 튜닝
    - 여러 실험을 동시에 실행하여 탐색 시간 절약
    - Ray Tune, Optuna Multi-node 지원
- **AutoML 플랫폼 비교 (Optuna, Ray Tune, KerasTuner 등)**

|플랫폼|특징|장점|단점|활용 예시|
|--|--|--|--|--|
|**Optuna**|베이지안 최적화 기반, TPE(Tree-structured Parzen Estimator) 활용, 동적 탐색 공간 지원|Pruning / Early Stopping 강력, 탐색 효율 우수, Pythonic한 API|분산 처리 기본 지원은 제한적, 병렬성은 Ray 등과 연동 필요|딥러닝 하이퍼파라미터 튜닝 (LR, Dropout, Hidden size 등), Kaggle 대회|
|**Ray Tune**|분산 환경에서 확장성 강력, 다양한 search algorithm(Grid, Random, BOHB, PBT 등) 지원|수천 개 실험 병렬 실행 가능, Hyperband/ASHA와 통합 최적화|설정 복잡, 클러스터 구성 필요|대규모 GPU 클러스터에서 LLM/멀티모달 모델 튜닝|
|**KerasTuner**|Keras/Tensorflow에 통합 용이, 직관적인 API|간단한 구현, 초보자 친화적, TF와 매끄러운 통합|PyTorch/Scikit-learn과는 호환성 낮음, 분산 처리 한계|CNN, RNN, Transformer 등 Keras 모델 튜닝|
|**Hyperopt**|TPE 기반 베이지안 최적화, Random/Annealing도 지원|간단한 코드로 빠른 구현 가능|Optuna 대비 기능 제한 (시각화, pruning 약함)|MLP, Random Forest, SVM 등 전통 ML 모델 튜닝|
|**Google Vizier**|Google 내부 AutoML 플랫폼 (비공개), Bayesian Optimization 기반|대규모 분산 최적화 가능, 안정성 검증됨|비공개 플랫폼이라 외부 사용 불가|Google 내부 서비스 최적화, TPU 클러스터에서 대규모 모델 튜닝|
|**SMAC**|랜덤 포레스트 기반 베이지안 최적화|카테고리형 변수 최적화 강점|딥러닝보다는 전통 ML에 최적화|ML benchmark에서 자주 활용|
|**Microsoft NNI**|다양한 HPO 기법(Grid, Random, BO, PBT 등) 지원, 분산 실행 가능|범용 AutoML 프레임워크, 시각화 도구 포함|설정 다소 복잡|산업용 AutoML 파이프라인 구축|


## 3. 클래스 불균형(Class Imbalanced) 문제 해결
### 3.1 클래스 불균형 개요
- 원인
    - 자연 발생: 예) 이상 탐지(Anomaly Detection), 사기 탐지(Fraud Detection), 의료 진단
    - 데이터 수집 편향: 드문 이벤트는 원천적으로 수집이 어려움
    - 레이블링 비용: 소수 클래스는 수작업 라벨링 비용이 큼
- 영향
    - 모델은 주로 다수 클래스에 집중 → 소수 클래스 재현율(Recall) 급감
    - 정확도(Accuracy) 중심 평가 시 허위 고성능 착시 발생
- 불균형 정도 측정 (불균형 비율, Gini Index 등)
    - **Imbalance Ratio (IR)** :  
      다수 클래스 샘플 수를 소수 클래스 샘플 수로 나눈 비율. 값이 클수록 불균형 심함.  
      예: 다수 900, 소수 100 → IR = 9:1
    - **Gini Index** :  
      불순도를 나타내는 지표. 클래스 분포가 한쪽으로 치우칠수록 값이 커짐.  
      공식: `Gini = 1 - Σ(p_i^2)` (p_i = 클래스 i의 비율)  
      → 균등할수록 낮고, 불균형할수록 높음.
    - **Entropy** :  
      데이터의 불확실성을 측정. 불균형할수록 특정 클래스 확률이 크기 때문에 엔트로피 값이 낮아짐.  
      공식: `Entropy = - Σ(p_i * log2(p_i))`  
      → 값이 높을수록 분포가 균등, 낮을수록 불균형.

### 3.2 데이터 수준(Data-level) 접근
- 오버샘플링(Oversampling, SMOTE)
    - 소수 클래스 데이터를 복제하거나 생성하여 균형 조정
    - 대표 기법: [SMOTE (Synthetic Minority Over-sampling Technique)](http://jaylala.tistory.com/entry/%EB%B6%88%EA%B7%A0%ED%98%95%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%B2%98%EB%A6%AC-%EC%98%A4%EB%B2%84%EC%83%98%ED%94%8C%EB%A7%81Oversampling-SMOTE)
        <img alt="image" src="https://github.com/user-attachments/assets/a35de7d4-1427-49df-b9a6-aab0f4f96e92" />
        - **원리**: 소수 클래스 샘플과 그 주변의 k-최근접 이웃(k-NN)을 선택 → 선형 보간(interpolation)하여 합성 샘플 생성  
        - **장점**: 데이터 다양성 확보, 단순 중복보다 일반화 성능 개선  
        - **단점**: 경계 근처에 불필요한 샘플 생성 가능 → 클래스 간 경계 왜곡 위험  
        - 변형 기법: Borderline-SMOTE (경계 데이터만 증강), ADASYN (적응형 합성) 등 


- 언더샘플링 (Undersampling)
    - 다수 클래스의 일부 데이터를 제거하여 균형 유지
    - 단순 무작위 제거(Random Undersampling)는 정보 손실이 크고, 일반화 성능 저하 가능  
    - 대표 기법:  
        - NearMiss: 다수 클래스 중 소수 클래스와 가까운 샘플만 선택
          <img alt="image" src="https://github.com/user-attachments/assets/fc51a06a-1b66-4497-bb40-bad642df7cf4" />
        - Tomek Links: 클래스 경계에 위치한 불필요한 다수 클래스 샘플 제거  
          <img alt="image" src="https://github.com/user-attachments/assets/e3e4bdb8-d15e-4ab4-9714-256dd111a1c2" />
        - Cluster Centroids: 다수 클래스를 클러스터링 → 중심점(centroid)만 대표로 사용 
          <img alt="image" src="https://github.com/user-attachments/assets/ac1515f5-92b6-42a3-979a-af784a6d183d" />

- 데이터 증강 (Augmentation)
    - 기존 데이터를 변형하거나 노이즈를 추가하여 새로운 샘플을 생성  
    - **이미지 데이터**: 회전, 반전, 이동, 자르기, 밝기/채도 변화, Gaussian Noise 추가, CutMix, Mixup 등  
    - **텍스트 데이터**: 단어 순서 바꾸기, 동의어 치환(Synonym Replacement), 단어 삽입/삭제, EDA(Easy Data Augmentation) 기법  
    - **시계열 데이터**: 윈도우 슬라이싱, 시프트(shift), 잡음 추가, 스펙트로그램 변환 후 증강  

- 합성 데이터 생성 (GAN 기반)
    - GAN(Generative Adversarial Network)을 활용하여 소수 클래스에 대한 **고품질 합성 샘플 생성**  
    - **대표 변형 기법**:  
        - **cGAN (Conditional GAN)**: 클래스 레이블을 조건으로 입력 → 특정 클래스 데이터를 직접 생성 가능  
        - **CTGAN (Conditional Tabular GAN)**: 범주형/연속형 변수 혼합된 **테이블(tabular) 데이터 생성**에 특화  
        - **Tabular GAN**: 테이블 데이터의 분포를 학습하여 새로운 합성 샘플 생성 (의료 데이터, 금융 데이터에서 활용)  
    - **장점**: 기존 분포와 유사한 데이터를 생성하여 모델 학습 강화  
    - **단점**: 학습 불안정성, 모드 붕괴(mode collapse) 문제 발생 가능  

### 3.3 알고리즘 수준(Algorithm-level) 접근
- **클래스 가중치(Class Weight) 조정**  
    - 소수 클래스 샘플에 더 높은 손실 가중치를 부여하여 학습 시 중요도를 강화  
    - 불균형 데이터에서 모델이 다수 클래스에 치우치지 않도록 유도  
    - **예시**:  
        - `sklearn` → `class_weight='balanced'` 옵션  
        - PyTorch → `nn.CrossEntropyLoss(weight=...)`에 클래스별 가중치 벡터 적용  
    - 장점: 간단하고 계산 자원 추가 소모가 적음  
    - 단점: 가중치 값 설정이 어렵고, 지나치면 과적합 위험  
- **비용 민감 학습(Cost-Sensitive Learning)**  
    - 분류 오류에 대해 클래스별로 다른 비용(cost)을 부여하는 학습 방식  
    - 소수 클래스 예측 실패(FN)에 더 큰 비용을 주어 모델이 이를 최소화하도록 학습  
    - 예: 의료 진단에서 암 환자를 놓치면 비용(위험)이 크므로 FN에 더 높은 비용 부여  
    - 장점: 실제 의사결정 상황에 적합  
    - 단점: 비용 함수 설계가 복잡하며 도메인 지식 필요 
- [앙상블 기법 (Bagging, Boosting 변형)](https://data-analysis-science.tistory.com/61)
   <img alt="image" src="https://github.com/user-attachments/assets/38ffc715-b6e7-4766-8077-a5f8ef6227ef" />
    - 여러 약한 학습기를 결합하여 성능 향상  
    - 클래스 불균형 상황에 맞게 **Bagging/Boosting 알고리즘을 변형**하여 활용  
    - **Bagging 기반 기법**  
        - 원리: 데이터 샘플링을 통해 여러 개의 서브셋 생성 → 개별 모델 학습 후 투표/평균  
        - 불균형 처리 변형:  
            - **Balanced Random Forest**: 각 부트스트랩 샘플을 구성할 때, 다수 클래스와 소수 클래스 샘플 수를 맞추어 학습  
            - **EasyEnsemble**: 다수 클래스를 여러 번 언더샘플링하여 여러 서브셋을 만들고, 각각에 학습된 분류기를 앙상블  
    - **Boosting 기반 기법**  
        - 원리: 오분류된 샘플에 가중치를 점점 높여가며 학습 → 소수 클래스 학습 강화  
        - 불균형 처리 변형:  
            - **AdaBoost**: 샘플 가중치를 동적으로 조정  
            - **SMOTEBoost**: 각 단계에서 SMOTE로 합성 샘플 생성 후 Boosting 적용  
            - **RUSBoost**: 단계별 학습 시 다수 클래스 샘플을 무작위 제거(Random Undersampling)하여 불균형 완화  
    - **장점**: 단일 모델보다 성능과 안정성이 뛰어나며, 클래스 불균형 상황에서도 강건한 성능 확보 가능  
    - **단점**: 모델 해석이 어려워지고, 계산 비용이 커질 수 있음 

### 3.4 평가 단계 고려
- 클래스 불균형 상황에서의 적합한 평가 지표
    - **F1-score (macro/micro)**  
        - Precision과 Recall의 조화 평균.  
        - **Macro-F1**: 클래스별 F1을 계산 후 평균 → 클래스 불균형에 민감 (소수 클래스 중요하게 반영).  
        - **Micro-F1**: 전체 샘플 단위로 Precision/Recall을 계산 → 클래스 불균형 영향이 상대적으로 적음.  
    - **PR-AUC (Precision-Recall AUC)**  
        - Precision-Recall 곡선 아래 면적.  
        - 클래스 불균형 상황에서 **ROC-AUC보다 더 민감**하게 소수 클래스 성능 반영
        - 특히 양성 클래스 비율이 매우 낮을 때 적합.  
          - ROC-AUC vs. PR-AUC
            - ROC-AUC: FPR 낮을수록, TPR 높을수록(곡선이 왼쪽, 위로 향할수록) 좋은 모델
            - PR-AUC: Recall 높을수록, Precision 높을수록(곡선이 오른쪽, 위로 향할수록) 좋은 모델
            - 확률 분포 히스토그램 vs. 확률 밀도 분포
                <img alt="image" src="https://github.com/user-attachments/assets/b803c9e2-621f-4c2f-a0f1-f53c58667ddb" />
            - ROC curve vs. PR curve
                <img alt="image" src="https://github.com/user-attachments/assets/7f56b293-28a9-4cff-b230-95f2c03390fd" />

    - **Balanced Accuracy**  
        - 각 클래스의 Recall을 평균하여 산출.  
        - 다수 클래스 위주로 계산되는 단순 Accuracy의 한계를 보완.  
        - 공식: `(Recall_1 + Recall_2 + ... + Recall_n) / n`  

- **Threshold Moving 및 Calibration 기법**  
    - **Threshold Moving**  
        - 기본적으로 분류기는 `p > 0.5`이면 Positive로 분류.  
        - 클래스 불균형 상황에서는 threshold를 조정하여 소수 클래스 탐지를 강화.  
        - 예: threshold를 0.3으로 낮추면 Recall은 올라가지만 Precision은 감소할 수 있음.  
        - 실제 서비스(의료 진단, 이상 탐지)에서 **False Negative 최소화**를 위해 자주 사용.  
    - **Calibration 기법**  
        - 모델의 출력 확률을 **실제 발생 확률에 가깝게 보정**하는 과정.  
        - 대표 기법:  
            - **Platt Scaling**: 로지스틱 회귀를 이용해 확률을 보정.  
            - **Isotonic Regression**: 비모수(non-parametric) 회귀로 확률 보정.  
        - 장점: ROC 커브, PR 커브 기반으로 threshold 설정 시 신뢰성 향상.  
        - 활용: 예측 확률이 의사결정에 직접 사용되는 분야(예: 금융, 의료 리스크 평가).  
- **추가 고려사항**  
    - **Confusion Matrix 기반 지표**: Precision, Recall, Specificity, G-mean 등 불균형 데이터 상황에 더 유용.  
    - **Cost-sensitive Metrics**: 클래스별 오류 비용(cost)을 반영하여 평가.  
    - **Kappa Score, Matthews Correlation Coefficient(MCC)**: 클래스 불균형에도 강건한 지표로 권장. 


## 4. 모델 튜닝과 평가 연계
- HPO와 모델 평가 지표의 관계
  - 튜닝 목적 함수(Objective Function)는 반드시 모델의 평가 지표와 연동되어야 함
  - 회귀: RMSE, MAE 등 → 작을수록 좋음
  - 분류: F1-score, AUC 등 → 클수록 좋음
  - 다중 지표 사용 시
        - 개별 평가 지표를 정규화하여 조합 (예: weighted sum)  
        - 혹은 멀티목적 최적화(Multi-objective Optimization) 적용  
- 클래스 불균형 상황에서의 HPO 전략
    - 단순 Accuracy 기반 튜닝은 소수 클래스 무시 위험  
    - **권장 지표**: macro F1, PR-AUC, balanced accuracy 등  
    - **데이터 수준 보완**: SMOTE, ADASYN 등 오버샘플링 기법 적용  
    - **알고리즘 수준 보완**: class weighting, cost-sensitive learning  
    - **Threshold 조정 포함**: 소수 클래스 탐지를 강화하기 위해 decision threshold를 낮추거나, cost-sensitive threshold 적용  
- **멀티목적 최적화(Multi-objective Optimization)**
  - 여러 성능 지표를 동시에 고려 (예: F1-score vs. Inference Time)
  - 대표 기법: [Pareto Optimization](https://wikidocs.net/253840), NSGA-II
    - **Pareto Optimization**  
        - 여러 목적 함수 간에 **동시에 개선할 수 없는 관계(트레이드오프)**를 고려하여 최적 해를 탐색하는 방법.  
        - **Pareto 최적(Pareto Optimal)**: 어떤 해의 하나의 성능 지표를 개선하면 다른 지표가 반드시 나빠지는 지점.  
        - 결과적으로 단일 최적점이 아닌, **Pareto Front(비지배해 집합)**을 형성 → 의사결정자는 이 중에서 요구사항에 맞는 해를 선택.  
        - 활용: 정확도 vs 연산시간, 모델 크기 vs 성능 등 다중 목적 고려 문제.  
    - **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**  
        - Pareto 최적화를 **진화 알고리즘 기반**으로 효율적으로 구현한 방법.  
        - **핵심 특징**:  
            1. **비지배 정렬 (Non-dominated Sorting)**: 개체들을 Pareto 지배 관계에 따라 계층적으로 정렬.  
            2. **군집 거리(Crowding Distance)**: Pareto front 상에서 다양성을 유지하기 위해 개체 간 거리를 계산.  
            3. **선택/교차/돌연변이 연산**을 통해 새로운 세대 생성.  
        - 장점: 다중 목적 문제에서 **균형 잡힌 Pareto front**를 빠르게 찾을 수 있음.  
        - 활용: 모델 성능 vs 메모리 사용량, 정확도 vs 추론 속도 최적화 등.  
  - Optuna 등에서는 optuna.multi_objective 지원
- **튜닝 과정에서의 과적합 방지 전략**
  - Validation set 고정 및 Cross-validation 병행
  - Early stopping 사용하여 validation loss 증가 시 학습 중단
  - HPO에서 튜닝 과정을 로그로 추적하고, test set은 절대 objective로 사용하지 않도록 주의


## 5. 실무 적용 사례
- 이미지 분류 모델 튜닝 예시
  - 적용 분야: 의료 이미지 분류, 품질 검사, 일반 이미지 인식 등
  - 사용 모델: ResNet, EfficientNet, ConvNeXt 등
  - 튜닝 대상 하이퍼파라미터
    - Optimizer 종류: SGD vs AdamW
    - Learning Rate & Scheduler: CosineAnnealing, ReduceLROnPlateau
    - Batch Size, Weight Decay
    - Data Augmentation 기법: RandomCrop, CutMix, MixUp, AutoAugment
  - 튜닝 기법
    - Optuna + PyTorch Lightning integration
    - Early Stopping + Cosine LR Scheduler
    - 실험 자동화: Weights & Biases (sweep) 또는 MLflow 사용
  - 평가 지표
    - Top-1 Accuracy, Macro-F1, Confusion Matrix 기반 Recall
```python
optuna_trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```
- 텍스트 분류 모델 튜닝 예시
  - 적용 분야: 감성 분석, 뉴스 카테고리 분류, 고객 문의 자동 분류 등
  - 사용 모델: BERT, RoBERTa, KoBERT, Electra
  - 튜닝 대상 하이퍼파라미터
    - Learning rate, Max sequence length
    - Warmup steps, Weight decay
    - Tokenizer truncation/padding 방식
  - 튜닝 기법
    - Huggingface + Optuna integration
    - k-fold CV 기반 tuning
    - Trainer API에서 TrainerCallback으로 metric logging
  - 평가 지표
    - Macro-F1, PR-AUC, Weighted Recall
    - Validation vs. Test performance gap 체크
```python
training_args = TrainingArguments(
  evaluation_strategy="epoch",
  learning_rate=trial.suggest_float("lr", 1e-5, 5e-5),
  ...
)
```
- **대규모 언어 모델(LLM) 튜닝 사례**  
  - 적용 분야: QA, 요약, 대화형 모델 등  
  - 사용 모델: LLaMA, Mistral, Falcon, GPT-NeoX 등  
  - 튜닝 방법  
    - Full fine-tuning 대신 **PEFT(Parameter-Efficient Fine-Tuning)** 기법 사용 (LoRA, QLoRA, Prefix Tuning 등)  
    - Quantization-aware tuning (4bit, 8bit) 적용하여 메모리 효율 개선  
    - RLHF(Reinforcement Learning from Human Feedback) 기반 보상 학습 연계  
  - 튜닝 대상 파라미터  
    - LoRA rank, alpha 값  
    - Target module (attention, FFN layer)  
    - Optimizer (AdamW, Adam8bit)  
  - 평가 지표  
    - 전통 지표: BLEU, ROUGE, Perplexity  
    - 벤치마크: MT-Bench, GPT-judge 기반 평가  

- **멀티모달 모델 튜닝 사례**  
  - 적용 분야: 이미지-텍스트 검색, 비전-언어 질의응답(VQA), 멀티모달 대화  
  - 사용 모델: CLIP, BLIP, Flamingo, LLaVA 등  
  - 튜닝 전략  
    - 텍스트/이미지 인코더별로 **다른 학습률** 적용 (Differential Learning Rate)  
    - Cross-modal fusion layer 및 projection tuning  
    - Vision encoder freeze 후, decoder/adapter만 tuning → 효율적 미세조정  
  - 튜닝 기법  
    - Prompt tuning, Linear probing  
    - Mixed precision, Gradient checkpointing을 통한 효율화  
  - 평가 지표  
    - Retrieval: Recall@K, mAP  
    - Generation: BLEU, CIDEr, GPTScore 


## 6. 튜닝 자동화와 MLOps
- 파이프라인 기반 HPO 자동화
  - ML 파이프라인 도구 (e.g. Kubeflow, Airflow, Vertex AI)를 사용하여 전체 튜닝 흐름 자동화
  - 구성 요소: 데이터 로딩 → 전처리 → 모델 학습 + 튜닝 → 성능 기록 → 최적 모델 저장
- MLflow / Weights & Biases를 통한 실험 관리
  - MLflow : 실험 로그, 파라미터 관리, 모델 아카이빙, API 배포 연계
  - Weights & Biases (wandb) : 대시보드 기반 실험 비교, HPO 스윕 기능, 커스텀 시각화
- (추가) **튜닝-배포-모니터링 연계 자동화**
  - 튜닝 완료된 모델을 자동으로 배포 & 서빙 파이프라인 연계
  - Serving 예시: FastAPI + Docker + Kubernetes
  - 실시간 성능 모니터링 도구 연계: Prometheus, Grafana
  - 성능 저하 감지 시 자동 재튜닝 트리거 (예: concept drift 대응)

