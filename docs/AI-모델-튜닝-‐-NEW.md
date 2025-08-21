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

- (추가) **튜닝 대상 파라미터 우선순위 선정 기준**
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
 - 그리드 서치(Grid Search)
    - 모든 하이퍼파라미터 조합을 완전 탐색
    - 단순, 직관적이지만, 계산 비용 매우 큼, 고차원에서는 비효율적
- 랜덤 서치(Random Search)
    - 무작위 조합으로 탐색
    - 효율적이고 빠름
- 베이지안 최적화(Bayesian Optimization)
    - 이전 탐색 결과를 바탕으로 다음 탐색 포인트를 탐색 vs 활용(Exploration vs Exploitation) 균형으로 선택(대표 라이브러리: Optuna, Hyperopt)
    - 효율적이지만 구현 복잡
- [진화 알고리즘(Evolutionary Algorithm)](https://jeongchul.tistory.com/845)
    - 하이퍼파라미터를 유전 연산자로 변형하면서 탐색, 개체군(Population) 기반 탐색
    - 적합도 함수(fitness)를 기준으로 다음 세대를 구성
- [Hyperband / ASHA (Asynchronous Successive Halving Algorithm)](https://iyk2h.tistory.com/143)
    - 자원을 적게 할당한 실험에서 성능이 좋은 조합만 다음 라운드로 진행
    - 효율적인 조기 종료(Early Stopping)를 포함함
- (추가) 메타러닝(Meta-Learning) 기반 HPO
    - 과거 유사한 문제에서의 튜닝 결과를 학습하여 새로운 문제에 적용
    - 적은 탐색으로 성능 좋은 하이퍼파라미터 추천
- (추가) [Population-Based Training(PBT)](https://rahites.tistory.com/354)
    - 학습 중간에 하이퍼파라미터를 동적으로 조정
    - 탐색(explore) + 활용(exploit)을 반복
    - 분산 환경에서 매우 유용함

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
- (추가) **AutoML 플랫폼 비교 (Optuna, Ray Tune, KerasTuner 등)**

|플랫폼|특징|
|--|--|
|Optuna | 베이지안 기반, pruning/early stopping 잘 지원|
|Ray Tune | 분산 튜닝 강력, 다양한 search algorithm 지원|
|KerasTuner | Keras/Tensorflow 통합 용이, 간단한 구현|
|Hyperopt | Tree-structured Parzen Estimator 사용|
|Google Vizier |구글 내부 AutoML 최적화 플랫폼 (비공개)|


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
    - Imbalance Ratio (IR) : 다수 클래스 수 / 소수 클래스 수
    - Gini Index : 불순도 측정 (높을수록 불균형)
    - Entropy : 정보 불확실성 기반 불균형 측정

### 3.2 데이터 수준(Data-level) 접근
- 오버샘플링(Oversampling, SMOTE)
    - 소수 클래스 데이터를 복제하거나 생성하여 균형 조정
    - 대표 기법: [SMOTE (Synthetic Minority Over-sampling Technique)](http://jaylala.tistory.com/entry/%EB%B6%88%EA%B7%A0%ED%98%95%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%B2%98%EB%A6%AC-%EC%98%A4%EB%B2%84%EC%83%98%ED%94%8C%EB%A7%81Oversampling-SMOTE)
        - k-NN 기반으로 소수 클래스 간 선형 보간을 통해 합성 샘플 생성
        - 원본 데이터 분포 왜곡 가능
- 언더샘플링 (Undersampling)
    - 다수 클래스의 일부 데이터를 제거하여 균형 유지
    - 정보 손실 및 일반화 성능 저하 위험
- 데이터 증강 (Augmentation)
    - 이미지/텍스트 도메인에서 변형을 가해 데이터 수 증가
    - 예: 회전, 노이즈 추가, 단어 순서 변경 등
- (추가) 합성 데이터 생성 (GAN 기반)
    - GAN을 활용하여 소수 클래스에 대한 실제 같은 데이터 생성
    - 예: cGAN, CTGAN, Tabular GAN
    - 이미지, 탭형(tabular) 데이터 모두에 응용 가능

### 3.3 알고리즘 수준(Algorithm-level) 접근
- 클래스 가중치(Class Weight) 조정
    - 소수 클래스에 더 높은 손실 가중치 부여
    - sklearn 예시: class_weight='balanced'
    - 대부분의 딥러닝 프레임워크에서도 손실 함수에 적용 가능 (e.g., CrossEntropyLoss(weight=...))
- 비용 민감 학습(Cost-Sensitive Learning)
    - 클래스별 분류 오류 비용을 다르게 설정
    - 예: 소수 클래스 예측 실패 시 비용 증가
- [앙상블 기법 (Bagging, Boosting 변형)](https://data-analysis-science.tistory.com/61)
    - Bagging: 클래스별로 균형 잡힌 서브셋에 모델 학습 (예: Balanced Random Forest)
    - Boosting: 오분류된 소수 클래스 샘플에 가중치 강화 (예: AdaBoost, SMOTEBoost, RUSBoost)

### 3.4 평가 단계 고려
- 클래스 불균형 상황에서의 적합한 평가 지표
  - F1-score (macro/micro): Precision과 Recall의 조화 평균. macro는 클래스별 평균, micro는 전체 샘플 기준
  - PR-AUC: Precision-Recall 곡선 하단 면적. 불균형 상황에서 ROC-AUC보다 더 민감
  - Balanced Accuracy: 각 클래스의 recall을 평균한 지표 → imbalance에 민감
- (추가) **Threshold Moving 및 Calibration 기법**
    - Threshold Moving: 기본 0.5 결정 경계(threshold)를 조정하여 소수 클래스 탐지 강화
        - 예: threshold = 0.3으로 낮춰 recall 증가
    - Calibration 기법: 모델의 출력 확률을 신뢰 가능한 확률로 변환
        - 대표 기법: Platt Scaling, Isotonic Regression
        - ROC 커브 아래 면적 최적화 외에도 threshold 선정에 활용


## 4. 모델 튜닝과 평가 연계
- HPO와 모델 평가 지표의 관계
  - 튜닝 목적 함수(Objective Function)는 반드시 모델의 평가 지표와 연동되어야 함
  - 회귀: RMSE, MAE 등 → 작을수록 좋음
  - 분류: F1-score, AUC 등 → 클수록 좋음
  - 다중 지표 사용 시: 개별 평가 지표를 정규화하여 조합하거나 멀티목적 최적화 사용
- 클래스 불균형 상황에서의 HPO 전략
  - 단순 Accuracy 기반 튜닝은 소수 클래스 무시 위험 → macro F1, PR-AUC, balanced accuracy 등 사용 권장
  - SMOTE, class weighting 등과 연동한 HPO 전략 필요
  - 모델이 소수 클래스를 무시할 수 없도록 loss 가중치나 threshold 조정 포함
- (추가) **멀티목적 최적화(Multi-objective Optimization)**
  - 여러 성능 지표를 동시에 고려 (예: F1-score vs. Inference Time)
  - 대표 기법: [Pareto Optimization](https://wikidocs.net/253840), NSGA-II
  - Optuna 등에서는 optuna.multi_objective 지원
- (추가) **튜닝 과정에서의 과적합 방지 전략**
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
- (추가) **대규모 언어 모델(LLM) 튜닝 사례**
- (추가) **멀티모달 모델 튜닝 사례**


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

