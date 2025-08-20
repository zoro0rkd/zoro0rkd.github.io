## 1. 모델 튜닝 개요
- 모델 튜닝의 정의와 필요성
- 하이퍼파라미터(Hyperparameter)와 파라미터(Parameter) 차이
- (추가) **튜닝 대상 파라미터 우선순위 선정 기준**

## 2. 하이퍼파라미터 최적화(HPO, Hyperparameter Optimization)
### 2.1 HPO 개요
- 정의와 목표
- 탐색 공간(Search Space) 설정
- 목적 함수(Objective Function) 정의
### 2.2 HPO 기법
- 그리드 서치(Grid Search)
- 랜덤 서치(Random Search)
- 베이지안 최적화(Bayesian Optimization)
- 진화 알고리즘(Evolutionary Algorithm)
- Hyperband / ASHA
- (추가) **메타러닝(Meta-Learning) 기반 HPO**
- (추가) **Population-Based Training(PBT)**  
### 2.3 HPO 실무 고려사항
- 계산 자원 관리
- 조기 종료(Early Stopping) 전략
- 분산/병렬 튜닝
- (추가) **AutoML 플랫폼 비교 (Optuna, Ray Tune, KerasTuner 등)**

## 3. 클래스 불균형(Class Imbalanced) 문제 해결
### 3.1 클래스 불균형 개요
- 원인과 영향
- 불균형 정도 측정 (불균형 비율, Gini Index 등)
### 3.2 데이터 수준(Data-level) 접근
- 오버샘플링(Oversampling, SMOTE)
- 언더샘플링(Undersampling)
- 데이터 증강(Augmentation)
- (추가) **합성 데이터 생성(GAN 기반)**
### 3.3 알고리즘 수준(Algorithm-level) 접근
- 클래스 가중치(Class Weight) 조정
- 비용 민감 학습(Cost-Sensitive Learning)
- 앙상블 기법 (Bagging, Boosting 변형)
### 3.4 평가 단계 고려
- 클래스 불균형 상황에서의 적합한 평가 지표
  - PR-AUC, F1-score (macro/micro)
  - Balanced Accuracy
- (추가) **Threshold Moving 및 Calibration 기법**

## 4. 모델 튜닝과 평가 연계
- HPO와 모델 평가 지표의 관계
- 클래스 불균형 상황에서의 HPO 전략
- (추가) **멀티목적 최적화(Multi-objective Optimization)**
- (추가) **튜닝 과정에서의 과적합 방지 전략**

## 5. 실무 적용 사례
- 이미지 분류 모델 튜닝 예시
- 텍스트 분류 모델 튜닝 예시
- (추가) **대규모 언어 모델(LLM) 튜닝 사례**
- (추가) **멀티모달 모델 튜닝 사례**

## 6. 튜닝 자동화와 MLOps
- 파이프라인 기반 HPO 자동화
- MLflow / Weights & Biases를 통한 실험 관리
- (추가) **튜닝-배포-모니터링 연계 자동화**