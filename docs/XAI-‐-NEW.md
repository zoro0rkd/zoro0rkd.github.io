## 1. XAI 개요
- XAI 정의
- 필요성 및 적용 배경
- 규제·윤리와의 연계성
- (추가) **AI 윤리 가이드라인과 XAI의 관계**

## 2. XAI의 주요 목표
- 모델의 예측 과정 이해
- 의사결정의 투명성 확보
- 사용자 신뢰 향상
- (추가) **모델 디버깅 및 성능 개선 활용**

## 3. XAI 접근 방식
### 3.1 모델 가시성 수준
- **White-box** 모델 해석
- **Black-box** 모델 해석
- **Gray-box** 접근

### 3.2 설명 시점
- 사전(Pre-hoc) 설명
- 사후(Post-hoc) 설명

### 3.3 설명 범위
- 전역(Global) 설명
- 국소(Local) 설명

## 4. 주요 XAI 기법
### 4.1 Feature Importance 기반
- Permutation Importance
- Feature Ablation
- (추가) **Mutual Information 기반 중요도**

### 4.2 Surrogate Model 기반
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)

### 4.3 시각화 기반
- Partial Dependence Plot (PDP)
- Individual Conditional Expectation (ICE)
- Saliency Map, Grad-CAM (이미지)
- Attention 시각화 (NLP, Vision Transformer)

### 4.4 고급/신규 기법 (추가)
- Integrated Gradients
- DeepLIFT
- Concept Activation Vectors (TCAV)
- Counterfactual Explanation

## 5. XAI 적용 사례
- 의료 영상 진단
- 금융 신용 평가
- 자율주행
- (추가) **LLM 및 멀티모달 모델의 설명 가능성 연구**

## 6. XAI 성능 평가
- Faithfulness (설명의 신뢰성)
- Consistency
- Stability
- Human Interpretability
- (추가) **정량적 vs 정성적 평가 지표 비교**

## 7. XAI 구현 도구
- LIME, SHAP 패키지
- Captum (PyTorch)
- InterpretML
- (추가) **Explainaboard와 같은 벤치마크 툴**

## 8. XAI의 한계와 도전 과제
- 설명의 모호성
- 계산 비용
- 복잡 모델 적용 시 제약
- (추가) **설명 결과의 사회적·법적 해석 문제**