## 1. XAI 개요
### XAI 정의
AI 시스템이 어떻게 결정을 내렸는지 사람에게 이해할 수 있도록 설명하는 기술과 방법론

### 필요성 및 적용 배경
필요성 | 설명
-- | --
신뢰성 확보 | 사용자가 AI의 판단 근거를 이해해야 신뢰하고 사용할 수 있음
책임성 | AI가 잘못된 판단을 했을 때, 원인을 파악하고 책임소재를 따질 수 있음
규제 준수 | 의료, 금융 등 규제 산업에서는 AI 의사결정 근거를 설명할 수 있어야 함 (예: GDPR)
디버깅 및 개선 | 모델이 오류를 내는 이유를 분석해 성능 개선 가능

- 규제·윤리와의 연계성
- (추가) **AI 윤리 가이드라인과 XAI의 관계**

## 2. XAI의 주요 목표
### 모델의 예측 과정 이해
- **설명**  
  XAI(설명 가능한 인공지능)의 핵심 목표 중 하나는 모델이 내리는 예측의 과정을 인간이 직관적으로 이해할 수 있도록 설명하는 것입니다.  
  예를 들어, 딥러닝 모델이 특정 이미지를 '고양이'로 분류했다면, 어떤 픽셀이나 특성이 결정적으로 작용했는지, 중간 계층에서 어떤 정보가 강조되었는지 시각화하거나, feature importance 혹은 attention 메커니즘 등을 분석함으로써 내부 작동 원리를 투명하게 보여줍니다.
- **활용 방법**  
  - Feature Attribution: 입력 변수의 중요도 분석 (예: SHAP, LIME)
  - Activation Visualization: 중간 계층의 활성화 시각화(예: CNN feature map)
  - Rule Extraction: 복잡한 모델에서 의사결정 규칙 부분 추출

### 의사결정의 투명성 확보
- **설명**  
  XAI는 모델이 어떻게 최종 의사결정에 도달했는지 명확히 드러내어 *블랙박스* 특성을 완화합니다. 의료, 금융, 법률 등 규제가 강화된 분야에서는 모델의 예측 및 추천이 왜 그렇게 나왔는지 설명이 필수적입니다.
- **활용 방법**  
  - Decision Trace: 예측에 이르는 근거 단계별 제공
  - Rule-based Explanation: 특정 조건과 규칙에 기반한 설명 텍스트 자동생성
  - Regulatory Compliance: 모델이 설명 및 해석 가능한 상태임을 입증하여 법적 요건 충족

### 사용자 신뢰 향상
- **설명**  
  사용자가 모델의 예측 결과와 그 과정에 대해 충분한 설명을 받을 때, AI 시스템에 대한 신뢰도가 높아집니다. 이는 실제 활용 및 채택률을 높이는 데 직결됩니다.
- **활용 방법**  
  - 직관적 시각화, 자연어 설명 등 접근성 높은 설명 기능 제공
  - 오해/불신 방지: 설명 부재로 인한 오류나 편향에 대한 불신 해소
  - 사용자 피드백 기반 개선: 설명 결과에 대한 사용자 피드백 활용하여 예측 및 설명의 품질을 점진적으로 향상

### 모델 디버깅 및 성능 개선 활용 (추가)
- **설명**  
  XAI를 활용하면 모델의 오작동이나 예기치 않은 결과에 대해 '왜' 그런 결과가 나왔는지 진단할 수 있어, **디버깅과 성능 개선**에 적극적으로 사용할 수 있습니다.
- **활용 방법**  
  - 오류 분석: 예측 실패 상황에서 중요한 feature 또는 계층을 식별하여 문제점을 pinpoint  
  - 데이터 품질 점검: 모델이 불완전하거나 편향된 데이터에 과도하게 의존하는지 탐색  
  - Hyperparameter 튜닝 및 구조 설계: 설명 결과를 기반으로 모델 구조 혹은 하이퍼파라미터 조절  
  - Continual Learning: 설명 피드백을 기반으로 모델 업데이트 및 재학습 전략 설계

> XAI를 유효하게 적용하면 단순히 "이해"를 넘어서, 실제 산업 현장에서 **신뢰 확보**, **규제 대응**, **시스템 효율화**에 이르는 전방위적 가치 창출이 가능합니다.  
> 향후 XAI 기술은 다양한 분야에서 인간-AI 협력의 근간이 될 것으로 기대됩니다.


## 3. XAI 접근 방식
### 3.1 모델 가시성 수준
모델 가시성 수준은 XAI(설명 가능한 인공지능)에서 모델 내부 구조와 동작 메커니즘에 대해 **얼마나 깊게 파악할 수 있는지**를 의미합니다.  
이 수준에 따라 설명 방식이 크게 달라지며, 보통 아래와 같이 White-box, Black-box, Gray-box 세 가지 접근 방식으로 구분합니다.

### White-box 모델 해석
- **설명**  
  White-box(화이트박스) 접근은 모델의 내부 구조, 파라미터, 알고리즘을 모두 완전히 공개하고 이해할 수 있는 경우에 적용됩니다. 즉, 모델이 "투명"하게 보이므로 직접적으로 수학적, 논리적으로 해석이 가능합니다.
- **특징**  
  - 내부 로직 및 파라미터 액세스 가능
  - 예측 산출 과정 추적 및 설명 용이
  - 흔히 사용하는 모델: 선형 회귀, 의사결정트리, 규칙기반 모델 등
- **예시**  
  - 의사결정트리: 각 노드에서 분할 기준 명확 (if-then 서술 가능)
  - 선형 회귀: 각 feature의 가중치 해석을 통해 결정 과정 설명
- **장점**  
  - 해석 및 구현의 용이성
  - 강력한 규제 및 감시가 요구되는 환경에 적합

### Black-box 모델 해석
- **설명**  
  Black-box(블랙박스) 접근은 모델의 내부 구조나 파라미터를 알 수 없는 경우에 적용됩니다. 복잡한 딥러닝, 앙상블 모델 등 대부분의 최신 AI 시스템이 해당됩니다.  
  이 경우 모델은 단순히 "입력 → 출력"만 제공하며, 내부 동작 원리는 직접적으로 파악할 수 없습니다.
- **특징**  
  - 내부 상세 구조 확인 불가
  - 직접적인 해석 대신 간접적인 방법 사용
  - 주로 사용되는 기법: Feature Attribution, 근사 모델(해석자), 입력 변형에 따른 예측 변화 분석 등
- **예시**  
  - LIME, SHAP: 모델의 입력 데이터를 변화시키며 예측 값에 미치는 영향 분석
  - Counterfactual Explanations: 입력을 약간 수정하여 결과 변화 관찰
- **장점**  
  - 복잡한 사전학습 모델이나 외부 제공 API 모델에도 적용 가능
  - 다양한 분야에 범용적으로 활용 가능

### Gray-box 접근
- **설명**  
  Gray-box(그레이박스) 접근은 White-box와 Black-box의 중간 형태입니다. 일부 모델 내부 정보를 바탕으로 해석하지만, 전체 구조가 완전히 투명하지는 않은 경우입니다.  
  예를 들어, 일부 계층만 해석하거나, 부분적으로 로직을 공개한 혼합형 모델 등에서 활용합니다.
- **특징**  
  - 제한적이나마 내부 정보 접근 가능
  - 부분적 구조 분석과 외부 출력 변화 분석 결합
  - 예측 신뢰도 향상 및 부분적 디버깅에 효과적
- **예시**  
  - 특정 모듈(예: attention layer)만 해석하는 Transformer 모델
  - 앙상블에서 각 개별 모델은 공개하되, 전체 조합/가중치는 비공개인 경우
- **장점**  
  - 모델 복잡성과 해석 용이성 사이의 균형
  - 실무에서 점진적 XAI 적용 계기

> 이처럼 XAI의 접근 방식은 모델에 대한 정보 공개 범위에 따라 적합한 해석 기술 및 설명 전략을 선택하게 됩니다.  
> 실무에서는 상황과 요구에 맞추어 White-box, Black-box, Gray-box 접근법을 혼합 적용하는 사례가 많습니다.

### 3.2 설명 시점
설명 시점은 모델 개발 또는 예측 단계에서 **설명이 생성되는 시기**를 나타냅니다.

#### 사전(Pre-hoc) 설명
- **설명**  
  Pre-hoc(사전) 설명은 모델 자체가 본질적으로 설명 가능하도록 설계되는 방식입니다. 즉, 모델 생성 또는 학습 과정에서부터 결과를 해석하거나 이유를 설명할 수 있게 설계합니다.
- **예시/특징**  
  - 의사결정트리, 선형회귀, 규칙기반 모델처럼 구조적으로 해석이 가능한 모델
  - 각 단계 자체가 명확한 근거를 제시
  - 규제 산업에서 주로 사용

#### 사후(Post-hoc) 설명
- **설명**  
  Post-hoc(사후) 설명은 이미 학습된 모델이나 예측 결과에 대해 추가적인 해석 방법을 통해 설명을 제공하는 방식입니다. 모델이 Black-box라 하더라도, 예측 결과 이후 다양한 방법으로 그 이유를 분석하고 설명합니다.
- **예시/특징**  
  - Feature Attribution(예: SHAP, LIME)
  - Counterfactual Explanation(유사 사례 제시)
  - 입력 변화에 따른 예측 변화 분석, attention map 시각화 등
  - 모델 정확도와 설명력을 독립적으로 높일 수 있음

### 3.3 설명 범위
설명 범위는 **설명이 적용되는 대상의 크기 또는 범위**를 의미합니다.

#### 전역(Global) 설명
- **설명**  
  Global(전역) 설명은 전체 모델의 행동과 예측 정책을 포괄적으로 해석하는 방식입니다. 모델이 어떤 전반적인 의사결정 정책을 사용하는지, 전체적인 feature 중요도는 무엇인지 등을 분석합니다.
- **예시/특징**  
  - 전체 feature importance, 모델 전체 규칙 또는 정책 추출
  - "모델은 X feature에 주로 의존한다"와 같은 일반적 설명
  - 규제 및 감사 목적, 모델 평가 및 수정에 활용

#### 국소(Local) 설명
- **설명**  
  Local(국소) 설명은 특정 예측 결과 또는 예시(샘플)에 대한 개별적 설명에 초점을 둡니다. 특정 입력에 모델이 왜 그런 결정을 내렸는지, 해당 샘플 주변의 설명적 정보만을 제공합니다.
- **예시/특징**  
  - 특정 고객, 환자, 이미지 등 개별 입력에 대한 feature 영향을 설명
  - "이 환자에게 위험 경고가 발생한 이유" 같은 사례 중심 설명
  - LIME, SHAP에서 샘플 단위 importance 산출

> XAI의 **설명 시점**과 **설명 범위**에 따라 사용되는 방법론과 해석 수준에 차이가 생기므로, 요구되는 투명성이나 신뢰도, 산업 분야에 맞추어 적절한 조합을 선택하는 것이 중요합니다.

## 4. 주요 XAI 기법
### 4.1 Feature Importance 기반
Feature Importance(특성 중요도) 기반 기법들은 **모델의 예측에 각 입력 feature(특성)가 얼마나 큰 영향을 미치는지**를 정량적으로 평가합니다. 이를 통해 모델의 작동 원리를 해석하고, 결과에 중요한 인사이트를 얻을 수 있습니다.

#### Permutation Importance  
- **개념**  
  Permutation Importance는 피처(특성)의 중요도를 평가하기 위해 특정 feature의 값을 **무작위로 섞어(permuting)** 모델 성능이 얼마나 저하되는지 측정합니다.  
- **원리와 과정**  
  1. 전체 데이터로 baseline 성능(예: 정확도, RMSE 등)을 측정  
  2. 한 feature의 값을 섞어 해당 feature와 target의 관계를 깨뜨림  
  3. 다시 모델의 성능을 평가  
  4. 성능 저하 정도가 클수록, 그 feature가 더 중요한 것임  
- **특징**  
  - 모델 agnostic: 어떤 타입의 모델에도 적용 가능  
  - 상호작용 또는 비선형 영향까지 반영 가능  
  - 데이터의 분포 및 관계 유지가 중요  
- **장점/단점**  
  - 단점: feature 간 상관성이 높을 때 편향 발생 가능, 계산 비용이 큼

#### Feature Ablation  
- **개념**  
  Feature Ablation은 각 feature를 **제거하거나 비활성화(ablation)** 한 뒤 예측 성능에 미치는 영향을 분석하는 방식입니다.  
- **원리와 과정**  
  1. baseline 모델 성능 측정  
  2. 각 feature를 하나씩 제거(혹은 0으로 대체)하여 모델 예측 실행  
  3. 성능이 얼마나 떨어지는지 평가지표로 비교  
  4. 성능 저하 크기가 클수록 해당 feature가 중요  
- **특징**  
  - 모델 agnostic  
  - 데이터의 자연스러운 분포와 다르게 될 수 있음  
- **장점/단점**  
  - 직관적이고 명확  
  - feature removal이 현실적으로 의미 없는 조합을 만들 수 있음

#### Mutual Information 기반 중요도  
- **개념**  
  Mutual Information(상호 정보량) 기반 중요도는 각 feature가 **target 변수와 공유하는 정보량**을 측정하여 중요도를 정의합니다.  
- **원리와 과정**  
  - Mutual Information은 통계적으로 두 변수 간의 정보 공유 정도, 즉 한 변수가 다른 변수를 얼마나 줄여주는가(불확실성 감소 정도)를 수치로 나타냅니다.  
  - 각 feature와 target 간의 mutual information 값을 계산하여 비교  
- **특징**  
  - 비선형 관계도 포착 가능  
  - 입력 feature와 target 간의 직접적 정보 의존성 분석  
  - 모델 훈련 전 전처리 단계에서도 활용 가능  
- **장점/단점**  
  - 모델을 만들지 않고도 사전적으로 feature 선정에 활용 가능  
  - 고차원, 범주형 데이터에 효율적으로 적용 가능  
  - feature 간 완전한 상관관계가 있으면 과평가 가능

#### 요약  
- **Permutation Importance**: feature를 섞어 예측 성능 변화 측정  
- **Feature Ablation**: feature를 제거하여 성능 변화 측정  
- **Mutual Information 기반 중요도**: feature와 target 간 공유 정보량으로 중요도 평가  

> 이 기법들은 단독으로 또는 조합하여 모델 해석, 데이터 전처리, feature selection 및 모델 디버깅 등 다양한 XAI 실무에 폭넓게 활용됩니다.

### 4.2 Surrogate Model 기반
* 정의
  * 블랙박스 모델의 예측 결과를 해석하기 위해 사용되는 보조 모델 
  * 일반적으로 로지스틱 회귀, 결정 트리 등 해석 가능한 모델로 구성 
  * 블랙박스의 예측 결과를 학습 데이터로 하여 학습시킴
* 예시
  * LIME (Local Interpretable Model-agnostic Explanations)
  * SHAP (SHapley Additive exPlanations): game theory 기반 
  * Surrogate 모델은 전체 혹은 부분적인 예측 과정을 모사하고 설명

#### LIME (Local Interpretable Model-agnostic Explanations)
* 입력 데이터를 약간씩 변화시켜 모델의 출력값이 어떻게 달라지는지를 관찰하며, 해당 입력의 각 요소가 예측 결과에 미치는 영향을 지역적으로(Locally) 설명한다.
* 입력 주변에서 모델을 선형으로 근사하여 해당 예측을 해석하는 대표적인 XAI 기법이다. 모델에 독립적으로 적용 가능하고, 설명 가능성을 높이는 데 자주 활용된다.
* 참고: https://myeonghak.github.io/xai/XAI-LIME(Local-Interpretable-Model-agnostic-Explanation)-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/

#### SHAP (SHapley Additive exPlanations)
SHAP(SHapley Additive exPlanations)는 설명 가능한 인공지능(XAI, Explainable AI) 분야에서 널리 사용되는 기법으로, 복잡한 머신러닝 모델의 예측 결과를 각 입력 특성(피처)별로 해석할 수 있도록 도와줍니다[1][2][3].

SHAP는 게임이론의 **Shapley Value(섀플리 값)** 개념을 기반으로 하며, 각 특성이 예측에 얼마나 기여했는지를 수치로 산출합니다. 이때, 특성의 유무에 따른 모든 가능한 조합을 고려하여, 각 특성이 예측값에 미치는 평균적인 영향력을 측정합니다[4][2][5]. 예를 들어, 어떤 예측 결과가 나왔을 때 각 특성이 그 결과를 얼마나 올렸는지(또는 내렸는지)를 정량적으로 보여줍니다.

SHAP는 모델의 종류에 따라 다양한 Explainer를 제공합니다:
- **KernelExplainer**: KNN, SVM, RandomForest, GBM 등 범용 모델에 적용 가능
- **TreeExplainer**: 트리 기반 모델(예: RandomForest, XGBoost)에 최적화
- **DeepExplainer**: 딥러닝 모델에 적용[4][3]

SHAP의 결과는 시각적으로도 표현할 수 있어, 특정 데이터 하나에 대해 각 특성의 기여도를 한눈에 볼 수 있습니다. 이를 통해 모델의 예측이 왜 그렇게 나왔는지, 어떤 특성이 결정에 중요한 역할을 했는지 명확히 알 수 있습니다[2][6][5].

요약하면, **SHAP는 복잡한 머신러닝 모델의 예측을 각 특성별로 공정하게 분해해 설명해주는 XAI 기법**입니다. 이를 통해 모델의 투명성과 신뢰성을 높이고, 의사결정 과정을 이해할 수 있게 도와줍니다.

출처
[1] Explainable AI Tools: SHAP's power in AI | Opensense Labs https://opensenselabs.com/blog/explainable-ai-tools
[2] 18 SHAP – Interpretable Machine Learning https://christophm.github.io/interpretable-ml-book/shap.html
[3] [XAI] SHAP - 딥러닝 소터디 - 티스토리 https://sotudy.tistory.com/40
[4] [XAI] SHAP (SHapley Additive exPlanation) https://velog.io/@jbeen2/XAI-SHAP-SHapley-Additive-exPlanation
[5] [개념정리]SHAP(Shapley Additive exPlanations) https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%ACSHAPShapley-Additive-exPlanations
[6] Explainable AI Technical Whiteboard Series: SHAP - YouTube https://www.youtube.com/watch?v=0MaLhIx-wKo
[7] [논문리뷰/설명] SHAP: A Unified Approach to Interpreting Model ... https://kicarussays.tistory.com/32
[8] 설명 가능한 AI② XAI(eXplainable AI) 주요 방법론 https://ahha.ai/2024/07/09/xai_methods/
[9] XAI(Explainable AI) (4) Model-agnostic methods: Shapley ... https://blog.naver.com/strong_song/222955944745
[10] [stock prediction] 3.1. 설명가능 AI (XAI), SHAP value https://ag-su.github.io/blog/posts/06.shap_value.html

##### SHAP(XAI) 응용 예시

SHAP는 실제 현장에서 다양한 방식으로 활용됩니다. 대표적인 응용 사례를 아래와 같이 정리할 수 있습니다.

1. **개별 예측 결과 설명**

- **의료 분야**:  
  예를 들어, 자궁경부암 위험을 예측하는 모델에서 SHAP를 사용하면, 한 환자의 예측 결과에 대해 “나이”, “흡연 여부”, “성병 이력” 등 각각의 특성이 예측값을 얼마나 높이거나 낮추는지 시각적으로 설명할 수 있습니다.  
  예측 기준선(전체 평균 예측값)에서 시작해, 각 특성이 예측을 증가(양수 SHAP 값) 또는 감소(음수 SHAP 값)시키는 힘으로 작용함을 화살표로 보여줍니다[1].

- **부동산 가격 예측**:  
  집값 예측 모델에서 SHAP를 적용하면, 특정 집의 예측 가격이 왜 그렇게 나왔는지, 예를 들어 “강가와의 거리”, “범죄율”, “세금 혜택” 등 각 특성이 집값에 미친 영향력을 수치로 분해해 보여줍니다. 이를 통해 사용자는 예측 결과의 근거를 명확히 이해할 수 있습니다[2].

2. **고객 응대 및 맞춤 서비스**

- **콜센터 상담**:  
  콜센터 상담원이 고객별 예측 결과에 대한 SHAP 값을 실시간으로 확인해, 해당 고객에게 가장 중요한 영향을 준 요인을 파악하고, 맞춤형 상담이나 제안을 할 수 있습니다. 예를 들어, 이탈 위험이 높은 고객의 경우 “최근 불만 접수”, “이용 빈도 감소” 등 주요 영향을 미친 요인을 중심으로 대화를 진행할 수 있습니다[3].

3. **학업 성취 관리**

- **학생 맞춤 지원**:  
  학생의 학업 성공 예측 모델에 SHAP를 적용하면, “출석률”, “과제 제출 여부”, “중간고사 성적” 등 개별 특성이 해당 학생의 성취 예측에 얼마나 기여했는지 확인할 수 있습니다. 이를 바탕으로 조기 위험 신호를 파악하고, 학생별 맞춤 지원을 제공할 수 있습니다[3].

4. **모델 전반적 해석 및 피처 중요도 분석**

- **글로벌 피처 중요도 시각화**:  
  전체 데이터셋에 대해 SHAP 값을 계산하면, 각 특성이 전체 예측에 얼마나 중요한지 순위를 매길 수 있습니다. 예를 들어, 주택 가격 예측에서 “방 개수”, “위치”, “면적” 등이 전체적으로 얼마나 중요한지 막대그래프 등으로 시각화할 수 있습니다[4][1].

5. **딥러닝 모델 해석**

- **딥러닝 모델의 블랙박스 해석**:  
  Deep SHAP와 같은 기법을 활용해 복잡한 신경망 모델의 예측 결과도 각 특성별로 분해·설명할 수 있습니다. 이는 의료 영상 진단, 금융 사기 탐지 등 고차원 데이터에서 특히 유용합니다[4].

6. NLP 으로 감정 인식
![스크린샷 2025-06-25 오후 3 39 42](https://github.com/user-attachments/assets/2b133790-2264-4089-b14c-bd2edcc35db9)

요약

SHAP는 개별 예측의 원인 분석, 고객 맞춤 서비스, 학업 지원, 전체 피처 중요도 분석, 딥러닝 모델 해석 등 다양한 실제 문제에서 **모델의 예측 결과를 신뢰할 수 있게 설명**하는 데 폭넓게 활용됩니다.  
이로써 사용자는 “왜 이런 예측이 나왔는가?”를 데이터와 모델 관점에서 구체적으로 이해할 수 있습니다[2][3][4][1].

출처
[1] SHAP (SHapley Additive exPlanations) - TooTouch https://tootouch.github.io/IML/shap/
[2] [개념정리]SHAP(Shapley Additive exPlanations) - velog https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%ACSHAPShapley-Additive-exPlanations
[3] Using SHAP values in real-world applications https://help.qlik.com/en-US/cloud-services/Subsystems/Hub/Content/Sense_Hub/AutoML/shap-applications.htm
[4] SHAP value에 대한 간단한 소개(with Python) - 세종대왕 - 티스토리 https://zzinnam.tistory.com/entry/SHAP-value%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%86%8C%EA%B0%9Cwith-Python
[5] An Introduction to SHAP Values and Machine Learning ... https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
[6] SHAP에 대한 모든 것 - part 1 : Shapley Values 알아보기 https://datanetworkanalysis.github.io/2019/12/23/shap1
[7] [ML] SHAP Plots 샘플 - URBAN COMMUNICATOR - 티스토리 https://narrowmoon.tistory.com/12
[8] [2주차] SHAP (SHapley Additive exPlanation) - velog https://velog.io/@tobigs_xai/2%EC%A3%BC%EC%B0%A8-SHAP-SHapley-Additive-exPlanation
[9] SHAP value에 대한 간단한 소개(with R) - 세종대왕 - 티스토리 https://zzinnam.tistory.com/entry/SHAP-value%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%86%8C%EA%B0%9Cwith-R
[10] 18 SHAP – Interpretable Machine Learning https://christophm.github.io/interpretable-ml-book/shap.html

## 4.3 시각화 기반 XAI 기법

시각화 기반 기법들은 모델의 예측과 관련된 중요 특성, 입력 변화에 따른 반응, 내부 신경망 동작 등을 **그래픽적 또는 시각적으로 표현**하여 이해를 돕습니다. 특히 복잡한 모델의 내부 작동을 직관적으로 파악하는 데 유용합니다.

### Partial Dependence Plot (PDP)  
- **개념**  
  PDP는 특정 feature에 대해 전체 데이터의 예측 값이 어떻게 변화하는지 평균적인 경향성을 시각화한 그래프입니다.  
- **동작 원리**  
  - 관심 feature를 일정 범위의 값으로 변화시키면서, 나머지 feature는 데이터 분포에 따라 유지  
  - 각 값별로 모델 예측값의 평균을 계산해 그래프를 그림  
- **용도**  
  - 비선형 관계 및 feature 영향력 탐색  
  - 전체 모델 관점에서 feature가 예측에 미치는 **전역적(Global)** 효과 분석  
- **한계**  
  - feature 간 강한 상호작용이 있을 때 해석이 왜곡될 수 있음  

### Individual Conditional Expectation (ICE)  
- **개념**  
  ICE는 PDP의 개인화 버전으로, 각 개별 샘플에 대해 특정 feature 값 변화 시 예측 결과 변화를 시각화합니다.  
- **동작 원리**  
  - 각 샘플별 관심 feature를 값 변화시키며 예측 값 연속 추적  
  - 여러 샘플의 곡선을 한 그래프에 겹쳐서 나타냄  
- **용도**  
  - 데이터 다양성 및 개별 특성에 따른 모델 반응 파악  
  - 전역(PDP) 평균 뒤에 숨겨진 개별 행동 차이 분석  
- **장점**  
  - 상대적으로 국소(Local) 설명 성격을 가지면서 전역적 추세와 비교 가능  

### Saliency Map  
- **개념**  
  Saliency Map은 주로 컴퓨터 비전 분야에서 입력 이미지의 어떤 부분이 모델 예측에 가장 강하게 기여하는지를 시각화한 지도입니다.  
- **동작 원리**  
  - 입력 이미지에 대해 모델 출력의 변화율(기울기)을 계산  
  - 기울기가 큰 픽셀 혹은 영역을 강조하여 표시  
- **용도**  
  - CNN 등 딥러닝 모델에서 결정적 영역 해석  
  - 모델의 시각적 집중 영역 확인  
- **확장 기법**  
  - Guided Backpropagation, Grad-CAM 등 다양한 방법으로 더욱 명확한 시각화 제공  

### Attention 시각화 (NLP, Vision Transformer)  
- **개념**  
  Attention 메커니즘은 모델이 입력에서 특정 부분에 얼마나 집중하는지를 가중치로 표현합니다. 이를 시각화하여 어떤 단어나 이미지 영역에 주로 주목하는지 보여줍니다.  
- **동작 원리**  
  - Transformer 기반 모델의 Attention weights를 추출  
  - NLP에서는 문장 내 단어 간 상호작용, 이미지에서는 시각 영역별 가중치 표현  
- **용도**  
  - 자연어 처리(NLP)에서 문장 해석 및 관계 반영 정도 시각화  
  - Vision Transformer에서 중요한 이미지 영역 강조  
  - 모델 작동 방식 및 의사결정 과정 이해에 도움  
- **장점**  
  - 직관적이며, Transformer 구조 해석에 필수적  
  - 다양한 레이어, 헤드 단위로 세밀한 분석 가능

> 시각화 기반 XAI 기법들은 복잡한 모델 결과와 내부 구조를 명확하고 직관적인 형태로 표현하여, 모델 신뢰성 확보와 개발자 및 최종 사용자 이해도를 대폭 높여줍니다. 이를 통해 효과적인 디버깅, 최적화, 사용자 소통이 가능해집니다.

#### CAM
보통 CNN의 구조를 생각해보면, **Input - Conv Layers - FC Layers** 으로 이루어졌습니다. 
CNN의 **마지막 레이어를 FC-Layer 로 Flatten 시키면 각 픽셀들의 위치 정보**를 잃게 됩니다. 
따라서 Classification의 정확도가 아무리 높더라도, 
우리는 그 CNN 이 무엇을 보고 특정 class로 분류했는지 알 수 없었습니다.
2016년 공개된 논문 Learning Deep Features for Discriminative Localization에서는 
FC Layer 대신에, **GAP (Global Average Pooling) 을 적용**하면 
특정 클라스 이미지의 Heat Map 을 생성할 수 있고, 
이 Heat Map 을 통해 CNN 이 어떻게 그 이미지를 특정 클라스로 예측했는지를 이해할 수 있다고 제안함.
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbKNNRO%2FbtqD6lTPEbe%2FfetXI4tsuUvkE12A32SJO0%2Fimg.png)
* FC(Fully connected layer)를 정의하자면, 완전히 연결 되었다라는 뜻으로, 한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로
  2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층.

![](https://tyami.github.io/assets/images/post/DL/2020-10-27-CNN-visualization-Grad-CAM/2020-10-27-cnn-visualization-grad-cam-6-gap.png)
- 입력 이미지 ─▶ [Conv 필터] ─▶ Feature Map
- Feature Map ─▶ [Max-pooling] ─▶ 차원 축소

**GAP (Global Average Pooling)**
Global Average Pooling (GAP) layer을 통해 각각의 feature map 마다 Global average pooling 
(각 feature map에 포함된 모든 원소 값을 평균함)을 시행합니다. 
그 결과 GAP layer로 들어오는 feature map의 channel 수와 동일한 길이의 벡터를 얻게 됩니다. 
(위의 그림에서는 벡터의 길이가 4가 되겠죠.)
GAP 뒤에는 FC layer가 하나 붙어있고 GAP에서 출력한 벡터를 Input으로 줍니다.
아래 그림에서는 GAP을 통해 길이 6인 벡터가 출력되었고 FC layer를 통과시켜 
토르/아이언맨/스파이더맨/캡틴아메리카 중 하나로 Input image를 분류하게 됩니다. 
각 FC layer의 weight는 **학습을 통해서** 구합니다.
해당 weight 1~6을 각 feature map에 곱한 뒤 각 pixel별로 더해주어 최종 heat map을 출력하게 됩니다.


마지막 convolution layer를 통과한 feature map은 input image의 전체 내용을 함축하고 있기 때문입니다. 
때문에 마지막 Feature map이 아닌 중간에 위치한 featue map에서는 CAM을 통해 Heatmap을 추출할 수 없다는 단점이 발생합니다.
* 참고: https://wikidocs.net/135874

- **CAM 상세 설명**  
<img width="1294" height="582" alt="image-4" src="https://github.com/user-attachments/assets/42371cf5-bc09-4eca-8379-d1840bae02c3" />
  1. **GAP**(특징별 전역 평균)을 구한 후 Fully Connected Layer(Feature → Category) 학습  
     → 기존 CNN 구조의 Classifier를 GAP 기반으로 변경 필요  
<img width="1419" height="938" alt="image-5" src="https://github.com/user-attachments/assets/c699079a-0038-4b68-9a9c-e54b9366c02b" />
  2. 각 feature map에 FCL의 weight를 곱하고, pixel-wise 합산하여 히트맵 구성  
<img width="1500" height="764" alt="image-6" src="https://github.com/user-attachments/assets/ecf8943d-b487-47a5-85b0-c983201877f9" />

**CAM 단점**
- Global average pooling layer를 반드시 사용하고 뒤에는 FC layer가 붙어있음.
  - Global Average Pooling에 의지하기 때문에, GAP을 사용하지 않는 모델에는 적용이 불가능
- 해당 FC layer의 weight를 구하기 위해 학습을 시켜야 함.
- 마지막 convolution layer를 통과해 나온 feature map에 대해서만 CAM을 통해 Heat map 추출이 가능함

🔗 [CAM 관련 설명](https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/)

#### Grad-CAM
Grad-CAM은 CAM과 무엇이 다를까요?
- 기존 CNN 모델 구조의 변화 없음. 즉, Global average pooling 없이 FC layer가 두 개 존재
- 기존 CNN 모델의 재학습이 필요 없음. 각 Feature map에 곱해줄 weight를 학습이 아닌 **미분(gradient)을 통해** 구하기 때문.
![](https://wikidocs.net/images/page/135874/Grad_CAM.PNG)

예를 들어 위 그림처럼 k=6개의 feature map을 이용해 y=2, 아이언맨으로 분류한다고 해봅시다.
class c에 대한 점수 y_c (before the softmax)을 각 원소로 미분합니다. (back propagation 하듯이 말이죠.) 이 미분값은 각 Feature map의 원소가 특정 class에 주는 영향력이 됩니다. 각 feature map에 포함된 모든 원소의 미분값을 모두 더하여 neuron importance weight, a를 구하면, 이 a는 해당 feature map이 특정 class에 주는 영향력이 됩니다.

neuron importance weight, a와 각 k개의 Feature map을 곱하여 weight sum of Feature map을 구함
→ ReLU를 취하여 최종 Grad-CAM에 의한 Heatmap이 출력
ReLU를 취한 이유는 오직 관심 있는 class에 positive 영향을 주는 feature에만 관심이 있기 때문입니다. 즉, y_c를 증가시키기 위해서 증가되어야 할 intensity를 가지는 pixel을 말합니다. ReLU를 적용하지 않으면, localization에서 더 나쁜 성능을 보여준다고 합니다.

**CAM vs Grad-CAM 차이**
| 항목    | CAM   | Grad-CAM        |
| ----- | ----- | --------------- |
| 구조 제약 | GAP 필수 | 없음              |
| 설명 방식 | FC layer 가중치 | Gradient (기울기)  |
| 적용 범위 | 제한적   | 대부분의 CNN에 적용 가능 |
| 해석력   | 제한적   | 더 넓고 일반적        |

- Attention 시각화 (NLP, Vision Transformer)

## 4.4 고급/신규 XAI 기법

고급 및 신규 XAI 기법들은 딥러닝 모델과 같은 복잡한 신경망의 해석을 강화하기 위해 개발된 방법들로, 보다 정교하고 심층적인 설명을 제공합니다.

### Integrated Gradients  
- **개념**  
  Integrated Gradients는 입력값과 기준값(baseline) 사이의 경로를 따라 모델 출력에 대한 기울기를 적분하여 feature 중요도를 계산하는 기법입니다.  
- **원리**  
  - 입력과 기준점 간 여러 중간 값을 생성  
  - 각 중간 점에서 기울기를 계산 후 합산(적분)하여 중요도 산출  
- **특징**  
  - 기울기 기반 방법의 잡음 문제 완화  
  - 명확한 이론적 근거와 수학적 보장(axioms) 있음  
- **활용**  
  - 이미지, 텍스트 등 다양한 데이터에 적용 가능  
  - 기울기 소실 문제에 유연하게 대처

### DeepLIFT (Deep Learning Important FeaTures)  
- **개념**  
  DeepLIFT는 뉴럴 네트워크의 각 뉴런 활성화 변화를 기준 상태 대비 누적해서 계산하여 입력 feature의 기여도를 평가하는 방법입니다.  
- **원리**  
  - 기준 입력과 실제 입력 간 활성화 차이를 추적  
  - 각 신경망 계층에서 기여도를 역전파 방식으로 분배  
- **특징**  
  - 기울기 기반 방법 대비 불연속 함수나 flat region 대처에 효과적  
  - 비교적 계산 효율성 높음  
- **활용**  
  - 복잡한 신경망 구조에 신뢰 가능한 설명 제공  
  - 이미지, 텍스트 모델 해석 가능

### Concept Activation Vectors (TCAV)  
- **개념**  
  TCAV는 사람이 이해할 수 있는 개념 단위(예: ‘줄무늬’, ‘날씨’)가 모델의 예측에 미치는 영향력을 측정하는 기법입니다.  
- **원리**  
  - 특정 개념에 해당하는 데이터 집합을 수집  
  - 모델 내부 특정 층(feature layer)의 벡터 공간에서 개념 방향 벡터 생성  
  - 개별 샘플 예측에 기여하는 정도(민감도)를 평가  
- **특징**  
  - 추상적, 인간 친화적 설명 제공  
  - 기능 해석에서 의미 있는 개념 기반 검증 가능  
- **활용**  
  - 의료 영상에서 특정 질병 징후, 얼굴 인식 등 개념 단위 영향도 분석  
  - 모델 설명과 윤리적 검증에 활용

### Counterfactual Explanation  
- **개념**  
  Counterfactual Explanation은 현재 예측 결과를 바꾸기 위해 **입력 데이터를 어떤 방식으로 바꾸어야 하는지(최소 변형)**를 제시하는 설명 방법입니다.  
- **원리**  
  - 특정 목표 결과를 얻기 위한 입력 값의 최소 조정 값 탐색  
  - ‘만약 이 특성이 이렇게 바뀌면 결과가 달라진다’는 인과관계 탐색  
- **특징**  
  - 이해하기 쉽고, 행동 가능한 인사이트 제시  
  - 사용자에게 특정 결정에 영향을 미칠 수 있는 요소 제시  
- **활용**  
  - 신용평가, 채용, 의료 등 의사결정 지원  
  - 개별 사례 중심 국소 설명 제공

> 이들 고급 XAI 기법은 딥러닝 모델의 복잡성을 해소하고, 사용자가 모델 작동 원리를 심도 깊게 이해할 수 있도록 돕는 최신 연구 결과이자 실무 핵심 도구입니다.  
> 특히, 신뢰성 향상, 규제 대응, 사용자 맞춤형 설명 제공에서 큰 역할을 합니다.

## 5. XAI 적용 사례
- 의료 영상 진단
- 금융 신용 평가
- 자율주행
- (추가) **LLM 및 멀티모달 모델의 설명 가능성 연구**

## 6. XAI 성능 평가

XAI(설명 가능한 인공지능) 기법의 성능 평가는 생성된 설명이 얼마나 신뢰할 수 있고 유용한지 판단하는 데 중요합니다. 평가 지표는 주로 설명의 질, 일관성, 인간 이해 가능성 등을 기준으로 합니다.

### Faithfulness (설명의 신뢰성)  
- **의미**  
  설명이 실제 모델의 예측 근거를 얼마나 정확히 반영하고 있는지를 나타냅니다. 즉, 설명이 모델 의사결정 과정과 진짜로 일치하는지를 평가합니다.  
- **평가 방법**  
  - Feature 제거/변경 시 설명에서 중요하다고 한 feature가 예측 결과에 큰 영향을 미치는지 확인  
  - 설명에 기반한 모델 예측 변화 측정 (e.g., Permutation Importance 활용)  
- **중요성**  
  신뢰할 수 없는 설명은 잘못된 이해를 초래할 수 있으므로 가장 핵심적인 평가 요소입니다.

### Consistency  
- **의미**  
  같은 입력 데이터에 대해 유사한 설명 기법이나 비슷한 상황에서, 생성된 설명 결과가 얼마나 일관적인지를 의미합니다.  
- **평가 방법**  
  - 동일 모델, 동일 입력에 여러 번 설명 생성 후 비교  
  - 다른 데이터 샘플이나 유사 입력에 대한 설명 결과 일관성 확인  
- **중요성**  
  일관성 없는 설명은 혼란을 유발하며 신뢰 저하 요소가 됨

### Stability  
- **의미**  
  입력 데이터가 약간 변하거나 노이즈가 추가되었을 때, 설명 결과가 얼마나 변하지 않고 안정적인지를 나타냅니다.  
- **평가 방법**  
  - 입력에 미세한 변동을 가한 뒤 설명의 변화 측정  
  - 안정적 설명은 모델 예측이 변하지 않는 한 비슷한 설명을 제공  
- **중요성**  
  안정성이 낮으면 설명이 불안정하고 불확실함을 시사함

### Human Interpretability (인간 해석 가능성)  
- **의미**  
  생성된 설명이 실제 사용자, 전문가가 이해하고 활용 가능한 정도를 평가합니다.  
- **평가 방법**  
  - 사용자 설문조사, 전문가 인터뷰  
  - 설명의 직관성, 명확성, 활용도 평가  
- **중요성**  
  기술적으로 완벽해도 인간이 이해하지 못하면 실용성이 크게 떨어짐

### (추가) 정량적 vs 정성적 평가 지표 비교  
- **정량적 평가**  
  - 수치화된 척도로 Faithfulness, Consistency, Stability 등 객관적 수치 기반 평가  
  - 예: 설명 기반 feature 중요도 변화, 상관계수, 정확도 변화 등  
  - 장점: 객관적, 반복 가능, 자동화 가능  
  - 단점: 인간 관점 반영 한계, 맥락 정보 부족  

- **정성적 평가**  
  - 인간 중심의 주관적 평가로 해석 용이성, 신뢰, 유용성 평가  
  - 설문조사, 페이퍼 리뷰, 인터뷰 활용  
  - 장점: 실제 사용자의 이해도 및 만족도 반영 가능  
  - 단점: 시간과 비용 소모, 평가자 주관 개입 가능성

> XAI 성능 평가는 기술적 정확성뿐 아니라, 실제 사용자가 신뢰하고 이해할 수 있는지를 모두 반영하는 복합적인 과제입니다.  
> 따라서 정량적 평가와 정성적 평가를 균형 있게 활용하여 설명 기법의 실효성과 신뢰성을 종합적으로 판단하는 것이 바람직합니다.

## 7. XAI 구현 도구
- LIME, SHAP 패키지
- Captum (PyTorch)
- InterpretML
- (추가) **Explainaboard와 같은 벤치마크 툴**

## 8. XAI의 한계와 도전 과제

XAI(설명 가능한 인공지능)는 인공지능 시스템의 투명성과 신뢰성을 높이기 위한 핵심 기술이지만, 여러 한계와 도전 과제를 안고 있습니다. 이러한 문제들을 극복해야 실질적이고 효과적인 설명 제공이 가능합니다.

### 설명의 모호성  
- **내용**  
  설명이 명확하지 않고 다의적일 때 발생하는 문제입니다. 동일한 설명이라도 해석자가 다르게 이해할 수 있어, 오해와 혼란을 초래할 수 있습니다.  
- **원인**  
  - 복잡한 모델 내부 작용을 단순화하는 과정에서 정보 손실  
  - 여러 해석 방법 간 불일치  
  - 기술적 용어 또는 추상적 개념의 사용  
- **결과**  
  설명의 신뢰성 저하 및 실제 활용 어려움

### 계산 비용  
- **내용**  
  XAI 기술은 특히 고차원 데이터와 복잡 모델에 대해 많은 연산 자원과 시간이 소요됩니다.  
- **원인**  
  - 설명 생성 과정에서 모델 반복 평가, 기울기 계산, 여러 입력 변환 등 부가 연산 필요  
  - 실시간 적용 시 성능 저하 문제  
- **결과**  
  실시간 의사결정 지원이나 대규모 데이터에 적용하는 데 한계 발생

### 복잡 모델 적용 시 제약  
- **내용**  
  딥러닝과 같은 복잡한 모델은 내부 작동 원리의 비선형성 및 고차원성 때문에 해석이 어렵습니다.  
- **문제점**  
  - 완전한 투명성 확보가 어려움  
  - 해석 기법이 근사적, 국소적 설명에 치중하며 전역적 이해 어려움  
  - 복잡성으로 인해 설명의 정확도가 떨어질 수 있음  
- **결과**  
  설명의 한계로 인한 신뢰성 문제 및 규제 적용의 어려움


### (추가) 설명 결과의 사회적·법적 해석 문제  
- **내용**  
  설명된 결과가 사회적, 법적 맥락에서 다르게 해석될 수 있어 책임소재, 투명성, 공정성 문제로 이어질 수 있습니다.  
- **사례 및 이슈**  
  - 설명이 특정 집단에 불리하게 해석되어 차별적 판단 유발 가능성  
  - 법적 규제에서 요구하는 명확한 설명과 기술적 설명 간 괴리  
  - 설명이 장황하거나 불충분해 법적 증거로 인정받기 어려움  
- **중요성**  
  AI 윤리, 규제 준수, 사회적 신뢰 확보를 위해 설명의 **사회적·법적 유효성**을 담보하는 추가 연구와 정책 마련 필요

> XAI 분야는 기술적 진보 외에도, 계산 자원 문제, 사회문화적 수용성, 법률적 정합성 문제 등 다차원적 도전에 직면해 있습니다.  
> 이를 해결하기 위해서는 학제 간 연구와 산업계, 법제도권의 긴밀한 협력이 필수적입니다.

## 기타
### 해석가능성(Interpretability)와 설명가능성(Explainability)의 차이

머신러닝과 인공지능 분야에서 **해석가능성(Interpretability)**와 **설명가능성(Explainability)**은 종종 혼용되지만, 실제로는 구분되는 개념입니다. 아래에서 각각의 개념과 차이점을 명확히 정리해드립니다.

#### 1. 해석가능성(Interpretability)

- **정의**:  
  모델의 내부 구조, 작동 원리, 각 구성 요소(특성, 파라미터 등)가 예측에 어떻게 기여하는지 사람이 직관적으로 이해할 수 있는 정도.
- **예시**:  
  - 선형 회귀, 얕은 결정나무, 희소 선형 모델 등은 각 특성의 영향(가중치, 분기 등)을 직접 확인할 수 있어 해석가능성이 높음.
  - “이 특성의 값이 커지면 결과가 얼마나 변하는가?”를 바로 알 수 있음.
- **특징**:  
  - 모델 자체가 단순하고 투명할수록 해석가능성이 높음.
  - 추가적인 도구 없이도 모델의 동작 원리를 파악할 수 있음.

#### 2. 설명가능성(Explainability)

- **정의**:  
  모델의 예측 결과에 대해 사람이 이해할 수 있는 방식으로 “왜 그런 결과가 나왔는지”를 설명해 주는 능력.  
  즉, 복잡하거나 블랙박스 모델의 예측에 대해 추가적인 설명 기법을 통해 원인을 밝히는 것.
- **예시**:  
  - LIME, SHAP, Counterfactual Explanation 등은 복잡한 딥러닝 모델의 예측 결과에 대해 각 특성의 기여도, 원인, 대안적 시나리오 등을 설명해줌.
  - “이 입력이 이런 결과를 낸 이유는 무엇인가?” 또는 “결과를 바꾸려면 무엇을 바꿔야 하는가?”를 설명함.
- **특징**:  
  - 모델이 복잡하거나 블랙박스일 때, 별도의 설명 기법을 사용해 결과를 해석·설명함.
  - 설명가능성은 해석가능성을 포함할 수 있지만, 더 넓은 개념임.

#### 3. 차이점 요약

| 구분           | 해석가능성(Interpretability)         | 설명가능성(Explainability)                |
|----------------|--------------------------------------|-------------------------------------------|
| **대상**       | 모델 구조 및 내부 메커니즘           | 모델의 예측 결과(출력)                    |
| **방법**       | 직접적·내재적(모델 자체로 이해)      | 간접적·외재적(추가 기법으로 설명)         |
| **적용 대상**  | 단순·투명 모델                      | 복잡·블랙박스 모델 포함 전체              |
| **예시**       | 선형 회귀, 결정나무                  | LIME, SHAP, Counterfactual, Feature importance 등 |
| **질문**       | “모델이 어떻게 동작하는가?”          | “왜 이런 결과가 나왔는가?”                |

#### 4. 한 줄 요약

- **해석가능성**:  
  모델 자체가 얼마나 직관적으로 이해될 수 있는가?
- **설명가능성**:  
  (모델이 복잡하더라도) 결과에 대해 사람이 납득할 수 있게 설명해줄 수 있는가?

#### 참고:  
- 해석가능한 모델은 별도의 설명 기법 없이도 설명이 가능하지만, 설명가능성은 복잡한 모델에 대해 추가적인 설명 기법을 도입하는 것이 특징입니다.  
- 실무에서는 “설명가능성”이 더 넓은 개념으로 사용됩니다.