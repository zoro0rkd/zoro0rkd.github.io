# Zero‐shot Learning
Zero-shot Learning(ZSL)은 훈련 과정에서 한 번도 본 적 없는 클래스나 개체에 대해 모델이 예측을 수행할 수 있게 하는 학습 방식입니다. 즉, 기존에 레이블된 데이터가 없던 새로운 상황에 대처할 수 있는 AI 모델을 만드는 것이 핵심입니다.

## 1. 개념 요약
|항목|설명|
|---|---|
|정의|학습 데이터에 존재하지 않는 클래스에 대해서도 예측이 가능하도록 하는 학습 기법|정의|
|핵심 아이디어|클래스 간의 **추상적인 관계(속성, 의미, 설명 등)**를 활용하여 일반화|
|목적|라벨링 비용 절감 및 새로운 상황에 대한 유연한 대응|

## 2. 예시

🐶 개, 🐱 고양이로만 학습된 이미지 분류 모델이 말(🐴) 사진을 처음 보고도 “말”이라고 예측

### 어떻게 가능한가?
* “말”의 속성(네 다리, 꼬리 있음, 초식동물 등)이 다른 동물들과의 관계로부터 추론됨
* **속성 벡터(attribute vector)**나 언어 설명(text embedding) 등을 사용해 비교

## 3. 작동 방식

### 일반 구조
1. Seen Classes (학습된 클래스): A, B, C 클래스에 대해 학습
2. Unseen Classes (보지 못한 클래스): D 클래스에 대해 예측 수행
3. 공통 공간: 이미지와 클래스 설명을 같은 벡터 공간에 매핑 (예: CLIP 모델)

## 4. 주요 기술 요소
|기술 요소|설명|
|-------|---|
|속성 기반 학습|클래스마다 사전 정의된 속성(attribute)을 이용 (ex: 색, 형태, 기능 등)|
|임베딩 기반 학습|텍스트 임베딩(BERT, Word2Vec)과 이미지 임베딩을 같은 공간에 위치|
|멀티모달 모델|CLIP(OpenAI)과 같이 이미지+텍스트를 함께 학습하여 일반화 능력 확보|

## 5. 확장 개념
### Generalized Zero-shot Learning (GZSL)
* 보던 클래스와 안 보던 클래스 모두를 구분
* 현실 적용성이 더 높지만 균형 잡힌 성능 확보가 어려움

## 6. 대표 모델
|모델|특징|
|---|---|
|CLIP (OpenAI)|텍스트 설명만 보고 이미지를 분류하는 대표적인 Zero-shot 모델|
|DeViSE|이미지와 텍스트를 동일 임베딩 공간으로 매핑|
|ZSL-RNN|자연어 설명 기반으로 이미지 분류|

## 7. 장단점
|장점|단점|
|---|---|
|● 라벨 비용 절감<br>● 일반화 능력 우수<br>● 실제 환경 대응력↑|● 추론 정확도 낮을 수 있음<br>● 클래스 설명 품질에 따라 성능 차이 발생|

## 8. 실생활 활용 예
|분야|적용 예|
|---|-----|
|비전 (CV)|새로운 제품, 동물, 위험 상황 자동 인식|
|언어 (NLP)|새로운 질문 유형 처리, 다국어 이해|
|로봇 제어|처음 보는 명령어에도 행동 수행 가능|

## 9. 예제 코드: OpenAI CLIP을 활용한 Zero-shot 이미지 분류
Hugging Face의 CLIPProcessor와 CLIPModel을 사용
````Python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 1. 모델 & 전처리기 불러오기
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 테스트 이미지 불러오기
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Arabian_Horse_in_Motion.jpg/640px-Arabian_Horse_in_Motion.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. 후보 클래스 정의 (텍스트)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a horse", "a photo of a cow"]

# 4. 전처리 & 예측
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# 5. 결과 계산
logits_per_image = outputs.logits_per_image  # shape = [1, N]
probs = logits_per_image.softmax(dim=1)

# 6. 결과 출력
for label, prob in zip(labels, probs[0]):
    print(f"{label:<25} → {prob.item():.4f}")
````

출력 결과
````Python
a photo of a cat           → 0.0012
a photo of a dog           → 0.0031
a photo of a horse         → 0.9934   ← ✅ 예측 성공!
a photo of a cow           → 0.0023
````

# Chain‐of‐Thought Prompting (CoT)
## 개요
**Chain-of-Thought Prompting(CoT)**은 대형 언어 모델(Large Language Model, LLM)에게 **단순한 정답 출력이 아닌, 사고 과정(reasoning process)**을 따라가도록 유도하는 Prompt 설계 기법입니다.<br>
### 핵심 개념: “정답을 바로 요구하지 말고, 푸는 과정을 먼저 유도하라”

## 기본 개념
|구분|내용|
|---|---|
|정의|LLM이 문제를 푸는 사고 과정을 단계별로 출력하게 유도하는 프롬프트 설계 방식|
|목적|복잡한 문제에 대해 더 정확하고 논리적인 결과를 얻기 위함|
|적용 분야|수학 계산, 논리 추론, 코딩, 복합 질의응답 등|

## 예시: 일반 Prompt vs CoT Prompt
### 일반 Prompt
````Text
Q: 철수가 사과 3개를 샀고, 5개를 더 받았습니다. 몇 개가 되었나요?
A: 8개
````
### Chain-of-Thought Prompt
````Text
Q: 철수가 사과 3개를 샀고, 5개를 더 받았습니다. 몇 개가 되었나요?
A: 철수는 3개의 사과를 샀습니다. 그리고 5개를 더 받았습니다. 따라서 3 + 5 = 8개입니다. 정답은 8개입니다.
````
### 중간 reasoning step을 거침으로써, 모델이 “왜 그렇게 답했는지” 추론하게 만듭니다.

## CoT 기법의 효과
|항목|일반 Prompt|Chain-of-Thought|
|---|----------|----------------|
|간단한 질문|정확도 높음|비슷|
|복잡한 문제|실수 많음|정답률 향상|
|설명 가능성|낮음|높음|
|활용성|제한적|복합적 추론에 매우 유용|

## 실습 코드 예시 (OpenAI GPT API 사용)
````Python
import openai

openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "당신은 사고 과정을 잘 설명하는 교사입니다."},
        {"role": "user", "content": "철수는 사과 3개를 가지고 있었고, 5개를 더 받았습니다. 총 몇 개입니까? 사고 과정을 말해주세요."}
    ]
)

print(response['choices'][0]['message']['content'])
````
## Chain-of-Thought과 연관된 개념
|개념|설명|
|---|---|
|Self-consistency|CoT를 여러 번 생성하고 가장 많이 나온 답을 선택|
|Tree-of-Thought|복수의 사고 경로(branch)와 평가 메커니즘을 결합한 고도화 방식|
|Least-to-most prompting|쉬운 질문부터 단계적으로 확장하여 해결 유도|

## 참고 논문 및 자료
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Google, 2022)](https://arxiv.org/abs/2201.11903)
* [Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)
* [OpenAI Cookbook](https://github.com/openai/openai-cookbook)


## Reinforcement Learning from Human Feedback (RLHF)
**Reinforcement Learning from Human Feedback (RLHF)**는 인간의 평가를 보상 신호로 활용해 AI 모델을 최적화하는 기계 학습 기법입니다. 이 방법은 복잡하거나 명확히 정의하기 어려운 작업(예: 자연스러운 대화 생성)에서 모델이 인간의 의도와 선호도에 부합하도록 조정하는 데 주로 적용됩니다.  

### RLHF의 핵심 개념  
1. **목적**:  
   - 기존 강화학습(RL)의 보상 함수 설계 한계를 해결합니다. 환경 자체의 보상 신호만으로는 인간의 주관적 기준(예: 유머, 창의성)을 반영하기 어렵기 때문입니다.  
   - 인간 피드백을 통해 **정렬(Alignment)**을 개선해 AI의 출력이 인간의 가치관과 일치하도록 유도합니다[1][2][4].  

2. **적용 분야**:  
   - 대규모 언어 모델(LLM)의 성능 향상(예: ChatGPT, Claude).  
   - 생성형 AI의 출력 품질 최적화(예: 자연스러운 대화, 창의적 텍스트 생성)[1][3][5].  

### RLHF 작동 단계  
RLHF는 일반적으로 **3단계 프로세스**로 구현됩니다:  

| 단계 | 목적 | 주요 작업 |  
|-------|-------|-----------|  
| **1. 지도 미세 조정**<br>(Supervised Fine-Tuning) | 초기 모델 준비 | 인간이 작성한 고품질 데이터(프롬프트-응답 쌍)로 기존 모델을 추가 학습[7][5]. |  
| **2. 보상 모델 학습**<br>(Reward Modeling) | 인간 선호도 예측 | 동일 프롬프트에 대한 여러 모델 출력을 인간이 순위 매기고, 이 데이터로 보상 모델 훈련[2][5]. |  
| **3. 강화 학습 적용**<br>(Reinforcement Learning) | 정책 최적화 | 보상 모델의 피드백을 사용해 PPO(Proximal Policy Optimization) 등으로 모델 업데이트[7][5].  

### 장점과 한계  
#### ⚡ **장점**  
- **정렬 향상**: 인간의 주관적 기준(유용성, 안전성)을 반영한 출력 생성[3][5].  
- **효율성**: 자동화된 보상 함수 설계보다 복잡한 태스크에 적합[2][4].  
- **비용 절감**: RLHF 최적화된 모델은 추론 시 계산 자원을 줄일 수 있습니다[5].  

#### ⚠️ **한계**  
- **인간 피드백의 모호성**: 평가자 간 기준 불일치로 인한 학습 편향 발생 가능[7][4].  
- **Reward Hacking**: 보상 모델의 결함을 악용해 비정상적 출력 생성 위험[7][4].  
- **고비용**: 대규모 인간 평가 데이터셋 구축에 리소스 소요[7][5].  

> **사례**: ChatGPT는 RLHF를 적용해 1) 인간 작성 응답으로 초기 튜닝, 2) 40여 명의 평가자가 답변 순위 지정, 3) PPO로 정책 최적화하는 3단계를 거쳤습니다[7][5].  

### 발전 방향  
- **RL from AI Feedback (RLAIF)**: 인간 대신 AI가 피드백을 제공해 비용 절감[7].  
- **자기 지도 학습**: 모델이 자체 출력을 미세 조정하는 방법 연구 진행 중[7][5].  

**요약**: RLHF는 인간의 주관적 판단을 AI 학습에 통합함으로써, 특히 LLM의 정렬 문제 해결에 혁신적 기여를 했습니다. 그러나 피드백 품질 관리와 비용 효율성 개선이 지속적인 과제입니다.

출처
[1] RLHF란 무엇인가요? - 인간 피드백을 통한 강화 학습 설명 - AWS https://aws.amazon.com/ko/what-is/reinforcement-learning-from-human-feedback/
[2] 휴먼 피드백을 통한 강화 학습(RLHF)이란 무엇인가요? - IBM https://www.ibm.com/kr-ko/think/topics/rlhf
[3] RLHF(인간 피드백 기반 강화 학습)란? - ServiceNow https://www.servicenow.com/kr/ai/what-is-rlhf.html
[4] RLHF: Reinforcement Learning from Human Feedback - 위키독스 https://wikidocs.net/225547
[5] RLHF(인간 피드백 기반 강화 학습)란? | appen 에펜 https://kr.appen.com/blog/rlhf-benefits-llm/
[6] 10분만에 RLHF(Reinforcement Learning with Human Feedback ... https://zeequ.tistory.com/11
[7] RLHF란? https://velog.io/@nellcome/RLHF%EB%9E%80
[8] RLHF 설명 (Training language models to follow ... - 유니의 공부 https://process-mining.tistory.com/220
[9] RLHF(Reinforcement Learning from Human Feedback)구현해보기_(1) https://coco0414.tistory.com/101
[10] AI학습방법의 종류 https://brunch.co.kr/@chorong92/32

## Contrastive Language-Image Pre-training (CLIP)

**Contrastive Language-Image Pre-training (CLIP)**은 OpenAI가 개발한 멀티모달 인공지능 모델로, 이미지와 텍스트 간의 의미적 관계를 대규모 데이터로 학습해 **제로샷(zero-shot) 이미지 분류**를 가능하게 하는 혁신적 기술입니다[2][4].  

### 핵심 원리  
CLIP은 **이중 인코더 구조**를 기반으로 합니다:  
- **이미지 인코더**: Vision Transformer(ViT) 또는 ResNet 기반으로 이미지를 벡터 임베딩으로 변환[4][5].  
- **텍스트 인코더**: Transformer 구조로 텍스트(이미지 설명, 레이블)를 벡터 임베딩으로 변환[1][6].  
두 인코더는 **공유 임베딩 공간**에서 이미지-텍스트 쌍의 유사도를 계산합니다.  

### 학습 방법  
1. **대조 학습(Contrastive Learning)**:  
   - 4억 개의 이미지-텍스트 쌍으로 사전 학습[2][3].  
   - **대조 손실 함수** 적용:  
     $$
     \mathcal{L} = -\frac{1}{N} \sum_{i} \ln \frac{e^{\mathbf{v}_i \cdot \mathbf{w}_i / T}}{\sum_j e^{\mathbf{v}_i \cdot \mathbf{w}_j / T}} -\frac{1}{N} \sum_{j} \ln \frac{e^{\mathbf{v}_j \cdot \mathbf{w}_j / T}}{\sum_i e^{\mathbf{v}_i \cdot \mathbf{w}_j / T}}
     $$  
     여기서 $$T$$는 학습 가능한 **온도 매개변수**, $$\mathbf{v}_i$$(이미지 벡터)와 $$\mathbf{w}_i$$(텍스트 벡터)의 유사도를 최대화[1][6].  
   - *Positive pair*(정답 쌍)는 가깝게, *Negative pair*(오답 쌍)는 멀게 임베딩[4].  

2. **제로샷 예측**:  
   - 학습 후 **레이블 없이** 새로운 이미지 분류 가능.  
   - 예시: "강아지", "고양이", "차량" 등 클래스명을 텍스트 인코더에 입력 → 이미지 임베딩과 코사인 유사도 계산 → 가장 높은 유사도 클래스 선택[5][6].  

### 장점  
- **제로샷 일반화**: 사전 학습 없이도 다양한 태스크(이미지 분류, 검색) 적용 가능[2][3].  
- **다중모달 통합**: 이미지와 텍스트의 의미적 연결 학습으로 콘텐츠 검색, 생성형 AI 향상[2][4].  
- **효율성**: 전통적 모델 대비 레이블 데이터 의존도 감소[2].  

### 한계  
- **계산 비용**: 4억 쌍 학습에 약 100만 달러 소요[2].  
- **편향 위험**: 웹 기반 데이터셋의 사회적 편향 반영 가능성[4].  
- **세부 태스크 부적합**: 객체 감지 등 정밀한 비전 태스크에는 미세 조정 필요[4].  

### 활용 사례  
1. **이미지 검색**: 텍스트 쿼리로 관련 이미지 검색.  
2. **콘텐츠 관리**: 부적합 이미지 자동 필터링.  
3. **생성형 AI**: DALL·E, Stable Diffusion 등과 결합해 텍스트-이미지 생성 정확도 향상[3].  

**요약**: CLIP은 대조 학습을 통해 이미지-텍스트 의미 공간을 정렬함으로써 제로샷 인식의 새로운 지평을 열었으며, 멀티모달 AI 발전의 토대가 되었습니다. 다만 높은 계산 비용과 편향 문제는 지속적 개선 과제입니다.

출처
[1] Contrastive Language-Image Pre-training https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training
[2] A Comprehensive Guide to OpenAI's CLIP Model https://www.pingcap.com/article/a-comprehensive-guide-to-openais-clip-model/
[3] CLIP Contrastive Language–Image Pre-Training Model https://blog.roboflow.com/openai-clip/
[4] CLIP: Contrastive Language-Image Pre-Training https://viso.ai/deep-learning/clip-machine-learning/
[5] CLIP (Contrastive Language-Image Pretraining), Predict ... https://github.com/openai/CLIP
[6] Contrastive Language-Image Pre-training (CLIP) https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/clip-and-relatives/clip
[7] [기본 개념] CLIP (Contrastive Language-Image Pre-training) https://xoft.tistory.com/67
[8] CLIP: Connecting text and images https://openai.com/index/clip/
[9] [논문 리뷰] CLIP : Learning Transferable Visual Models From ... https://simonezz.tistory.com/88
[10] openai/clip-vit-large-patch14 - Hugging Face https://huggingface.co/openai/clip-vit-large-patch14

### 파라미터 수를 줄이는 기술
- 프루닝/양자화, 지식 증류(Knowledge Distillation), 파라미터 공유(Parameter Sharing), 저차원 근사(Low-rank Approximation) 등

---

### 입력 길이 제한을 해결하는 기법
- 스트리밍 처리(Streaming Input)
- 페이징 기반 세션 확장(Segmented Context Paging)
- 메모리 보강 모델(Memory-Augmented Models, 외부 메모리 확장)
- 압축 토큰화 및 요약 기법
- 모델 아키텍처 개선: Long Context Models
- Sliding Window Attention Mechanism

> ※ RAG는 검색 기반으로 핵심 정보만 요약된 형태로 전달하여 **우회적으로** 입력 길이 제한을 해결

---

### AGI의 특징
- **범용성**: 특정 작업이 아닌 모든 분야에 유연하게 적용 (예: 멀티모달)
- **자율성**: 사용자 의도를 파악하고 자율적 판단으로 문제 해결/아이디어 제시
- **적응력**: 지속적인 학습, 스스로 학습 등

---

### RAG란? — [참고 링크](http://brunch.co.kr/@vsongyev/28)
- **정의**: 응답을 생성하기 전에 외부의 신뢰할 수 있는 지식 베이스를 참조하는 기술
- **효과**: 사용자 신뢰 강화(허위정보·불신뢰 출처·부정확 응답 방지), 최신 지식 반영, 비용 효율, 개발자 제어 강화
- **동작 구조**  
<img width="898" height="532" alt="image-9" src="https://github.com/user-attachments/assets/26305fd2-bcb4-46d3-a8ff-0ff481497129" />

---

### RAG 발전 현황

| 발전 단계 | 비고 |
| --- | --- |
| **Naive RAG** | **[단점]**<br>ㅇ **제한된 문맥 이해**: 검색된 청크가 질문과 충분히 일치하지 않거나 노이즈가 포함되면 부정확한 답변 가능<br>ㅇ **의미론적 표류(Semantic Drift)**: 검색 정보의 맥락을 완전히 이해하지 못해 의도에서 벗어난 응답 발생<br>ㅇ **검색 정확도 부족**: 단순 유사도 기반 검색만으로는 복잡/미묘한 질의에 최적 문서를 찾기 어려움 |
| **Advanced RAG (고급 RAG)** | **[주요 개선 사항]**<br>**Pre-Retrieval (검색 전)**<br>ㅇ 쿼리 변환/확장/재작성으로 검색 정확도 향상 (질문 재구성, 키워드 추가, 하위질문 분해)<br>ㅇ 최적 청크 전략: 의미론적 경계와 문맥 보존 고려<br>ㅇ 하이브리드 인덱싱: 키워드(BM25), 지식 그래프 등 결합<br>**Retrieval (검색 과정)**<br>ㅇ 개선된 임베딩 모델 사용<br>ㅇ **하이브리드 검색**: 벡터 + 키워드 결합<br>**Post-Retrieval (검색 후)**<br>ㅇ **재랭킹(Reranking)**: Reranker로 최종 문서 선별<br>ㅇ **문맥 재정렬(Context Reordering)**: 활용 효율 높이도록 순서 재배치<br>ㅇ **압축(Compression)**: 컨텍스트 윈도우 한계를 고려해 불필요 정보 제거 |
| **Modular RAG (모듈형 RAG)** | RAG를 **독립 모듈**로 분리해 유연하게 조합/교체하는 아키텍처<br><br>**핵심 개념**<br>• **모듈화(Modularity)**: 쿼리 이해, 검색, 재랭킹, 생성 등을 독립 모듈로 구현<br>• **유연성·재구성 가능성**: 요구사항에 맞춰 모듈 선택·조합<br>• **확장성·유지보수 용이성**: 필요한 모듈만 교체/업데이트<br>• **오케스트레이션(Orchestration)**: 모듈 간 상호작용을 효율적으로 관리 |


## 전이 학습(Transfer Learning)

전이 학습(Transfer Learning)은 AI(인공지능)와 딥러닝 분야에서 널리 이용되는 학습 방법으로, 한 작업에 대해 훈련된 모델(사전 학습된 모델)의 지식을 새로운, 하지만 관련 있는 다른 작업에 적용하는 기술입니다. 쉽게 말해, 이미 특정 태스크에서 좋은 성능을 보이는 모델을 다른, 비슷하거나 유사한 문제에 다시 이용하는 것을 의미합니다[2][5][9].

### 주요 특징

- **사전 학습(pre-training)**: 먼저 많은 데이터와 컴퓨팅 자원을 들여 대규모 데이터셋으로 모델을 학습합니다.
- **전이(transfer)**: 그 지식을 새로운 데이터셋이나 과제에 재사용합니다.
- **미세 조정(fine-tuning)**: 기존 모델의 일부 또는 전체 파라미터를 새로운 태스크에 맞게 조정합니다.

### 예시

예를 들어, 수백만 장의 이미지를 분류하는 데 최적화된 모델(예: ImageNet, VGG, ResNet)에서 학습된 특징 추출 층은 개·고양이 분류처럼 데이터가 적은 특정 문제에 재사용될 수 있습니다. 자연어 처리 분야의 BERT나 GPT도 대규모 말뭉치를 사전 학습한 뒤 각종 텍스트 분석에 맞게 미세 조정되어 사용됩니다[6][8].

### 장점

- **학습 속도 향상**: 이미 학습된 특징을 활용하므로, 새롭게 학습할 데이터 양을 크게 줄일 수 있습니다[4][7].
- **적은 데이터로 높은 성능**: 데이터가 적은 환경에서도 좋은 성능을 낼 수 있습니다.
- **오버피팅 방지**: 사전 학습된 모델을 기반으로 하여 복잡한 모델이 오버피팅되는 위험을 줄여 줍니다[2].

### 적용 분야

- 컴퓨터 비전(이미지 분류, 객체 탐지, 세분화)
- 자연어 처리(텍스트 분류, 개체명 인식, 번역 등)
- 음성 인식, 생체 인증 등 다양한 분야

### 비교: 기존 머신러닝 vs. 전이 학습

| 구분               | 기존 머신러닝                | 전이 학습                      |
|--------------------|-----------------------------|--------------------------------|
| 데이터 요구량       | 매우 많음                    | 적어도 높은 성능 가능           |
| 효율성             | 처음부터 모두 학습           | 이미 학습된 모델을 재사용       |
| 일반화 가능성      | 한 작업에만 최적화           | 여러 작업에 쉽게 적용 가능      |

### 핵심 요약

- **전이 학습**은 기존에 훈련된 모델의 지식을 다른 새로운 작업에서 재사용하고, 빠른 학습과 데이터 부족 상황에서 강점을 갖는 딥러닝 방법입니다.
- 컴퓨터 비전, 자연어 처리 등에서 사전 학습 모델을 파인튜닝하여 다양한 실제 문제 해결에 광범위하게 활용됩니다[2][4][9].

출처
[1] 전이학습(Transfer Learning)이란? https://dacon.io/forum/405988
[2] 딥러닝 기초 - 전이 학습 이해하기 https://velog.io/@tjdtnsu/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EC%B4%88-%EC%A0%84%EC%9D%B4-%ED%95%99%EC%8A%B5-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
[3] Transfer Learning - 전이 학습이란? - 매스웍스 https://kr.mathworks.com/discovery/transfer-learning.html
[4] 전이 학습(Transfer learning)이란? 정의, 사용 방법, AI 구축 - 에펜 https://kr.appen.com/blog/transfer-learning/
[5] L_01. Transfer Learning - Deep Learning Bible https://wikidocs.net/240401
[6] [NLP기초] 트랜스퍼 러닝이란? Transfer Learning - 공부 기록장 https://dream-and-develop.tistory.com/342
[7] [DL] Transfer learning이란? - 내 취미는 머신러닝 - 티스토리 https://binnni.tistory.com/24
[8] 트랜스퍼 러닝(Transfer Learning) https://ratsgo.github.io/nlpbook/docs/introduction/transfer/
[9] 전이 학습이란 무엇인가요? https://www.ibm.com/kr-ko/think/topics/transfer-learning
[10] 전이 학습이란 무엇인가요? https://aws.amazon.com/ko/what-is/transfer-learning/

## Function Calling / Tool Calling

### 핵심 개념

- **Function Calling**(함수 호출)은 대규모 언어 모델(LLM)이 단순히 텍스트 생성에 그치지 않고, 외부 시스템과 연동하여 다양한 기능(API, DB, 서비스 등)을 실행할 수 있도록 하는 확장 방식입니다. 최근 등장한 LLM들은 사용자의 질문을 이해해 필요한 함수(도구)를 직접 선택하여 호출하고, 그 결과를 바탕으로 답변을 생성합니다[4][3].

- **Tool Calling**도 유사한 개념으로, LLM이 여러 도구(검색, 계산, 외부 서비스 등)를 사용할 수 있는 인터페이스를 제공하여, 기존 챗봇의 제한을 극복합니다.

### 등장 배경

- 과거 LLM은 훈련 데이터 내에서만 답변 가능해 최신 정보 접근 또는 실제 액션(예: 일정 추가, 이메일 발송)이 불가능함.
- 최근 모델(예: HyperCLOVA X, OpenAI GPT-OSS, 구글 Gemma 3 등)은 함수/도구 연결을 도입해 실시간 정보 조회, 실제 액션 처리 등 진짜 비서와 같은 능력을 갖춤[4][7][5][6].

### 주요 동작 방식

1. **함수 정의(Function Schema) 등록:** 개발자가 LLM에 사용할 함수의 정보(이름, 설명, 입력 값 등)를 JSON 등으로 정의.
2. **질문 분석 및 함수 선택:** 사용자의 질문을 이해해, 어떤 함수가 필요한지 판단(예: "서울 날씨 알려줘" → 날씨 함수 호출).
3. **실행 요청 생성:** AI가 함수 실행을 제안하면, 직접 실행(혹은 자동화 시스템 실행) 후 결과를 반환.
4. **최종 응답 생성:** 실행 결과와 기존 대화 맥락을 바탕으로 자연스러운 답변 생성.

### 적용 사례 및 기능

- **예시:** 일정 예약, 날씨 조회, 이메일 발송, 데이터베이스 검색, 쇼핑 주문 등 업무 자동화[4][5].
- 특히 멀티턴 대화(여러 질문-응답 반복), 복잡한 인텐트 처리에서 뛰어난 성능을 보임.

### 기술적 특징

| 구분                  | Function Calling           | 기존 LLM                         |
|----------------------|---------------------------|----------------------------------|
| 데이터 접근          | 실시간 외부 데이터 가능    | 학습 데이터 한정                 |
| 액션 수행            | 직접적인 작업 수행         | 대화 응답만 가능                 |
| 확장성               | 다양한 도구 연동 자유로움  | 고정적 정보 제공                 |
| 사용자 경험          | 맞춤형·실제 서비스 처리    | 제한적 자동화·정보 제공          |

### 최신 트렌드와 대표 모델

- **HyperCLOVA X**: 스킬(Function Calling) 기능으로 외부 API 연동, 다양한 액션 수행[4].
- **OpenAI GPT-OSS**: 오픈소스 LLM에 브라우징, 함수 호출, 파이썬 실행 등 내장 지원[7].
- **Gemma 3, Kanana 1.5**: 전문 함수 호출, 코드 생성, 수학 문제 풀이 등 실용 작업 강화[6][5].

### 요약

최근 LLM의 Function Calling 및 Tool Calling 기능은 AI가 단순 답변 생성에서 벗어나 **실제 작업 자동화와 비즈니스 액션 수행이 가능한 에이전트형 AI**로 진화하는 핵심 동력입니다. 서비스, 개발, 연구 등 다양한 분야에서 효과적으로 적용되고 있습니다[4][7][5].

출처
[1] 2025년 LLM 모델 종류 총정리 : 성능 비교, 업무 활용 사례, LLM AGENT https://app.dalpha.so/blog/llm/
[2] 1 - 2025년, LLM은 어디까지 왔는가? - gsroot https://gsroot.tistory.com/114
[3] 2025 LLM 트렌드: from FM to AI Agent - LG AI Research BLOG https://www.lgresearch.ai/blog/view?seq=565
[4] 당신의 AI에게 행동을 맡겨라: 스킬과 Function Calling - 클로바 https://clova.ai/tech-blog/skill-function-calling
[5] 오픈소스 LLM 모델 젬마 3 기반 AI 에이전트 개발해 보기 https://www.cadgraphics.co.kr/newsview.php?pages=lecture&sub=lecture02&catecode=8&num=77089
[6] 국내 LLM 모델들의 현황과 비교 - MSAP.ai https://www.msap.ai/blog-home/blog/korea-llm/
[7] OpenAI GPT-OSS 공개: 오픈소스 대규모 언어모델의 특징과 의미 https://tilnote.io/pages/68923fb3541ceac8a1a30e40
[8] 대규모 언어 모델 목록 - IBM https://www.ibm.com/kr-ko/think/topics/large-language-models-list
[9] [기술리포트] 2025 구글 AI 에이전트 아키텍처 구조 완벽 이해 https://tech.ktcloud.com/entry/2025-07-google-ai-agent-architecture-%EC%8B%9C%EC%8A%A4%ED%85%9C%EA%B5%AC%EC%A1%B0-%EC%9D%B4%ED%95%B4


## MLOps에서 Feature Store

**Feature Store**는 MLOps의 핵심 구성 요소 중 하나로, 머신러닝 모델에 사용되는 *feature(특징)* 데이터를 중앙에서 관리·저장·공유·재사용할 수 있도록 하는 시스템입니다. 모델 학습과 실제 서비스(추론) 환경 모두에서 일관된 Feature 제공을 가능하게 하여, 효율적이고 신뢰성 있는 ML 파이프라인 운영을 돕습니다.

### Feature Store의 주요 역할

- **중앙 집중적 관리**  
  여러 팀이 만든 다양한 Feature를 한 곳에서 관리함으로써, 중복 개발과 재생산 비용을 줄입니다[1][5].
- **Feature의 재사용과 공유**  
  프로젝트나 팀 간에 검증된 Feature를 쉽게 재사용할 수 있고, 같은 데이터를 기반으로 다른 모델이 동일한 Feature를 쓸 수 있어 효율적입니다[1][6].
- **일관된 Feature 제공**  
  학습(offline)과 추론(online) 환경에서 모두 동일한 Feature를 사용할 수 있어, 모델의 일관성과 신뢰성을 높입니다[1][5].
- **Feature 버전 관리 및 모니터링**  
  Feature의 생성 이력, 원본 데이터, 버전 정보 등을 관리하고, 저장된 Feature의 특성 변화를 실시간 모니터링합니다. 데이터 드리프트 감지에도 활용됩니다[2][4][5].
- **접근 제어와 개인정보 보호**  
  중요한 혹은 민감한 Feature에 대해서는 인증·접근제어 및 보안 기능을 적용할 수 있습니다[2].

### 저장소의 유형

Feature Store는 두 가지 저장소 영역으로 구성됩니다[2][5]:
- **Offline 저장소**  
  모델 학습용 대용량 Feature를 저장. S3, BigQuery, Snowflake 등 대용량 배치 환경에 적합합니다.
- **Online 저장소**  
  실시간 추론 시 빠른 Feature 조회를 위한 저장소. Redis, Cassandra, DynamoDB 등 빠른 응답을 지원합니다.
  
이 두 저장소 사이의 데이터 일관성을 유지하는 것이 매우 중요합니다[5][6].

### 도입 이유와 장점

- 반복적인 Feature Engineering을 줄임
- 모델 학습과 서비스의 데이터 차이(정합성 문제) 예방
- 팀/프로젝트 간 표준화된 Feature 제공으로 데이터 거버넌스 강화
- ML 파이프라인의 생산성과 신뢰성 대폭 향상[1][4][5]

### 대표적인 오픈소스 및 상용 Feature Store

- Feast (Gojek & Google Cloud)
- Hopsworks
- Uber Michelangelo
- AirBnB Zipline
- Netflix Metaflow 등[3][4][9]

***

**정리:**  
Feature Store는 머신러닝 모델에서 사용하는 각종 Feature를 효율적으로 관리·저장·공유하는 시스템으로, MLOps 환경에서 재현성, 일관성, 확장성, 협업을 높이고 ML Lifecycle 전반의 생산성을 극대화합니다. Feature Store의 도입은 대규모/복잡한 ML 프로젝트의 성공적 운영에 매우 중요한 역할을 합니다[1][2][5].

출처
[1] MLOps 최고의 실천법 - MLOps Gym: 크롤 | Databricks Blog https://www.databricks.com/kr/blog/mlops-best-practices-mlops-gym-crawl
[2] Feature Store 필요해?! - JUST WRITE - 티스토리 https://developnote-blog.tistory.com/158
[3] Machine Learning의 Feature Store란? - 어쩐지 오늘은 https://zzsza.github.io/mlops/2020/02/02/feature-store/
[4] 6.6. Feature Store - note - 티스토리 https://sy-note-0.tistory.com/29
[5] Feature store 에 대해서 - Slow Thinking - 티스토리 https://gipyeonglee.tistory.com/369
[6] Understanding the Feature Store with Feast - VISION HONG - 티스토리 https://visionhong.tistory.com/47
[7] 카카오 광고 AI 추천 MLOps 아키텍처 - Feature Store 편 / if ... - YouTube https://www.youtube.com/watch?v=r1ELaD1DiU0
[8] [ML 스터디]MLOps Concepts - velog https://velog.io/@cjkangme/MLOps-Concepts
[9] 제로부터 시작하는 MLOps 도구와 활용 - 4. 데이터 관리 (2/2) https://blog.taehun.dev/from-zero-to-hero-mlops-tools-4-2
