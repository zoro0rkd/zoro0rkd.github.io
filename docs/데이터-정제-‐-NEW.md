## 1. 데이터 정제 개요
- 데이터 정제의 정의
 AI 모델이 학습하기 전에, 오류가 있거나 불완전한 데이터를 찾아내고 수정하거나 삭제하는 일련의 과정을 말합니다.
- 정제의 필요성
- AI 모델 성능과 데이터 품질의 관계

## 2. 결측치 처리
- 결측치 탐지 (`isnull()`, `notnull()`)
- 결측치 제거 (`dropna()`)
- 결측치 대체 (`fillna()` - 평균, 중앙값, 최빈값, 고정값 등)
- (추가) **고급 결측치 대체 기법**
  - KNN Imputation
  - MICE(Multiple Imputation by Chained Equations)

## 3. 이상치 처리
- 이상치 정의 및 영향
- 이상치 탐지 방법
  - **IQR**: Q1(25%), Q3(75%) 기준, IQR = Q3 - Q1
  - **z-score**: (측정값 - 평균) / 표준편차
  - **Isolation Forest**: 무작위 분할로 빠르게 격리되는 포인트를 이상치로 판단
  - **One-Class SVM**: 정상 데이터를 둘러싼 경계 형성, 그 밖의 점들을 이상치로 간주
  - **DBSCAN**: 밀도가 낮은 영역의 점을 노이즈로 간주
  - **Local Outlier Factor (LOF)**: 주변 밀도와 비교해 상대적으로 밀도가 낮은 포인트를 이상치로 판단
  - **AutoEncoder 기반 탐지**: 복원 오차가 큰 데이터를 이상치로 판단
- 이상치 처리 방법 (제거, 변환, 대체)
- (추가) **도메인 지식 기반 규칙 설정**

## 4. 데이터 변환
- **데이터 정규화(Normalization)**
  - **Min-Max 정규화**: X_norm = (X - X_min) / (X_max - X_min)
  - 데이터를 0과 1 사이 또는 -1과 1 사이로 스케일링
  - 이상치 영향을 크게 받음, 딥러닝/거리 기반 모델에서 주로 사용
  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```
- **데이터 표준화(Standardization)**
  - **Z-score 표준화**: X_std = (X - mean) / standard_deviation
  - 평균 0, 표준편차 1인 정규분포로 변환
  - 이상치에 덜 민감, 선형 모델/SVM/PCA에서 주로 사용
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```
- 스케일링 필요성
- (추가) **로버스트 스케일링(Robust Scaling)** — 이상치에 강한 스케일링 기법
- **정규화(Regularization)**: 과적합 방지
  - L1 (Lasso): 불필요한 특성의 가중치를 0으로 설정
  - L2 (Ridge): 큰 가중치에 패널티 부여
  - Dropout: 일부 뉴런을 랜덤하게 비활성화
  - Early Stopping: 검증 성능 개선 없을 때 학습 조기 종료

## 5. 특성(Feature) 처리
- 범주형 데이터 인코딩
  - One-Hot Encoding
  - Label Encoding
- 수치형 데이터 변환
  - 로그 변환, 제곱근 변환
- 데이터 타입 변환 (`astype()`)

## 6. 문자열 데이터 정제
- 공백 제거 (`strip()`)
- 대소문자 변환
- 특수문자 제거 및 대체 (`replace()`)
- 정규표현식 활용
- (추가) **토큰화 및 불용어 제거** — NLP 데이터 전처리
- **NLP 토큰**: `<SOS>` (문장 시작), `<EOS>` (문장 끝), `<PAD>` (길이 맞추기), `<UNK>` (사전에 없는 단어)
- **Pandas 문자열 처리**:
   * `dropna()`
     * 결측값(NaN)이 있는 행 또는 열 제거
   ```python
   df.dropna()  # NaN이 있는 행 제거
   df.dropna(axis=1)  # NaN이 있는 열 제거
   df.dropna(subset=['column1'])  # 특정 열 기준으로만 제거
   ```
   * `fillna()`
     * 결측값(NaN)을 지정한 값으로 채움
   ```python
   df.fillna(0)  # NaN을 0으로 대체
   df['col'].fillna(df['col'].mean())  # 평균값으로 대체
   ```
   * `replace()`
     * 특정 값이나 패턴을 다른 값으로 바꿈
   ```python
   df.replace('?', np.nan)  # 특수문자를 NaN으로 변경
   df['col'].replace({1: 'Male', 2: 'Female'})  # 값 매핑
   ```
   * `astype()`
     * 데이터 타입 변경
   ```python
   df['col'] = df['col'].astype(int)  # 정수형으로 변환
   df['date'] = pd.to_datetime(df['date'])  # 문자열을 날짜로 변환
   ```
   * `str.strip(), str.lower(), str.replace()`
     * 문자열 데이터 정제
   ```python
   df['col'] = df['col'].str.strip()  # 앞뒤 공백 제거
   df['col'] = df['col'].str.lower()  # 소문자로 변환
   df['col'] = df['col'].str.replace('-', '')  # 특정 문자 제거
   ```
   * `duplicated() + drop_duplicates()`
     * 용도: 중복 행 탐지 및 제거
   ```python
   df.duplicated()  # 중복 여부를 Boolean으로 반환
   df.drop_duplicates()  # 중복 행 제거
   ```
   * `isnull(), notnull()`
     * 용도: 결측값 탐지
   ```python
   df.isnull().sum()  # 열별 결측값 개수
   df[df['col'].notnull()]  # 결측값이 아닌 행만 선택
   ```
   * `apply() / map() / lambda`
     * 용도: 함수 기반 데이터 변환
   ```python
   df['col'] = df['col'].apply(lambda x: x*100)  # 열 전체에 함수 적용
   df['col2'] = df['col1'].map({'yes': 1, 'no': 0})  # 값 매핑
   ```


## 7. 중복 데이터 처리
- 중복 탐지 (`duplicated()`)
- 중복 제거 (`drop_duplicates()`)

## 8. 데이터 품질 점검 및 시각화
- 결측치 및 이상치 시각화
- 분포 확인
- (추가) **EDA(Exploratory Data Analysis)와 연계한 정제**

## 9. 데이터 정제 자동화 (추가)
- Pandas 파이프라인 구성
- 함수 기반 전처리 (`apply()`, `map()`, `lambda`)
- **파이프라인 자동화 도구**
  - Scikit-learn `Pipeline`
  - Airflow, Prefect