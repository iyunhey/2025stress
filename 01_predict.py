import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # scikit-learn 설치 후 정상 작동
import json # LLM 응답을 JSON으로 파싱하기 위해 추가
import time # 로딩 스피너를 위해 추가

# --- 데이터 로드 및 모델 학습 (페이지 공통) ---
# CSV 파일 로드
try:
    # CSV 파일 경로를 main.py와 동일한 레벨로 가정
    df = pd.read_csv("stress_sj.csv", encoding="cp949")
except FileNotFoundError:
    st.error("오류: 'stress_sj.csv' 파일을 찾을 수 없습니다. 파일이 앱과 같은 위치에 있는지 확인해주세요.")
    st.stop()
except UnicodeDecodeError:
    st.error("오류: CSV 파일 인코딩이 CP949가 아닌 것 같습니다. 파일 인코딩을 확인해주세요.")
    st.stop()

# 예측에 사용할 입력 변수 리스트 (컬럼명이 정확히 일치해야 함)
features = [
    "이상심박수", "자율신경활성도 값", "자율신경 균형도",
    "피로도 값", "심장안정도 값", "혈관연령",
    "동맥혈관탄성도", "말초혈관탄성도"
]

# 데이터프레임 컬럼 공백 제거 (일관성을 위해)
df.columns = df.columns.str.strip()

# 예측에 필요한 모든 특성 컬럼이 데이터프레임에 있는지 확인
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    st.error(f"오류: 예측에 필요한 다음 컬럼이 CSV 파일에 없습니다: {', '.join(missing_features)}")
    st.stop()

# 입력 변수와 타깃 변수 분리
X = df[features]
y_physical = df["신체스트레스 값"]
y_mental = df["정신스트레스 값"]

# 선형 회귀 모델 학습
model_physical = LinearRegression()
model_mental = LinearRegression()

# 데이터에 NaN 값이 있을 경우 모델 학습 전 제거
# 결측치 처리 전략은 데이터 특성에 따라 달라질 수 있습니다 (평균, 중앙값 대체 등)
X_clean = X.dropna()
y_physical_clean = y_physical[X_clean.index]
y_mental_clean = y_mental[X_clean.index]

if X_clean.empty:
    st.error("오류: 입력 데이터에 유효한 값이 없어 모델을 학습할 수 없습니다. CSV 파일의 데이터를 확인해주세요.")
    st.stop()

model_physical.fit(X_clean, y_physical_clean)
model_mental.fit(X_clean, y_mental_clean)


# --- 예측값을 5단계 범주로 변환하는 함수 (수정됨: 높은 스트레스 값이 '안좋음'을 의미) ---
def categorize_stress_level(value, all_values):
    # 스트레스 값은 높을수록 안 좋은 것으로 간주합니다.
    # 따라서 분위수 기준을 반대로 적용하여 높은 값일수록 '매우 안좋음'에 가깝게 분류합니다.
    p20, p40, p60, p80 = np.percentile(all_values, [20, 40, 60, 80])

    if value >= p80:
        return "매우 안좋음" # 상위 20%
    elif value >= p60:
        return "안좋음"    # 60% ~ 80%
    elif value >= p40:
        return "보통"    # 40% ~ 60%
    elif value >= p20:
        return "좋음"     # 20% ~ 40%
    else: # value < p20
        return "매우 좋음" 
