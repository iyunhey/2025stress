import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# CSV 파일 로드
df = pd.read_csv("stress_sj.csv", encoding="cp949")

# 예측에 사용할 입력 변수 리스트 (컬럼명이 정확히 일치해야 함)
features = ["이상심박수", "자율신경활성도 값", "자율신경 균형도", 
            "피로도 값", "심장안정도 값", "혈관연령", 
            "동맥혈관탄성도", "말초혈관탄성도"]

# 입력 변수와 타깃 변수 분리
X = df[features]
y_physical = df["신체스트레스 값"]
y_mental = df["정신스트레스 값"]

# 선형 회귀 모델 학습
model_physical = LinearRegression()
model_mental = LinearRegression()
model_physical.fit(X, y_physical)
model_mental.fit(X, y_mental)


# 사용자 인터페이스: 이름 입력 및 지표 선택
name = st.text_input("이름을 입력하세요")

# 입력값을 저장할 딕셔너리
input_vals = {}
if name:
    for col in features:
        # 해당 컬럼의 최소/최대값 구하기
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        # 슬라이더 생성 (기본값은 최소값)
        input_vals[col] = st.slider(f"{col}", min_val, max_val, min_val)

    # 예측 수행
    X_new = pd.DataFrame([input_vals])
    pred_physical = model_physical.predict(X_new)[0]
    pred_mental = model_mental.predict(X_new)[0]


# 예측값을 5단계 범주로 변환하는 함수
def categorize(value, all_values):
    perc = np.percentile(all_values, [20, 40, 60, 80])
    if value <= perc[0]:
        return "매우 안좋음"
    elif value <= perc[1]:
        return "안좋음"
    elif value <= perc[2]:
        return "보통"
    elif value <= perc[3]:
        return "좋음"
    else:
        return "매우 좋음"

if name:
    phys_cat = categorize(pred_physical, y_physical)
    ment_cat = categorize(pred_mental, y_mental)

if name:
    st.write(f"**{name}님의 예측 결과:**")
    st.write(f"- 신체 스트레스 예상 평가: **{phys_cat}**")
    st.write(f"- 정신 스트레스 예상 평가: **{ment_cat}**")
