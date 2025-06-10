import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# CSV 파일 불러오기 (CP949 인코딩)
df = pd.read_csv("stress_sj.csv", encoding="cp949")

# 컬럼 이름 정리: 앞뒤 공백 제거
df.columns = df.columns.str.strip()

# 스트레스 관련 종속 변수
y_vars = ["신체스트레스 값", "정신스트레스 값"]

# 독립 변수 후보 선택
candidate_x_vars = [
    "평균심박수", "이상심박수", "자율신경활성도 값",
    "자율신경 균형도", "피로도 값", "심장안정도 값",
    "혈관연령", "동맥혈관탄성도 값", "말초혈관탄성도 값"
]

st.title("스트레스 분석 시각화")

# X축 변수 선택
x_var = st.selectbox("X축에 사용할 변수 (독립변수) 선택", candidate_x_vars)

# 슬라이더 필터 적용 여부 확인
if pd.api.types.is_numeric_dtype(df[x_var]):
    min_val, max_val = int(df[x_var].min()), int(df[x_var].max())
    filter_range = st.slider(f"{x_var} 범위 필터", min_val, max_val, (min_val, max_val))
    df = df[(df[x_var] >= filter_range[0]) & (df[x_var] <= filter_range[1])]

# 플롯 생성
def plot_relationship(x, y):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    
    # 회귀선 추가
    if df[x].nunique() > 2:
        sns.regplot(data=df, x=x, y=y, scatter=False, color='red', line_kws={'label':"선형 회귀선"})

    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# 각 종속 변수에 대해 시각화
for y_var in y_vars:
    st.subheader(f"{x_var}과(와) {y_var}의 관계")
    plot_relationship(x_var, y_var)
