import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # 현재 코드에서 직접적으로 사용되지는 않지만, 혹시 모를 확장을 위해 남겨둡니다.

# CSV 파일 불러오기 (CP949 인코딩)
# 실제 파일 경로에 맞게 'stress_sj.csv'를 수정해주세요.
try:
    df = pd.read_csv("stress_sj.csv", encoding="cp949")
except FileNotFoundError:
    st.error("Error: 'stress_sj.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop() # 파일이 없으면 앱 실행을 중지합니다.

# 컬럼 이름 정리: 앞뒤 공백 제거
df.columns = df.columns.str.strip()

# 독립 변수 후보 선택
candidate_x_vars = [
    "평균심박수", "이상심박수", "자율신경활성도 값",
    "자율신경 균형도", "피로도 값", "심장안정도 값",
    "혈관연령", "동맥혈관탄성도", "말초혈관탄성도" # 컬럼명 일치 확인 (예: '동맥혈관탄성도 값' -> '동맥혈관탄성도')
]

# 종속 변수 후보 선택
candidate_y_vars = [
    "신체스트레스 값", "정신스트레스 값", "평균심박수", "이상심박수",
    "자율신경활성도 값", "자율신경활성도 단계", "피로도 값", "피로도 단계",
    "심장안정도 값", "심장안정도 단계", "자율신경 균형도", "신체스트레스 값",
    "신체스트레스 단계", "정신 스트레스 값", "정신스트레스 단계",
    "스트레스대처능력 값", "스트레스대처능력 단계", "종합점수",
    "동맥혈관탄성도", "동맥혈관탄성도 단계", "말초혈관탄성도",
    "말초혈관탄성도 단계", "혈관연령", "혈관단계"
]

# 실제 데이터프레임에 존재하는 컬럼만 필터링
candidate_x_vars = [col for col in candidate_x_vars if col in df.columns]
candidate_y_vars = [col for col in candidate_y_vars if col in df.columns]


st.title("스트레스 분석 시각화")
st.write("X축과 Y축 변수를 선택하여 스트레스 관련 지표들 간의 관계를 시각화합니다.")

# X축 변수 선택
x_var = st.selectbox("X축에 사용할 변수 (독립변수) 선택", candidate_x_vars)

# Y축 변수 선택
y_var = st.selectbox("Y축에 사용할 변수 (종속변수) 선택", candidate_y_vars, index=candidate_y_vars.index("신체스트레스 값") if "신체스트레스 값" in candidate_y_vars else 0)


# 슬라이더 필터 적용 (선택된 X축 변수가 숫자형일 경우에만)
filtered_df = df.copy() # 원본 데이터프레임을 보존하기 위해 복사본 사용
if pd.api.types.is_numeric_dtype(filtered_df[x_var]):
    min_val, max_val = filtered_df[x_var].min(), filtered_df[x_var].max()
    # 슬라이더는 정수형이 더 일반적이므로, 정수형으로 변환 가능한 경우 정수 슬라이더를 사용
    if pd.api.types.is_integer_dtype(filtered_df[x_var]):
        min_val, max_val = int(min_val), int(max_val)
        filter_range = st.slider(f"{x_var} 범위 필터", min_val, max_val, (min_val, max_val))
    else: # 실수형 변수일 경우
        filter_range = st.slider(f"{x_var} 범위 필터", float(min_val), float(max_val), (float(min_val), float(max_val)))
    
    filtered_df = filtered_df[(filtered_df[x_var] >= filter_range[0]) & (filtered_df[x_var] <= filter_range[1])]
else:
    st.info(f"선택된 X축 변수 '{x_var}'는 숫자형이 아니므로, 슬라이더 필터링을 적용할 수 없습니다.")


# 플롯 생성 함수
def plot_relationship(data_frame, x, y):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6)) # 명시적으로 figure와 axes 생성

    sns.scatterplot(data=data_frame, x=x, y=y, alpha=0.6, ax=ax)
    
    # 회귀선 추가 (X축 변수가 최소 2개 이상의 고유 값을 가질 때만)
    if data_frame[x].nunique() > 1 and pd.api.types.is_numeric_dtype(data_frame[x]):
        # 선형 회귀선은 `regplot`의 `scatter=False`를 이용
        sns.regplot(data=data_frame, x=x, y=y, scatter=False, color='red', line_kws={'label':"선형 회귀선"}, ax=ax)
        
        # OLS (Ordinary Least Squares) 회귀 분석을 통한 통계 정보 표시 (선택 사항)
        # 기울기와 R-squared 값을 플롯에 추가하여 관계의 강도를 더 명확히 시각화할 수 있습니다.
        X = sm.add_constant(data_frame[x])
        model = sm.OLS(data_frame[y], X)
        results = model.fit()
        
        # R-squared 값을 플롯 제목이나 주석으로 추가
        ax.set_title(f"{x}와(과) {y}의 관계\n(R-squared: {results.rsquared:.3f})")
        
    else:
        ax.set_title(f"{x}와(과) {y}의 관계")


    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    st.pyplot(fig) # 생성한 figure 객체를 전달

# 선택된 변수에 대해 시각화
st.subheader(f"{x_var}과(와) {y_var}의 관계")
if not filtered_df.empty:
    plot_relationship(filtered_df, x_var, y_var)
else:
    st.warning("선택된 필터링 조건에 해당하는 데이터가 없습니다. 슬라이더 범위를 조절해주세요.")
