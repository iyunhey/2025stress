import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 불러오기 (CP949 인코딩)
# 'stress_sj.csv' 파일은 앱이 실행되는 디렉토리에 있어야 합니다.
try:
    df = pd.read_csv("stress_sj.csv", encoding="cp949")
except FileNotFoundError:
    st.error("오류: 'stress_sj.csv' 파일을 찾을 수 없습니다. 파일이 앱과 같은 위치에 있는지 확인해주세요.")
    st.stop() # 파일이 없으면 앱 실행을 중지합니다.
except UnicodeDecodeError:
    st.error("오류: CSV 파일 인코딩이 CP949가 아닌 것 같습니다. 파일 인코딩을 확인해주세요.")
    st.stop()

# 컬럼 이름 정리: 앞뒤 공백 제거
df.columns = df.columns.str.strip()

# 스트레스 관련 종속 변수 (Y축) 목록
y_vars = ["신체스트레스 값", "정신스트레스 값"]

# 독립 변수 후보 (X축) 목록
# 사용자에게 시각화할 X축 변수를 선택할 수 있도록 제공합니다.
candidate_x_vars = [
    "평균심박수", "이상심박수", "자율신경활성도 값",
    "자율신경 균형도", "피로도 값", "심장안정도 값",
    "혈관연령", "동맥혈관탄성도", "말초혈관탄성도" 
]

# Matplotlib 그래프에 표시될 한글 컬럼 이름의 영문 맵핑
korean_to_english_labels = {
    "평균심박수": "Average Heart Rate",
    "이상심박수": "Abnormal Heart Rate",
    "자율신경활성도 값": "Autonomic Nerve Activity Value",
    "자율신경 균형도": "Autonomic Nerve Balance",
    "피로도 값": "Fatigue Value",
    "심장안정도 값": "Cardiac Stability Value",
    "혈관연령": "Vascular Age",
    "동맥혈관탄성도": "Arterial Elasticity",
    "말초혈관탄성도": "Peripheral Vascular Elasticity",
    "신체스트레스 값": "Physical Stress Value",
    "정신스트레스 값": "Mental Stress Value"
}

# 실제 데이터프레임에 존재하는 컬럼만 독립 변수 후보로 필터링합니다.
# 이렇게 하면 CSV 파일에 없는 컬럼을 선택하여 발생하는 오류를 방지할 수 있습니다.
existing_x_vars = [col for col in candidate_x_vars if col in df.columns]
if not existing_x_vars:
    st.error("오류: 독립 변수 후보 중 CSV 파일에서 유효한 컬럼을 찾을 수 없습니다. CSV 파일의 컬럼 이름을 확인해주세요.")
    st.stop()

st.title("스트레스 분석 시각화")
st.markdown("X축 변수를 선택하고, 슬라이더를 사용하여 데이터 범위를 필터링하여 스트레스 값과의 관계를 시각화합니다.")

# X축 변수 선택 UI - 한글 이름으로 보여줍니다.
x_var_korean = st.selectbox("X축에 사용할 변수 (독립변수) 선택", existing_x_vars)


# 필터링을 위한 데이터프레임 복사 (원본 데이터에 영향을 주지 않기 위함)
filtered_df = df.copy()

# 선택된 X축 변수가 숫자형일 경우에만 슬라이더 필터링을 적용합니다.
if pd.api.types.is_numeric_dtype(filtered_df[x_var_korean]):
    # 선택된 변수의 최소/최대 값을 기준으로 슬라이더 범위를 설정합니다.
    min_val, max_val = filtered_df[x_var_korean].min(), filtered_df[x_var_korean].max()
    
    # 정수형 변수와 실수형 변수를 구분하여 슬라이더를 생성합니다.
    if pd.api.types.is_integer_dtype(filtered_df[x_var_korean]):
        min_val, max_val = int(min_val), int(max_val) # 정수형으로 변환
        filter_range = st.slider(f"{x_var_korean} 범위 필터", min_val, max_val, (min_val, max_val))
    else: # float 또는 기타 숫자형
        filter_range = st.slider(f"{x_var_korean} 범위 필터", float(min_val), float(max_val), (float(min_val), float(max_val)))

    # 슬라이더에서 선택된 범위에 따라 데이터프레임을 필터링합니다.
    filtered_df = filtered_df[(filtered_df[x_var_korean] >= filter_range[0]) & (filtered_df[x_var_korean] <= filter_range[1])]
else:
    # 숫자형이 아닌 변수가 선택되었을 경우 필터링이 적용되지 않음을 안내합니다.
    st.info(f"선택된 변수 '{x_var_korean}'는 숫자형이 아니므로 범위 필터링이 적용되지 않습니다.")


# 데이터 관계 플롯 생성 함수
# 이 함수는 필터링된 데이터프레임과 실제 컬럼 이름(한글), 그리고 Matplotlib에 표시될 영문 이름을 인자로 받습니다.
def plot_relationship(data_frame, x_col_korean, y_col_korean):
    # seaborn 스타일 설정
    sns.set(style="whitegrid")
    # 그래프를 그릴 Matplotlib Figure 객체를 생성합니다.
    plt.figure(figsize=(10, 6)) # 그래프 크기를 조정합니다.

    # x축과 y축의 영문 표시 이름을 맵핑에서 가져옵니다.
    x_display_english = korean_to_english_labels.get(x_col_korean, x_col_korean)
    y_display_english = korean_to_english_labels.get(y_col_korean, y_col_korean)

    # 산점도를 그립니다.
    sns.scatterplot(data=data_frame, x=x_col_korean, y=y_col_korean, alpha=0.6)
    
    # 선형 회귀선 추가
    # X축 변수의 고유값이 2개 초과일 때만 선형 회귀선이 의미 있으므로 이 조건에서만 그립니다.
    if data_frame[x_col_korean].nunique() > 2:
        # sns.regplot은 산점도 위에 선형 회귀선을 자동으로 그려줍니다.
        # scatter=False로 설정하여 산점도를 다시 그리지 않고 회귀선만 추가합니다.
        sns.regplot(data=data_frame, x=x_col_korean, y=y_col_korean, scatter=False, color='red', 
                    line_kws={'label':"Linear Regression Line"})

    # 그래프 제목 및 축 라벨 설정 - 영문으로 표기합니다.
    plt.title(f"Relationship between {x_display_english} and {y_display_english}", fontsize=16)
    plt.xlabel(x_display_english, fontsize=14)
    plt.ylabel(y_display_english, fontsize=14)
    
    # 범례 표시
    if data_frame[x_col_korean].nunique() > 2: # 회귀선이 그려질 때만 범례 표시
        plt.legend()

    # 그래프 레이아웃 자동 조정 (라벨이 겹치지 않도록)
    plt.tight_layout()
    # Streamlit에 Matplotlib 그래프를 표시합니다.
    st.pyplot(plt.gcf())
    # 현재 Matplotlib Figure를 지워서 다음 그래프가 이전 그래프 위에 겹치지 않도록 합니다.
    plt.clf()

# 각 종속 변수에 대해 시각화 실행
# '신체스트레스 값'과 '정신스트레스 값' 각각에 대해 X축 변수와의 관계를 플롯합니다.
for y_var_korean in y_vars:
    # 스트림릿 부제목은 한글로 표기합니다.
    st.subheader(f"{x_var_korean}과(와) {y_var_korean}의 관계")
    # 필터링된 데이터프레임과 선택된 X, Y 변수의 한글 이름을 플롯 함수에 전달합니다.
    plot_relationship(filtered_df, x_var_korean, y_var_korean)

st.markdown("---")
st.markdown("스트레스 관련 지표들 간의 관계를 탐색해보세요.")
