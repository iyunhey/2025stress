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
    st.error("Error: 'stress_sj.csv' file not found. Please ensure the file is in the same directory as the app.")
    st.stop() # 파일이 없으면 앱 실행을 중지합니다.
except UnicodeDecodeError:
    st.error("Error: CSV file encoding seems not to be CP949. Please check the file encoding.")
    st.stop()

# 컬럼 이름 정리: 앞뒤 공백 제거
df.columns = df.columns.str.strip()

# 스트레스 관련 종속 변수 (Y축) 목록
y_vars = ["신체스트레스 값", "정신스트레스 값"]

# 독립 변수 후보 (X축) 목록 - 영문으로 간단히 표기합니다.
# 사용자에게 시각화할 X축 변수를 선택할 수 있도록 제공합니다.
# 실제 데이터의 컬럼 이름은 한글이지만, 여기서 표시될 선택 박스 옵션은 영문으로 지정합니다.
# 맵핑을 통해 실제 데이터 컬럼을 참조합니다.
x_var_display_names = {
    "Average Heart Rate": "평균심박수",
    "Abnormal Heart Rate": "이상심박수",
    "Autonomic Nerve Activity Value": "자율신경활성도 값",
    "Autonomic Nerve Balance": "자율신경 균형도",
    "Fatigue Value": "피로도 값",
    "Cardiac Stability Value": "심장안정도 값",
    "Vascular Age": "혈관연령",
    "Arterial Elasticity": "동맥혈관탄성도",
    "Peripheral Vascular Elasticity": "말초혈관탄성도"
}

# 실제 데이터프레임에 존재하는 컬럼만 독립 변수 후보로 필터링합니다.
# 맵핑된 한글 컬럼 이름이 실제로 df에 존재하는지 확인합니다.
valid_x_var_keys = [key for key, value in x_var_display_names.items() if value in df.columns]
if not valid_x_var_keys:
    st.error("Error: No valid columns found in the CSV file for X-axis variables. Please check column names in your CSV.")
    st.stop()

st.title("Stress Analysis Visualization")
st.markdown("Select an X-axis variable and filter the data range using the slider to visualize its relationship with stress values.")

# X축 변수 선택 UI - 영문 이름으로 보여줍니다.
selected_x_display_name = st.selectbox("Select X-axis variable (Independent Variable)", valid_x_var_keys)
# 실제 데이터프레임에서 사용할 한글 컬럼 이름으로 맵핑합니다.
x_var = x_var_display_names[selected_x_display_name]


# 필터링을 위한 데이터프레임 복사 (원본 데이터에 영향을 주지 않기 위함)
filtered_df = df.copy()

# 선택된 X축 변수가 숫자형일 경우에만 슬라이더 필터링을 적용합니다.
if pd.api.types.is_numeric_dtype(filtered_df[x_var]):
    # 선택된 변수의 최소/최대 값을 기준으로 슬라이더 범위를 설정합니다.
    min_val, max_val = filtered_df[x_var].min(), filtered_df[x_var].max()
    
    # 정수형 변수와 실수형 변수를 구분하여 슬라이더를 생성합니다.
    if pd.api.types.is_integer_dtype(filtered_df[x_var]):
        min_val, max_val = int(min_val), int(max_val) # 정수형으로 변환
        filter_range = st.slider(f"{selected_x_display_name} Range Filter", min_val, max_val, (min_val, max_val))
    else: # float 또는 기타 숫자형
        filter_range = st.slider(f"{selected_x_display_name} Range Filter", float(min_val), float(max_val), (float(min_val), float(max_val)))

    # 슬라이더에서 선택된 범위에 따라 데이터프레임을 필터링합니다.
    filtered_df = filtered_df[(filtered_df[x_var] >= filter_range[0]) & (filtered_df[x_var] <= filter_range[1])]
else:
    # 숫자형이 아닌 변수가 선택되었을 경우 필터링이 적용되지 않음을 안내합니다.
    st.info(f"The selected variable '{selected_x_display_name}' is not numeric, so range filtering is not applied.")


# 데이터 관계 플롯 생성 함수
# 이 함수는 필터링된 데이터프레임과 X, Y축 변수를 인자로 받아 그래프를 그립니다.
def plot_relationship(data_frame, x_col_name, y_col_name, x_display_name, y_display_name):
    # seaborn 스타일 설정
    sns.set(style="whitegrid")
    # 그래프를 그릴 Matplotlib Figure 객체를 생성합니다.
    plt.figure(figsize=(10, 6)) # 그래프 크기를 조정합니다.

    # 산점도를 그립니다.
    sns.scatterplot(data=data_frame, x=x_col_name, y=y_col_name, alpha=0.6)
    
    # 선형 회귀선 추가
    # X축 변수의 고유값이 2개 초과일 때만 선형 회귀선이 의미 있으므로 이 조건에서만 그립니다.
    if data_frame[x_col_name].nunique() > 2:
        # sns.regplot은 산점도 위에 선형 회귀선을 자동으로 그려줍니다.
        # scatter=False로 설정하여 산점도를 다시 그리지 않고 회귀선만 추가합니다.
        sns.regplot(data=data_frame, x=x_col_name, y=y_col_name, scatter=False, color='red', line_kws={'label':"Linear Regression Line"})

    # 그래프 제목 및 축 라벨 설정 - 영문으로 표기합니다.
    plt.title(f"Relationship between {x_display_name} and {y_display_name}", fontsize=16)
    plt.xlabel(x_display_name, fontsize=14)
    plt.ylabel(y_display_name, fontsize=14)
    
    # 범례 표시
    # 'No artists with labels found to put in legend' 경고를 방지하기 위해
    # 범례에 표시할 항목(여기서는 회귀선)이 있을 때만 plt.legend()를 호출합니다.
    if data_frame[x_col_name].nunique() > 2:
        plt.legend()

    # 그래프 레이아웃 자동 조정 (라벨이 겹치지 않도록)
    plt.tight_layout()
    # Streamlit에 Matplotlib 그래프를 표시합니다.
    st.pyplot(plt.gcf())
    # 현재 Matplotlib Figure를 지워서 다음 그래프가 이전 그래프 위에 겹치지 않도록 합니다.
    plt.clf()

# 각 종속 변수에 대해 시각화 실행
# '신체스트레스 값'과 '정신스트레스 값' 각각에 대해 X축 변수와의 관계를 플롯합니다.
# Y축 변수도 영문으로 표기하기 위한 맵핑을 추가합니다.
y_var_display_names = {
    "신체스트레스 값": "Physical Stress Value",
    "정신스트레스 값": "Mental Stress Value"
}

for y_var_korean in y_vars:
    y_var_english = y_var_display_names[y_var_korean]
    st.subheader(f"Relationship between {selected_x_display_name} and {y_var_english}")
    # 필터링된 데이터프레임과 실제 데이터 컬럼 이름, 그리고 표시될 영문 이름을 전달합니다.
    plot_relationship(filtered_df, x_var, y_var_korean, selected_x_display_name, y_var_english)

st.markdown("---")
st.markdown("Explore the relationships between stress-related indicators.")
