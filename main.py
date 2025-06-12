import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import prettify # matplotlib-prettify 라이브러리 임포트

# statsmodels는 regplot이 내부적으로 처리하므로, 여기서는 직접 사용하지 않아도 됩니다.
# import statsmodels.api as sm 

# --- 한글 폰트 설정 시작 ---
# matplotlib-prettify 라이브러리를 사용하여 한글 폰트를 자동으로 설정합니다.
# 이 함수는 시스템에 설치된 나눔고딕 폰트 등을 자동으로 찾아 적용합니다.
# 내부적으로 Matplotlib 폰트 캐시를 관리하고, 폰트 경로를 탐색해 줍니다.
try:
    prettify.set_font()
    st.info("Matplotlib 폰트가 'matplotlib-prettify'를 통해 설정되었습니다.")
except Exception as e:
    # 폰트 설정 실패 시 기본 폰트로 폴백하고 경고 메시지를 표시합니다.
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    st.error(f"오류: 한글 폰트 설정에 실패했습니다. ({e}) 기본 폰트로 표시됩니다.")
    st.warning("스트림릿 클라우드에서 한글이 깨져 보일 경우, GitHub 저장소의 `apt.txt`에 `fonts-nanum`이, `requirements.txt`에 `matplotlib-prettify`가 올바르게 추가되었는지 확인 후 재배포해 주세요.")

# 마이너스 기호가 깨지는 것을 방지 (prettify.set_font()에서 처리되지만, 명시적으로 유지)
plt.rcParams['axes.unicode_minus'] = False
# --- 한글 폰트 설정 끝 ---

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

# 실제 데이터프레임에 존재하는 컬럼만 독립 변수 후보로 필터링합니다.
# 이렇게 하면 CSV 파일에 없는 컬럼을 선택하여 발생하는 오류를 방지할 수 있습니다.
existing_x_vars = [col for col in candidate_x_vars if col in df.columns]
if not existing_x_vars:
    st.error("오류: 독립 변수 후보 중 CSV 파일에서 유효한 컬럼을 찾을 수 없습니다. CSV 파일의 컬럼 이름을 확인해주세요.")
    st.stop()

st.title("스트레스 분석 시각화")
st.markdown("X축 변수를 선택하고, 슬라이더를 사용하여 데이터 범위를 필터링하여 스트레스 값과의 관계를 시각화합니다.")

# X축 변수 선택 UI
x_var = st.selectbox("X축에 사용할 변수 (독립변수) 선택", existing_x_vars)

# 필터링을 위한 데이터프레임 복사 (원본 데이터에 영향을 주지 않기 위함)
filtered_df = df.copy()

# 선택된 X축 변수가 숫자형일 경우에만 슬라이더 필터링을 적용합니다.
if pd.api.types.is_numeric_dtype(filtered_df[x_var]):
    # 선택된 변수의 최소/최대 값을 기준으로 슬라이더 범위를 설정합니다.
    min_val, max_val = filtered_df[x_var].min(), filtered_df[x_var].max()
    
    # 정수형 변수와 실수형 변수를 구분하여 슬라이더를 생성합니다.
    if pd.api.types.is_integer_dtype(filtered_df[x_var]):
        min_val, max_val = int(min_val), int(max_val) # 정수형으로 변환
        filter_range = st.slider(f"{x_var} 범위 필터", min_val, max_val, (min_val, max_val))
    else: # float 또는 기타 숫자형
        filter_range = st.slider(f"{x_var} 범위 필터", float(min_val), float(max_val), (float(min_val), float(max_val)))

    # 슬라이더에서 선택된 범위에 따라 데이터프레임을 필터링합니다.
    filtered_df = filtered_df[(filtered_df[x_var] >= filter_range[0]) & (filtered_df[x_var] <= filter_range[1])]
else:
    # 숫자형이 아닌 변수가 선택되었을 경우 필터링이 적용되지 않음을 안내합니다.
    st.info(f"선택된 변수 '{x_var}'는 숫자형이 아니므로 범위 필터링이 적용되지 않습니다.")


# 데이터 관계 플롯 생성 함수
# 이 함수는 필터링된 데이터프레임과 X, Y축 변수를 인자로 받아 그래프를 그립니다.
def plot_relationship(data_frame, x, y):
    # seaborn 스타일 설정
    sns.set(style="whitegrid")
    # 그래프를 그릴 Matplotlib Figure 객체를 생성합니다.
    plt.figure(figsize=(10, 6)) # 그래프 크기를 조정합니다.

    # 산점도를 그립니다.
    sns.scatterplot(data=data_frame, x=x, y=y, alpha=0.6)
    
    # 선형 회귀선 추가
    # X축 변수의 고유값이 2개 초과일 때만 선형 회귀선이 의미 있으므로 이 조건에서만 그립니다.
    if data_frame[x].nunique() > 2:
        # sns.regplot은 산점도 위에 선형 회귀선을 자동으로 그려줍니다.
        # scatter=False로 설정하여 산점도를 다시 그리지 않고 회귀선만 추가합니다.
        sns.regplot(data=data_frame, x=x, y=y, scatter=False, color='red', line_kws={'label':"선형 회귀선"})

    # 그래프 제목 및 축 라벨 설정
    plt.title(f"{x}과(와) {y}의 관계", fontsize=16)
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    
    # 범례 표시
    # 'No artists with labels found to put in legend' 경고를 방지하기 위해
    # 범례에 표시할 항목(여기서는 회귀선)이 있을 때만 plt.legend()를 호출합니다.
    if data_frame[x].nunique() > 2:
        plt.legend()

    # 그래프 레이아웃 자동 조정 (라벨이 겹치지 않도록)
    plt.tight_layout()
    # Streamlit에 Matplotlib 그래프를 표시합니다.
    st.pyplot(plt.gcf())
    # 현재 Matplotlib Figure를 지워서 다음 그래프가 이전 그래프 위에 겹치지 않도록 합니다.
    plt.clf()

# 각 종속 변수에 대해 시각화 실행
# '신체스트레스 값'과 '정신스트레스 값' 각각에 대해 X축 변수와의 관계를 플롯합니다.
for y_var in y_vars:
    st.subheader(f"{x_var}과(와) {y_var}의 관계")
    # 필터링된 데이터프레임과 선택된 X, Y 변수를 플롯 함수에 전달합니다.
    plot_relationship(filtered_df, x_var, y_var)

st.markdown("---")
st.markdown("스트레스 관련 지표들 간의 관계를 탐색해보세요.")
