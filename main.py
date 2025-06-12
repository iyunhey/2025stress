import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm # 폰트 관리를 위한 모듈 임포트
import os # 파일 시스템 경로 확인을 위해 os 모듈 임포트

# statsmodels는 regplot이 내부적으로 처리하므로, 여기서는 직접 사용하지 않아도 됩니다.
# import statsmodels.api as sm 

# --- 한글 폰트 설정 시작 ---
# Matplotlib 폰트 캐시를 새로고침하여 시스템에 새로 설치된 폰트를 인식하도록 합니다.
# apt.txt를 통해 폰트를 설치한 후 Matplotlib이 이를 즉시 인식하지 못할 때 유용합니다.
try:
    fm._load_fontmanager(try_read_cache=False) # 캐시를 읽지 않고 다시 로드하도록 강제
    st.info("Matplotlib 폰트 매니저 캐시를 새로고침했습니다.")
except Exception as e:
    # 캐시 새로고침 중 오류가 발생할 경우 경고 메시지를 표시합니다.
    # 이 오류는 Matplotlib 버전이 오래되었을 때 발생할 수 있습니다.
    st.warning(f"Matplotlib 폰트 매니저 새로고침 중 오류 발생: {e}. 사용 중인 Matplotlib 버전이 낮을 수 있습니다.")

# 시스템에 설치된 한글 폰트(나눔고딕 계열)를 찾아 설정합니다.
font_name = None
# 시스템의 모든 폰트 경로를 가져와 나눔고딕 폰트 파일을 찾습니다.
# 'NanumGothic' 또는 'NanumBarunGothic'과 같은 나눔 폰트 계열의 이름을 포함하는 폰트 파일을 우선적으로 찾습니다.
found_nanum_font_path = None
for font_path_in_system in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    # 파일 이름에 'nanumgothic' 또는 'nanumbarungothic'이 포함된 폰트를 찾습니다.
    # 대소문자를 구분하지 않고 찾기 위해 lower()를 사용합니다.
    if 'nanumgothic' in os.path.basename(font_path_in_system).lower() or \
       'nanumbarungothic' in os.path.basename(font_path_in_system).lower():
        found_nanum_font_path = font_path_in_system
        break

if found_nanum_font_path:
    try:
        # 폰트 매니저에 폰트 파일을 직접 추가합니다.
        # 이 단계는 Matplotlib이 자동으로 폰트를 찾지 못할 때 강제로 등록하는 역할을 합니다.
        fm.fontManager.addfont(found_nanum_font_path)
        
        # 추가된 폰트의 실제 내부 이름(메타데이터)을 가져옵니다.
        prop = fm.FontProperties(fname=found_nanum_font_path)
        font_name_from_file = prop.get_name()

        # Matplotlib의 기본 폰트 패밀리를 이 폰트 이름으로 설정합니다.
        plt.rcParams['font.family'] = font_name_from_file
        
        # sans-serif 폰트 목록의 가장 앞에 추가하여 한글 폰트가 영문 폰트보다 우선적으로 사용되도록 합니다.
        # 이렇게 하면 한글이 있는 부분은 나눔고딕, 없는 부분은 다음 sans-serif 폰트를 사용합니다.
        plt.rcParams['font.sans-serif'] = [font_name_from_file] + plt.rcParams['font.sans-serif'] 
        
        st.success(f"한글 폰트 '{font_name_from_file}'이(가) 성공적으로 설정되었습니다.")
    except Exception as e:
        # 폰트 파일은 찾았지만 설정 과정에서 오류가 발생한 경우
        st.error(f"한글 폰트 설정 중 예기치 않은 오류 발생: {e}")
        plt.rcParams['font.family'] = 'sans-serif' # 오류 시 기본 폰트
else:
    # 나눔고딕 폰트를 찾지 못했을 경우 경고 메시지를 표시합니다.
    # 이는 apt.txt 설정이 잘못되었거나 fonts-nanum 설치에 실패했을 가능성이 큽니다.
    plt.rcParams['font.family'] = 'sans-serif'
    st.warning("경고: 시스템에 한글 폰트(나눔고딕 계열)를 찾을 수 없습니다. 기본 폰트('sans-serif')로 표시됩니다.")
    st.warning("한글이 깨져 보일 경우, Streamlit 클라우드 배포 환경에서 `apt.txt`에 'fonts-nanum'이 올바르게 추가되었는지 확인하고 앱을 재배포해 주세요.")

# 마이너스 기호가 깨지는 것을 방지합니다.
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
