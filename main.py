import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm # 폰트 관리를 위한 모듈 임포트
import os # 폰트 캐시 삭제를 위해 os 모듈 임포트

# statsmodels는 regplot이 내부적으로 처리하므로, 여기서는 직접 사용하지 않아도 됩니다.
# import statsmodels.api as sm 

# --- 한글 폰트 설정 시작 ---
# Matplotlib 폰트 캐시 재생성 강제
# 이전 버전 Matplotlib에서 'get_cachedir' 오류가 발생할 수 있으므로,
# 직접 캐시 파일을 다루는 부분은 제거하고, 폰트 매니저를 다시 로드하도록 시도합니다.
try:
    fm._load_fontmanager(try_read_cache=False) # 캐시를 읽지 않고 다시 로드하도록 강제
    st.info("Matplotlib 폰트 매니저 캐시를 새로고침했습니다.")
except Exception as e:
    st.warning(f"Matplotlib 폰트 매니저 새로고침 중 오류 발생: {e}")

# 시스템에 설치된 나눔고딕 폰트를 찾아 설정합니다.
font_name = None
# 나눔고딕 폰트의 일반적인 파일 경로 및 이름 패턴을 기반으로 탐색
# Streamlit Cloud의 Ubuntu 환경에서 fonts-nanum 설치 시 폰트 파일 경로
nanum_font_dirs = [
    '/usr/share/fonts/truetype/nanum/', # 일반적인 리눅스 나눔 폰트 설치 경로
    '/usr/local/share/fonts/', # 다른 설치 경로 (예: 수동 설치 시)
    os.path.expanduser('~/.fonts/') # 사용자 폰트 경로 (일반적으로 Streamlit Cloud에서는 해당 없음)
]

# NanumGothic 또는 유사한 나눔 폰트를 찾습니다.
found_nanum_font_path = None
for font_dir in nanum_font_dirs:
    if os.path.exists(font_dir):
        for f in os.listdir(font_dir):
            if 'nanumgothic' in f.lower() and f.endswith('.ttf'):
                found_nanum_font_path = os.path.join(font_dir, f)
                break
        if found_nanum_font_path:
            break

if found_nanum_font_path:
    # 폰트 매니저에 폰트 추가 및 이름 가져오기
    # 이미 폰트 매니저가 재로드되었으므로, addfont는 추가적인 보장입니다.
    fm.fontManager.addfont(found_nanum_font_path)
    font_name = fm.FontProperties(fname=found_nanum_font_path).get_name()
    
    # Matplotlib의 font.family 및 font.sans-serif에 나눔 폰트 설정
    plt.rcParams['font.family'] = font_name
    # 나눔고딕을 sans-serif 계열의 최우선 순위로 추가하여 한글이 먼저 선택되도록 합니다.
    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif'] 
    st.info(f"Matplotlib 폰트: '{font_name}'으로 설정되었습니다.")
else:
    # 그래도 한글 폰트를 찾지 못했다면
    plt.rcParams['font.family'] = 'sans-serif'
    st.warning("경고: 시스템에 한글 폰트(나눔고딕 등)를 찾을 수 없습니다. 기본 폰트('sans-serif')로 표시됩니다.")
    st.warning("한글이 깨져 보일 경우, Streamlit 클라우드 배포 환경에서 `apt.txt`에 'fonts-nanum'이 올바르게 추가되었는지 확인하고 앱을 재배포해 주세요.")

# 마이너스 기호가 깨지는 것을 방지 (한글 폰트 적용 후에도 깨질 수 있으므로 유지)
plt.rcParams['axes.unicode_minus'] = False
# --- 한글 폰트 설정 끝 ---

# CSV 파일 불러오기 (CP949 인코딩)
# 'stress_sj.csv' 파일이 앱이 실행되는 디렉토리에 있어야 합니다.
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

# 스트레스 관련 종속 변수 (Y축)
y_vars = ["신체스트레스 값", "정신스트레스 값"]

# 독립 변수 후보 (X축)
# 사용자에게 선택권을 줄 변수들
candidate_x_vars = [
    "평균심박수", "이상심박수", "자율신경활성도 값",
    "자율신경 균형도", "피로도 값", "심장안정도 값",
    "혈관연령", "동맥혈관탄성도", "말초혈관탄성도" # 값 접미사를 제거하고 컬럼명 그대로 사용
]

# 실제 데이터프레임에 존재하는 컬럼만 필터링합니다.
# 사용자가 제공한 컬럼 이름이 실제 파일에 없을 수도 있기 때문입니다.
existing_x_vars = [col for col in candidate_x_vars if col in df.columns]
if not existing_x_vars:
    st.error("오류: 독립 변수 후보 중 CSV 파일에서 유효한 컬럼을 찾을 수 없습니다. 컬럼 이름을 확인해주세요.")
    st.stop()

st.title("스트레스 분석 시각화")
st.markdown("X축 변수를 선택하고, 슬라이더를 사용하여 데이터 범위를 필터링하여 스트레스 값과의 관계를 시각화합니다.")

# X축 변수 선택
x_var = st.selectbox("X축에 사용할 변수 (독립변수) 선택", existing_x_vars)

# 필터링을 위한 데이터프레임 초기화 (원본 데이터의 복사본으로 시작)
# 슬라이더 필터링 결과가 이 filtered_df에 반영됩니다.
filtered_df = df.copy()

# 선택된 X축 변수가 숫자형일 경우에만 슬라이더 필터링을 적용
if pd.api.types.is_numeric_dtype(filtered_df[x_var]):
    # 데이터의 최소/최대 값을 기준으로 슬라이더 범위 설정
    min_val, max_val = filtered_df[x_var].min(), filtered_df[x_var].max()
    
    # 정수형이 아닌 경우 float으로 변환하여 슬라이더 생성 (소수점 고려)
    if pd.api.types.is_integer_dtype(filtered_df[x_var]):
        min_val, max_val = int(min_val), int(max_val)
        filter_range = st.slider(f"{x_var} 범위 필터", min_val, max_val, (min_val, max_val))
    else: # float 또는 기타 숫자형
        filter_range = st.slider(f"{x_var} 범위 필터", float(min_val), float(max_val), (float(min_val), float(max_val)))

    # 슬라이더 범위에 따라 데이터프레임 필터링
    filtered_df = filtered_df[(filtered_df[x_var] >= filter_range[0]) & (filtered_df[x_var] <= filter_range[1])]
else:
    st.info(f"선택된 변수 '{x_var}'는 숫자형이 아니므로 범위 필터링이 적용되지 않습니다.")


# 플롯 생성 함수
# 이 함수는 필터링된 데이터프레임(data_frame)을 인자로 받아 시각화를 수행합니다.
def plot_relationship(data_frame, x, y):
    # seaborn 스타일 설정
    sns.set(style="whitegrid")
    # 그래프를 그릴 Matplotlib Figure 객체 생성
    plt.figure(figsize=(10, 6)) # 그래프 크기 조정

    # 산점도 그리기
    # data_frame에서 x와 y 변수를 사용하여 산점도를 그립니다.
    sns.scatterplot(data=data_frame, x=x, y=y, alpha=0.6)
    
    # 선형 회귀선 추가
    # x축 변수의 고유값이 2개 초과일 때만 회귀선을 그립니다. (회귀 분석에 의미가 있는 경우)
    # sns.regplot은 산점도 위에 선형 회귀선을 자동으로 그려줍니다.
    # scatter=False로 설정하여 산점도를 다시 그리지 않고, 오직 회귀선만 추가합니다.
    if data_frame[x].nunique() > 2:
        sns.regplot(data=data_frame, x=x, y=y, scatter=False, color='red', line_kws={'label':"선형 회귀선"})

    # 그래프 제목 및 축 라벨 설정
    plt.title(f"{x}과(와) {y}의 관계", fontsize=16)
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    
    # 범례 표시
    if data_frame[x].nunique() > 2: # 회귀선이 그려질 때만 범례 표시
        plt.legend()

    # 그래프 레이아웃 자동 조정 (라벨이 겹치지 않도록)
    plt.tight_layout()
    # Streamlit에 Matplotlib 그래프 표시
    st.pyplot(plt.gcf())
    # 현재 Matplotlib Figure를 지워서 다음 그래프가 겹치지 않도록 합니다.
    plt.clf()

# 각 종속 변수에 대해 시각화 실행
# '신체스트레스 값'과 '정신스트레스 값' 각각에 대해 그래프를 그립니다.
for y_var in y_vars:
    # 각 종속 변수와의 관계를 설명하는 부제목
    st.subheader(f"{x_var}과(와) {y_var}의 관계")
    # 필터링된 데이터프레임(filtered_df)과 선택된 x, y 변수를 plot_relationship 함수에 전달
    plot_relationship(filtered_df, x_var, y_var)

st.markdown("---")
st.markdown("스트레스 관련 지표들 간의 관계를 탐색해보세요.")
