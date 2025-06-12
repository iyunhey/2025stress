import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor  # 대안 모델
import io

st.set_page_config(page_title="스트레스 진단 앱", layout="wide")
st.title("🧠 스트레스 예측 및 권고 시스템")

df = pd.read_csv("stress_sj.csv")

st.sidebar.title("사용자 정보 입력")
name = st.sidebar.text_input("이름을 입력하세요", "홍길동")
age = st.sidebar.slider("나이", 10, 100, 30)
gender = st.sidebar.selectbox("성별", ["남성", "여성"])

st.sidebar.markdown("---")
st.sidebar.subheader("건강 및 스트레스 관련 정보 입력")

X = df.drop(columns=["Physical stress", "Mental stress"])
y_physical = df["Physical stress"]
y_mental = df["Mental stress"]
input_vals = {}
for col in X.columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    input_vals[col] = st.sidebar.slider(f"{col} (범위: {min_val:.2f} ~ {max_val:.2f})", min_val, max_val, float(df[col].median()))
input_df = pd.DataFrame([input_vals])

model_physical = LinearRegression()
model_mental = LinearRegression()
# model_physical = RandomForestRegressor(random_state=42)  # 대안
# model_mental = RandomForestRegressor(random_state=42)
model_physical.fit(X, y_physical)
model_mental.fit(X, y_mental)

def categorize_by_percentile(value, ref_values, reverse=True):
    perc = np.percentile(ref_values, [20, 40, 60, 80])
    if reverse:
        if value <= perc[0]:
            return "매우 좋음"
        elif value <= perc[1]:
            return "좋음"
        elif value <= perc[2]:
            return "보통"
        elif value <= perc[3]:
            return "안좋음"
        else:
            return "매우 안좋음"
    else:
        if value >= perc[3]:
            return "매우 좋음"
        elif value >= perc[2]:
            return "좋음"
        elif value >= perc[1]:
            return "보통"
        elif value >= perc[0]:
            return "안좋음"
        else:
            return "매우 안좋음"

if st.button("스트레스 예측하기"):
    pred_physical = model_physical.predict(input_df)[0]
    pred_mental = model_mental.predict(input_df)[0]
    phys_cat = categorize_by_percentile(pred_physical, y_physical)
    ment_cat = categorize_by_percentile(pred_mental, y_mental)

    st.markdown("---")
    st.subheader(f"👤 {name}님의 스트레스 예측 결과")
    st.metric("신체 스트레스", f"{pred_physical:.2f}", label_visibility="visible")
    st.metric("정신 스트레스", f"{pred_mental:.2f}", label_visibility="visible")

    result_df = pd.DataFrame({
        "스트레스 유형": ["신체 스트레스", "정신 스트레스"],
        "예측 점수": [pred_physical, pred_mental],
        "예측 등급": [phys_cat, ment_cat]
    })
    st.table(result_df)

    st.subheader("📌 권고 사항")
    st.markdown("✅ 신체 스트레스 등급: **" + phys_cat + "**")
    st.markdown("✅ 정신 스트레스 등급: **" + ment_cat + "**")

    with st.expander("⚠️ 진단의 의미와 활용 안내"):
        st.info("본 결과는 참고용 예측이며, 전문가 상담이 필요할 수 있습니다.")

    report = f"""
[{name}님의 스트레스 진단 보고서]

신체 스트레스 점수: {pred_physical:.2f} - {phys_cat}
정신 스트레스 점수: {pred_mental:.2f} - {ment_cat}

<권고 요약>
- 신체 상태가 '{phys_cat}' 수준입니다.
- 정신 상태가 '{ment_cat}' 수준입니다.
- 스트레스 완화를 위해 충분한 수면, 운동, 명상 등을 실천해보세요.
"""
    st.download_button(
        label="📄 보고서 다운로드",
        data=report,
        file_name=f"{name}_stress_report.txt",
        mime="text/plain"
    )

