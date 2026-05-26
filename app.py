# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------
# 페이지 설정
# -----------------------------------
st.set_page_config(
    page_title="스마트팜 착과율 예측 시스템",
    page_icon="🍅",
    layout="centered"
)

# -----------------------------------
# 제목
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57;'>
    🍅 스마트팜 착과율 예측 시스템
    </h1>
    """,
    unsafe_allow_html=True
)

st.success("📌 환경 데이터를 입력하면 AI가 착과율을 예측합니다.")

# -----------------------------------
# 샘플 데이터 생성
# -----------------------------------
np.random.seed(42)

data_size = 200

df = pd.DataFrame({
    "내부온도": np.random.normal(25, 3, data_size),
    "내부습도": np.random.normal(70, 8, data_size),
    "지온": np.random.normal(22, 2, data_size),
})

# 가상의 착과율 생성
df["착과율"] = (
    0.3 * df["내부온도"]
    + 0.4 * df["내부습도"]
    + 0.2 * df["지온"]
    + np.random.normal(0, 3, data_size)
)

df["착과율"] = np.clip(df["착과율"], 0, 100)

# -----------------------------------
# 모델 학습
# -----------------------------------
X = df[["내부온도", "내부습도", "지온"]]
y = df["착과율"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------------
# 사용자 입력
# -----------------------------------
st.subheader("🌡️ 환경 데이터 입력")

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input(
        "내부온도 (℃)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

with col2:
    humidity = st.number_input(
        "내부습도 (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col3:
    soil_temp = st.number_input(
        "지온 (℃)",
        min_value=0.0,
        max_value=40.0,
        value=22.0
    )

# -----------------------------------
# 예측 버튼
# -----------------------------------
if st.button("🔍 착과율 예측하기"):

    input_data = pd.DataFrame({
        "내부온도": [temp],
        "내부습도": [humidity],
        "지온": [soil_temp],
    })

    prediction = model.predict(input_data)[0]

    st.markdown("## 📈 예측 결과")

    st.markdown(
        f"""
        <div style="
            background-color:#E8F5E9;
            padding:40px;
            border-radius:20px;
            text-align:center;
        ">
            <h1 style='color:#1B5E20; font-size:64px;'>
                {prediction:.1f}%
            </h1>
            <h3 style='color:#2E7D32;'>
                예상 착과율입니다.
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(min(int(prediction), 100))