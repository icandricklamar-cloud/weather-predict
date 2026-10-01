import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 타이틀
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="서울 연평균 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울시 기온 데이터를 활용하여 연도별 기온 변화 추세를 분석하고 미래/과거 연도의 기온을 예측합니다.")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 날짜 데이터 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df_filtered = df[df["연도"] <= 2025].copy()
    
    # 연도별 관측일수 및 평균기온 계산
    yearly = df_filtered.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일수가 300일 이상인 해만 정제
    yearly_valid = yearly[yearly["관측일수"] >= 300].copy()
    return yearly_valid

try:
    df_valid = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 회귀 모델 구축 (독립 변수: 1908년부터 지난 연수)
# -----------------------------------------------------------------------------
# 독립 변수 X = 연도 - 1908
df_valid["X"] = df_valid["연도"] - 1908
X = df_valid["X"]
y = df_valid["연평균기온"]

# 1차 선형 회귀 계수 (기울기, 절편)
slope, intercept = np.polyfit(X, y, 1)

# 연도와 평균기온 간 피어슨 상관계수
corr = np.corrcoef(df_valid["연도"], y)[0, 1]

# 요약 정보 추출
num_years = len(df_valid)
start_year = int(df_valid["연도"].min())
end_year = int(df_valid["연도"].max())

# -----------------------------------------------------------------------------
# 4. 분석 결과 요약 화면 출력
# -----------------------------------------------------------------------------
st.subheader("📊 분석 데이터 요약")
col1, col2, col3, col4 = st.columns(4)

col1.metric("회귀선 활용 해의 개수", f"{num_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{corr:.4f}")

st.divider()

# -----------------------------------------------------------------------------
# 5. 연도 선택 슬라이더 및 예측 기온 표시
# -----------------------------------------------------------------------------
st.subheader("🔮 예측 연도 선택 및 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요 (1900년 ~ 2100년)",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 회귀 방정식에 따른 예측 기온 계산: y = slope * (Year - 1908) + intercept
pred_x = selected_year - 1908
predicted_temp = slope * pred_x + intercept

st.markdown(f"#### **{selected_year}년** 서울 예상 연평균 기온")
st.markdown(
    f"<div style='font-size: 3.5rem; font-weight: bold; color: #FF4B4B; margin-bottom: 20px;'>"
    f"{predicted_temp:.2f} °C"
    f"</div>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------------------------------
# 6. Plotly 시각화 (산점도 및 회귀선)
# -----------------------------------------------------------------------------
st.subheader("📈 연평균 기온 추이 및 회귀 직선")

# 회귀 직선 라인용 x (연도 기준 1900 ~ 2100년)
years_range = np.arange(1900, 2101)
x_range = years_range - 1908
y_range = slope * x_range + intercept

fig = go.Figure()

# 1) 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_valid["연도"],
    y=df_valid["연평균기온"],
    mode='markers',
    name='관측 연평균기온',
    marker=dict(size=8, color='#1f77b4', opacity=0.85)
))

# 2) 회귀 직선
fig.add_trace(go.Scatter(
    x=years_range,
    y=y_range,
    mode='lines',
    name='회귀 직선',
    line=dict(color='#ff7f0e', width=2.5, dash='dash')
))

# 3) 슬라이더 선택 연도 강조 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode='markers+text',
    name=f'선택 연도 ({selected_year}년)',
    text=[f"{selected_year}년 ({predicted_temp:.2f}°C)"],
    textposition="top center",
    marker=dict(size=14, color='#e377c2', symbol='star')
))

fig.update_layout(
    xaxis=dict(title="연도 (Year)", tickmode='linear', dtick=20),
    yaxis=dict(title="평균기온 (°C)"),
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)
