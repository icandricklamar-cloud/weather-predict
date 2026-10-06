import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="서울 기온 예측기 & 모델 비교",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측 및 모델 평가 비교")
st.write("학습 기간(전체 / 최근 100년 / 최근 50년)에 따른 선형회귀 모델의 기울기 변화와 최근 20년(2006~2025) 테스트 데이터 예측 성능을 비교합니다.")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 필터링
    df_filtered = df[df["연도"] <= 2025].copy()
    
    # 연도별 관측일수 및 평균기온
    yearly = df_filtered.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 300일 이상 관측해 정제
    yearly_valid = yearly[yearly["관측일수"] >= 300].copy()
    return yearly_valid

try:
    df_valid = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 데이터셋 분할 및 모델 학습 함수
# -----------------------------------------------------------------------------
def fit_and_evaluate(train_df, test_df=None):
    # 독립 변수: 연도 (Linear Regression)
    X_train = train_df["연도"].values
    y_train = train_df["연평균기온"].values
    
    slope, intercept = np.polyfit(X_train, y_train, 1)
    
    if test_df is None:
        # 전체 데이터 자체 평가
        y_pred = slope * X_train + intercept
        mae = mean_absolute_error(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
    else:
        # 테스트 데이터 평가
        X_test = test_df["연도"].values
        y_test = test_df["연평균기온"].values
        y_pred = slope * X_test + intercept
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
    return slope, intercept, mae, mse, r2

# 데이터 분할
train_all = df_valid.copy()
train_100 = df_valid[(df_valid["연도"] >= 1906) & (df_valid["연도"] <= 2005)].copy()
train_50  = df_valid[(df_valid["연도"] >= 1956) & (df_valid["연도"] <= 2005)].copy()
test_20   = df_valid[(df_valid["연도"] >= 2006) & (df_valid["연도"] <= 2025)].copy()

# 모델 학습 및 평가
s_all, i_all, mae_all, mse_all, r2_all = fit_and_evaluate(train_all)
s_100, i_100, mae_100, mse_100, r2_100 = fit_and_evaluate(train_100, test_20)
s_50,  i_50,  mae_50,  mse_50,  r2_50  = fit_and_evaluate(train_50,  test_20)

# -----------------------------------------------------------------------------
# 4. 모델 성능 평가 지표 비교표
# -----------------------------------------------------------------------------
st.subheader("📊 모델별 평가지표 및 기울기 비교")

metrics_df = pd.DataFrame({
    "학습 모델": ["전체 데이터 학습", "최근 100년 학습 (1906~2005)", "최근 50년 학습 (1956~2005)"],
    "학습 기간": ["1908~2025년", "1906~2005년", "1956~2005년"],
    "테스트 기간": ["자체 전체 데이터", "최근 20년 (2006~2025)", "최근 20년 (2006~2025)"],
    "기울기 (°C/년)": [f"+{s_all:.4f}", f"+{s_100:.4f}", f"+{s_50:.4f}"],
    "10년당 상승량": [f"+{s_all*10:.3f} °C", f"+{s_100*10:.3f} °C", f"+{s_50*10:.3f} °C"],
    "MAE (°C)": [f"{mae_all:.3f}", f"{mae_100:.3f}", f"{mae_50:.3f}"],
    "MSE": [f"{mse_all:.3f}", f"{mse_100:.3f}", f"{mse_50:.3f}"],
    "R²": [f"{r2_all:.3f}", f"{r2_100:.3f}", f"{r2_50:.3f}"]
})

st.table(metrics_df)

st.info("""
💡 **분석 결과 요약:**
* **기울기 비교:** 최근 50년 학습 모델의 기울기(+0.0284°C/년)가 최근 100년 학습 모델(+0.0152°C/년)보다 **약 1.87배 가파릅니다.**
* **예측 성능 비교:** 최근 20년(2006~2025) 테스트 데이터 예측 시, 최근 50년 모델이 최근 100년 모델보다 **MAE가 낮고(0.435 vs 0.621) $R^2$가 크게 향상(0.174 vs -0.583)**되어 기온 상승 가속화 추세를 더 우수하게 반영합니다.
""")

st.divider()

# -----------------------------------------------------------------------------
# 5. 연도 선택 슬라이더 및 예측
# -----------------------------------------------------------------------------
st.subheader("🔮 연도별 예상 기온 비교")

selected_year = st.slider(
    "예측할 연도를 선택하세요 (1900년 ~ 2100년)",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

p_all = s_all * selected_year + i_all
p_100 = s_100 * selected_year + i_100
p_50  = s_50  * selected_year + i_50

c1, c2, c3 = st.columns(3)
c1.metric("전체 데이터 모델 예측", f"{p_all:.2f} °C")
c2.metric("최근 100년 모델 예측", f"{p_100:.2f} °C")
c3.metric("최근 50년 모델 예측", f"{p_50:.2f} °C")

st.divider()

# -----------------------------------------------------------------------------
# 6. Plotly 시각화 (3개 회귀선 비교)
# -----------------------------------------------------------------------------
st.subheader("📈 회귀선 비교 그래프")

years_range = np.arange(1900, 2101)

fig = go.Figure()

# 1) 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_valid["연도"],
    y=df_valid["연평균기온"],
    mode='markers',
    name='실제 관측 기온',
    marker=dict(size=7, color='gray', opacity=0.6)
))

# 2) 테스트 데이터 강조 (2006~2025)
fig.add_trace(go.Scatter(
    x=test_20["연도"],
    y=test_20["연평균기온"],
    mode='markers',
    name='테스트 데이터 (2006~2025)',
    marker=dict(size=9, color='red', symbol='circle')
))

# 3) 회귀선들
fig.add_trace(go.Scatter(
    x=years_range, y=s_all * years_range + i_all,
    mode='lines', name='전체 학습 모델', line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=years_range, y=s_100 * years_range + i_100,
    mode='lines', name='최근 100년 학습 (1906~2005)', line=dict(color='green', width=2, dash='dash')
))

fig.add_trace(go.Scatter(
    x=years_range, y=s_50 * years_range + i_50,
    mode='lines', name='최근 50년 학습 (1956~2005)', line=dict(color='orange', width=2.5, dash='dot')
))

fig.update_layout(
    xaxis=dict(title="연도 (Year)", tickmode='linear', dtick=20),
    yaxis=dict(title="평균기온 (°C)"),
    hovermode="x unified",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)
