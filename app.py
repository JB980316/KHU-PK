import streamlit as st
import pandas as pd
from models.ode_models import simulate_two_comp_po
from models.nca import run_nca
from utils.fit import fit_model
from utils.plots import plot_fit
from utils.residuals import plot_residuals
from utils.model_comparison import compare_models, display_comparison
from utils.download import generate_download_button

st.set_page_config(layout="wide")
st.title("💊 약물동태학 모델링 및 분석 플랫폼")

# 1. 모델 선택
model_choice = st.selectbox("모델을 선택하세요", [
    "NCA (비구획 모델)",
    "1 컴파트먼트 IV bolus",
    "1 컴파트먼트 PO",
    "2 컴파트먼트 IV bolus",
    "2 컴파트먼트 PO",
    "IV infusion (PI)"
])

# 2. 계산 방식 선택
method = st.radio("계산 방법을 선택하세요", ["지수함수 기반", "ODE 기반"])

# 3. 추가 입력 받기
R = duration = dose = None
if "PO" in model_choice:
    dose = st.number_input("투여 용량 (mg)", value=100.0)
if model_choice == "IV infusion (PI)":
    R = st.number_input("주입 속도 R (mg/hr)", value=50.0)
    duration = st.number_input("주입 시간 (hr)", value=1.0)

# 4. 데이터 업로드
uploaded_file = st.file_uploader("시간-농도 데이터 업로드 (CSV, time, conc 열 필요)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("업로드된 데이터")
    st.dataframe(data)

    # 5. NCA 분석일 경우 별도 처리
    if model_choice == "NCA (비구획 모델)":
        all_times = list(data['time'].unique())
        terminal_mode = st.radio("터미널 페이즈 선택 방법", ["자동", "수동"])
        selected_times = None
        use_manual = False

        if terminal_mode == "수동":
            selected_times = st.multiselect("터미널 페이즈로 사용할 시간 선택", all_times)
            use_manual = True if selected_times else False

        result = run_nca(data, selected_times=selected_times, use_manual=use_manual)
        st.subheader("🔍 추정된 파라미터")
        st.json(result['params'])
        st.subheader("📈 터미널 페이즈 적합 그래프")
        plot_fit(data, result['pred'])

    else:
        # 6. 모델 피팅 실행
        if st.button("📈 모델 피팅 및 분석 시작"):
            result = fit_model(data, model_choice, method, dose=dose, R=R, duration=duration)

            st.subheader("🔍 추정된 파라미터")
            st.json(result['params'])

            st.subheader("📉 예측 vs 실측 그래프")
            plot_fit(data, result['pred'])

            st.subheader("📊 잔차 분석")
            plot_residuals(data['conc'], result['pred'])

            st.subheader("📌 모델 적합성 평가")
            st.write(f"AIC: {result['aic']:.2f}")
            st.write(f"BIC: {result['bic']:.2f}")

            generate_download_button(result)

        if st.button("📊 모든 모델 비교 (AIC 기반)"):
            results = compare_models(data, method=method, dose=dose, R=R, duration=duration)
            display_comparison(results, st)
