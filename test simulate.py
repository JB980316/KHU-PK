import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

st.title("1 Compartment PK Model (IV Bolus) Simulator")

st.write(
    """
    **모델 개요:**  
    이 앱은 약물동태학 1구획 IV bolus 모델의 주요 파라미터(투여량, 분포용적, 소실속도상수)를 실시간으로 조정하여 
    농도-시간 곡선이 어떻게 달라지는지를 시각적으로 보여줍니다.
    """
)

# Streamlit 슬라이더로 입력받기
A0 = st.slider("투여량 (mg)", min_value=10, max_value=2000, value=100, step=10)
Vd = st.slider("분포용적 (L)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
k = st.slider("소실속도상수 k (1/hr)", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
end_time = st.slider("시뮬레이션 시간 (hr)", min_value=1, max_value=72, value=24, step=1)

# 시간 설정
time = np.linspace(0, end_time, 200)

# 미분방정식 함수 정의
def one_compartment_iv_bolus(A, t, k):
    return -k * A

# ODE 풀기
A_t = odeint(one_compartment_iv_bolus, A0, time, args=(k,))
C_t = A_t.flatten() / Vd

# 결과 플롯
fig, ax = plt.subplots()
ax.plot(time, C_t, lw=2)
ax.set_xlabel('Time (hr)')
ax.set_ylabel('Concentration (mg/L)')
ax.set_title('1 Compartment IV Bolus Model')
ax.grid(True)

st.pyplot(fig)

# 파라미터 요약
st.write(f"**선택된 파라미터**")
st.write(f"- 투여량 (A₀): {A0} mg")
st.write(f"- 분포용적 (Vd): {Vd} L")
st.write(f"- 소실속도상수 (k): {k} 1/hr")
st.write(f"- 1구획 반감기: {np.log(2)/k:.2f} hr")

st.caption("Powered by Streamlit & SciPy. Created by JB980316")
