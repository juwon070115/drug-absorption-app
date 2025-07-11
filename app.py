import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# ----------------------- 사이드바 정보 -----------------------
st.sidebar.title("부가 정보")

st.sidebar.markdown("""
### 조건 설명 및 흡수 차이 이유

- **분자량 (Molecular Weight)**  
  분자가 클수록 세포막 통과가 어렵습니다. 큰 분자는 주로 소장/대장에서 흡수됩니다.

- **LogP (지용성)**  
  적절한 지용성은 세포막 투과에 필수입니다. 너무 낮거나 높으면 흡수가 어려워집니다.

- **TPSA (극성 표면적)**  
  극성이 크면 수용성은 높으나 세포막 투과는 어려워집니다.

- **pKa (이온화 상수)**  
  약물의 이온화 상태는 pH에 따라 변하며, 이온화된 상태는 투과가 어렵습니다.

- **pH (위장 내 pH)**  
  공복과 식후 상태에 따라 pH가 다르며, 이는 이온화 정도와 흡수 위치에 큰 영향을 미칩니다.

---

### 🍽 공복 vs 식후 상태 비교

- **공복 (Fasted)**  
  - 위 pH ≈ 1.7, 십이지장 pH ≈ 6.1  
  - 약산성 약물은 비이온화↑ → 흡수 유리  
  - 약염기성 약물은 이온화↑ → 흡수 감소

- **식후 (Fed)**  
  - 위 pH ≈ 6.7, 십이지장 pH ≈ 5.4  
  - 담즙↑, 위 배출 지연↑ → 지용성 약물 흡수 유리  
  - 약염기성 약물은 비이온화↑ → 흡수 증가

- **결론**  
  pH 변화에 따른 이온화 및 용해도 변화가 흡수율에 영향을 줍니다.
""")

# ----------------------- 학습 데이터 및 증폭 -----------------------
# 원본 데이터: MW, LogP, TPSA, pKa, Site, pH(공복 상태 기준값 임의 부여)
real_data = pd.DataFrame({
    'MW': [300.2, 450.3, 150.1],
    'LogP': [2.5, 4.0, 0.5],
    'TPSA': [40.2, 80.4, 60.5],
    'pKa': [5.5, 7.2, 3.8],
    'Site': [1, 0, 2],
    'pH': [1.7, 1.7, 1.7],  # 기본 공복 pH 세팅
})

augmented = []
for _, row in real_data.iterrows():
    for _ in range(10):  # 10배 증폭
        # 공복 상태 증폭
        pH_fasted = 1.7 + np.random.normal(0, 0.1)
        augmented.append([
            row.MW * np.random.normal(1, 0.05),
            row.LogP + np.random.normal(0, 0.2),
            row.TPSA + np.random.normal(0, 5),
            row.pKa + np.random.normal(0, 0.2),
            row.Site,
            pH_fasted
        ])
        # 식후 상태 증폭 (pH 변화 반영)
        pH_fed = 6.7 + np.random.normal(0, 0.1)
        augmented.append([
            row.MW * np.random.normal(1, 0.05),
            row.LogP + np.random.normal(0, 0.2),
            row.TPSA + np.random.normal(0, 5),
            row.pKa + np.random.normal(0, 0.2),
            row.Site,
            pH_fed
        ])

df_aug = pd.DataFrame(augmented, columns=['MW', 'LogP', 'TPSA', 'pKa', 'Site', 'pH'])
df_full = pd.concat([real_data, df_aug], ignore_index=True)

X = df_full[['MW', 'LogP', 'TPSA', 'pKa', 'pH']].values
y = df_full['Site'].values

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# ----------------------- 앱 UI -----------------------
st.title("💊 약물 흡수 위치 예측기 (pH 반영, 공복/식후 구분)")

state = st.radio("식사 상태 선택", ("공복 (Fasted)", "식후 (Fed)"))
default_pH = 1.7 if state == "공복 (Fasted)" else 6.7

mw = st.number_input("📦 분자량 (MW)", 50.0, 1000.0, 300.0)
logp = st.number_input("💧 LogP", -5.0, 10.0, 2.5)
tpsa = st.number_input("🧪 TPSA", 0.0, 200.0, 40.0)
pka = st.number_input("🧬 pKa", 0.0, 14.0, 5.5)
ph = st.number_input("🌡 위장 pH (자동 설정됨)", min_value=1.0, max_value=7.0, value=float(default_pH))

def predict_site(mw, logp, tpsa, pka, ph):
    input_data = np.array([[mw, logp, tpsa, pka, ph]])
    pred = model.predict(input_data)[0]
    site_map = {0: "위", 1: "소장", 2: "대장"}
    return site_map[pred]

if st.button("예측하기"):
    site = predict_site(mw, logp, tpsa, pka, ph)
    st.success(f"✅ 이 약물은 **'{site}'**에서 흡수될 가능성이 높습니다. (식사 상태: {state}, 위 pH: {ph:.2f})")
