import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# 🧩 사이드바 정보
st.sidebar.title("부가 정보")
st.sidebar.markdown("""
### 조건 설명 및 흡수 차이 이유

- **분자량 (MW)**  
  분자가 클수록 세포막 통과가 어렵습니다. 큰 분자는 주로 소장/대장에서 흡수되며 위 흡수는 제한적입니다.

- **LogP (지용성)**  
  지용성은 세포막 투과에 중요합니다. 너무 낮거나 높으면 흡수에 불리합니다.

- **TPSA (극성 표면적)**  
  극성 표면적이 크면 수용성이 높아지지만, 세포막 투과는 어려워집니다.

- **pKa (이온화 상수)**  
  pKa와 체내 pH 차이에 따라 이온화 상태가 달라지며, 이는 흡수 위치에 영향을 줍니다.

---

### 흡수 위치별 특징

- **위 (Stomach)**: 산성 환경, 두꺼운 점막 → 흡수 제한적  
- **소장 (Small Intestine)**: 넓고 얇은 점막 → 흡수에 가장 유리  
- **대장 (Large Intestine)**: 수분 흡수 중심, 일부 지용성 약물 흡수
""")

# 📊 실제 기반 샘플 데이터 (SwissADME 및 문헌 기반)
real_df = pd.DataFrame({
    'MW':   [300.2, 450.3, 150.1],
    'LogP': [2.5,   4.0,   0.5],
    'TPSA': [40.2,  80.4,  60.5],
    'pKa':  [5.5,   7.2,   3.8],
    'Site': [1,     0,     2]  # 0=위, 1=소장, 2=대장
})

# 증폭 (10배)
aug = []
for _, r in real_df.iterrows():
    for _ in range(10):
        aug.append([
            r.MW * np.random.normal(1, 0.05),
            r.LogP + np.random.normal(0, 0.2),
            r.TPSA + np.random.normal(0, 5),
            r.pKa + np.random.normal(0, 0.2),
            r.Site
        ])
aug_df = pd.DataFrame(aug, columns=['MW','LogP','TPSA','pKa','Site'])
df = pd.concat([real_df, aug_df], ignore_index=True)

# 모델 학습
X = df[['MW','LogP','TPSA','pKa']].values
y = df['Site'].values
model = DecisionTreeClassifier(random_state=42).fit(X, y)

# 🎯 사용자 입력 및 예측 UI
st.title("💊 약물 흡수 위치 예측기 (AI 기반)")
st.write("분자 특성을 입력하고, 흡수 위치(위·소장·대장)를 확인하세요.")

mw = st.number_input("📦 분자량 (MW)", min_value=50.0, max_value=1000.0, step=1.0, value=300.0)
logp = st.number_input("💧 LogP", min_value=-5.0, max_value=10.0, step=0.1, value=2.5)
tpsa = st.number_input("🧪 TPSA", min_value=0.0, max_value=200.0, step=1.0, value=40.0)
pka = st.number_input("🧬 pKa", min_value=0.0, max_value=14.0, step=0.1, value=5.5)

if st.button("예측하기"):
    pred = model.predict([[mw, logp, tpsa, pka]])[0]
    name = {0: "위", 1: "소장", 2: "대장"}[pred]
    st.success(f"✅ 이 약물은 **'{name}'**에서 흡수될 가능성이 높습니다.")
