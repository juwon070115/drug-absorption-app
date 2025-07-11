import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 사이드바에 부가 정보 표시
st.sidebar.title("부가 정보")

st.sidebar.markdown("""
### 조건 설명 및 흡수 차이 이유

- **분자량 (Molecular Weight)**  
  분자가 클수록 세포막을 통과하기 어렵습니다. 따라서 큰 분자는 흡수가 제한되어 주로 소장이나 대장에서 흡수되며, 작은 분자는 위에서도 일부 흡수가 가능합니다.

- **LogP (지용성)**  
  적절한 지용성은 세포막 통과에 필수입니다. 너무 낮으면 지용성이 부족해 투과가 어렵고, 너무 높으면 수용성이 부족해 흡수에 불리합니다.

- **TPSA (극성 표면적)**  
  극성 표면적이 크면 분자가 수용성이 높아지지만, 세포막 통과는 어려워집니다. 따라서 TPSA가 큰 약물은 주로 소장 같은 투과가 용이한 부위에서 흡수됩니다.

- **pKa (이온화 pH 값)**  
  약물의 이온화 정도는 체내 부위별 pH에 따라 달라집니다. 이온화된 상태는 세포막 투과가 어렵기 때문에, pKa 값과 체내 pH 차이에 따라 흡수되는 위치가 달라집니다.

---

### 흡수 위치별 특징

- **위 (Stomach)**  
  산성 환경이며 점막이 두꺼워 흡수가 제한적입니다. 작은 분자와 특정 이온화 상태의 약물이 일부 흡수됩니다.

- **소장 (Small Intestine)**  
  넓고 얇은 점막으로 흡수에 가장 유리한 장소입니다. 대부분의 약물이 이곳에서 흡수됩니다.

- **대장 (Large Intestine)**  
  주로 수분 흡수가 이루어지며, 일부 지용성 약물이 흡수됩니다.
""")

# 학습 데이터
X_train = [
    [151, 0.5, 49, 4.5],
    [206, 2.1, 20, 5.2],
    [180, -0.9, 78, 3.8],
    [296, 3.0, 40, 7.0],
    [260, 1.5, 58, 6.3],
    [194, -1.2, 95, 3.1],
    [320, 2.8, 32, 6.8],
    [180, 0.0, 44, 4.2],
    [210, 2.3, 30, 5.0],
    [230, 1.8, 50, 4.7],
    [250, 3.1, 25, 6.0],
    [190, -1.0, 80, 3.5],
    [300, 2.5, 35, 7.2],
    [220, 1.3, 45, 5.8],
    [270, 2.9, 38, 6.5],
    [160, -0.5, 55, 4.0],
    [200, 0.7, 60, 5.1],
    [280, 3.4, 40, 6.7],
    [240, 1.6, 50, 5.6],
    [195, -1.1, 70, 3.4],
    [310, 2.7, 33, 6.9],
    [185, 0.3, 48, 4.3],
    [215, 2.0, 29, 5.3],
    [225, 1.9, 52, 4.8],
    [255, 3.2, 27, 6.1],
    [192, -0.8, 78, 3.6],
    [305, 2.6, 36, 7.1],
    [225, 1.4, 44, 5.7],
    [275, 3.0, 39, 6.4],
    [165, -0.6, 53, 4.1],
]

y_train = [
    1,0,2,1,1,2,0,1,0,1,0,2,1,0,1,
    2,1,0,1,2,0,1,0,1,0,2,1,0,1,2
]

# 모델 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

st.title("💊 약물 흡수 위치 예측기 (AI 기반)")
st.write("분자 특성을 입력하면 약물이 주로 흡수되는 신체 부위를 예측합니다.")

mw = st.number_input("📦 분자량 (Molecular Weight)", min_value=50.0, max_value=1000.0, step=1.0)
logp = st.number_input("💧 LogP (지용성)", min_value=-5.0, max_value=10.0, step=0.1)
tpsa = st.number_input("🧪 극성 표면적 (TPSA)", min_value=0.0, max_value=200.0, step=1.0)
pka = st.number_input("🧬 pKa (이온화 상수)", min_value=0.0, max_value=14.0, step=0.1)

def predict_absorption_site(mw, logp, tpsa, pka):
    input_data = np.array([[mw, logp, tpsa, pka]])
    prediction = model.predict(input_data)[0]
    site_map = {0: "위", 1: "소장", 2: "대장"}
    return site_map[prediction]

if st.button("예측하기"):
    site = predict_absorption_site(mw, logp, tpsa, pka)
    st.success(f"✅ 이 약물은 **'{site}'**에서 흡수될 가능성이 높습니다.")
