import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:1rem;
}

.title{
    text-align:center;
    color:#00D4FF;
    font-size:50px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
    margin-bottom:20px;
}

.metric-card{
    background:#1A1D24;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

.prediction-card{
    background:#1A1D24;
    padding:30px;
    border-radius:15px;
    border:2px solid #00D4FF;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & DATASET
# =========================

model = pickle.load(open("ANN_model.pkl", "rb"))

df = pd.read_csv(
    "laptop_price.csv",
    encoding="latin1"
)

# Convert Ram and Weight exactly like training
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">💻 Laptop Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Artificial Neural Network (ANN) Based Prediction</div>',
    unsafe_allow_html=True
)

# =========================
# TOP BANNER
# =========================

st.markdown("""
<div style="
background:linear-gradient(90deg,#00D4FF,#0066FF);
padding:25px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h2 style="color:white;">
Predict Laptop Prices Instantly
</h2>

<p style="color:white;">
Machine Learning Project using Artificial Neural Networks
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# METRICS
# =========================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Dataset Rows",
        len(df)
    )

with c2:
    st.metric(
        "Features",
        df.shape[1]-1
    )

with c3:
    st.metric(
        "Model",
        "ANN"
    )

with c4:
    st.metric(
        "Target",
        "Price"
    )

st.write("")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙ Laptop Specifications")

company = st.sidebar.selectbox(
    "Company",
    sorted(df["Company"].unique())
)

typename = st.sidebar.selectbox(
    "Laptop Type",
    sorted(df["TypeName"].unique())
)

screen = st.sidebar.selectbox(
    "Screen Resolution",
    sorted(df["ScreenResolution"].unique())
)

cpu = st.sidebar.selectbox(
    "CPU",
    sorted(df["Cpu"].unique())
)

ram = st.sidebar.slider(
    "RAM (GB)",
    2,
    64,
    8
)

memory = st.sidebar.selectbox(
    "Storage",
    sorted(df["Memory"].unique())
)

gpu = st.sidebar.selectbox(
    "GPU",
    sorted(df["Gpu"].unique())
)

os = st.sidebar.selectbox(
    "Operating System",
    sorted(df["OpSys"].unique())
)

inches = st.sidebar.slider(
    "Screen Size (Inches)",
    10.0,
    20.0,
    15.6
)

weight = st.sidebar.slider(
    "Weight (Kg)",
    1.0,
    5.0,
    2.0
)

# =========================
# PREDICTION BUTTON
# =========================

predict = st.sidebar.button(
    "🚀 Predict Price",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================

if predict:

    input_df = pd.DataFrame({
        "Company":[company],
        "TypeName":[typename],
        "Inches":[inches],
        "ScreenResolution":[screen],
        "Cpu":[cpu],
        "Ram":[ram],
        "Memory":[memory],
        "Gpu":[gpu],
        "OpSys":[os],
        "Weight":[weight]
    })

    try:

        train_df = df.drop(
            columns=["Price_euros"],
            errors="ignore"
        )

        full_df = pd.concat(
            [train_df, input_df],
            ignore_index=True
        )

        categorical_cols = [
            "Company",
            "TypeName",
            "ScreenResolution",
            "Cpu",
            "Memory",
            "Gpu",
            "OpSys"
        ]

        full_df = pd.get_dummies(
            full_df,
            columns=categorical_cols,
            drop_first=True
        )

        full_df.drop(
            columns=["laptop_ID", "Product"],
            errors="ignore",
            inplace=True
        )

        numerical_cols = [
            "Inches",
            "Ram",
            "Weight"
        ]

        scaler = StandardScaler()

        full_df[numerical_cols] = scaler.fit_transform(
            full_df[numerical_cols]
        )

        final_input = full_df.tail(1)

        prediction = model.predict(final_input)

        price = float(prediction[0][0])

        st.write("")

        st.markdown(
            f"""
            <div class="prediction-card">

            <h3 style="color:white;">
            Predicted Laptop Price
            </h3>

            <h1 style="color:#00D4FF;">
            ₹ {price:,.0f}
            </h1>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction completed!"
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )

