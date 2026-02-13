import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Learning Patterns Dashboard", layout="wide")

st.title("Learning Patterns & Risk Dashboard")
st.write("Minimal interactive dashboard for student-level and classroom insights.")

DATA_PATH = "dataset/student_data.csv"


def risk_label(g3: float) -> str:
    if g3 < 10:
        return "High Risk"
    if g3 < 14:
        return "Medium Risk"
    return "Low Risk"


def teacher_recommendation(risk: str) -> str:
    if risk == "High Risk":
        return "Immediate intervention: 1:1 mentoring, parent outreach, and weekly progress check-ins."
    if risk == "Medium Risk":
        return "Structured support: guided practice, attendance monitoring, and bi-weekly feedback."
    return "Maintain momentum: enrichment tasks and periodic progress monitoring."


def risk_color(risk: str) -> str:
    if risk == "High Risk":
        return "#d62728"
    if risk == "Medium Risk":
        return "#ff8c00"
    return "#2ca02c"


@st.cache_data
def prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Base preprocessing for model pipeline.
    df_model = df.drop(columns=["G1", "G2"]).copy()

    # Risk prediction logic from notebooks.
    df_model["risk"] = df_model["G3"].apply(risk_label)

    # Clustering pipeline mirrors notebook flow by excluding grade columns.
    df_cluster = df.drop(columns=["G1", "G2", "G3"]).copy()

    cat_cols = df_cluster.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df_cluster[col] = le.fit_transform(df_cluster[col])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_model["learning_cluster"] = kmeans.fit_predict(x_scaled)

    return df_model


df = prepare_data(DATA_PATH)

st.sidebar.header("Student Selection")
student_index = st.sidebar.slider("Select student index", 0, len(df) - 1, 0)
student = df.iloc[student_index]

st.subheader("Student Snapshot")
left, right = st.columns(2)

with left:
    col1, col2, col3 = st.columns(3)
    col1.metric("Study Time", int(student["studytime"]))
    col2.metric("Absences", int(student["absences"]))
    col3.metric("Failures", int(student["failures"]))

    col4, col5 = st.columns(2)
    col4.metric("Learning Cluster", int(student["learning_cluster"]))
    col5.metric("Risk Level", student["risk"])

with right:
    risk = student["risk"]
    color = risk_color(risk)
    st.markdown(
        f"<h4 style='color:{color}; margin-bottom:0;'>Risk Indicator: {risk}</h4>",
        unsafe_allow_html=True,
    )
    st.write("Teacher Recommendation")
    st.info(teacher_recommendation(risk))
