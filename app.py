import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Learning Patterns Dashboard", layout="wide")

st.title("Learning Patterns & Risk Dashboard")
st.write("This system provides interpretable learning pattern and risk insights for educators.")

DATA_PATH = "dataset/student_data.csv"
CLUSTER_NAMES = {
    0: "Consistent Learner",
    1: "Moderate Learner",
    2: "At-Risk Learner",
}


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
    df_model["learning_cluster_name"] = df_model["learning_cluster"].map(CLUSTER_NAMES)

    return df_model


def plot_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    risk_order = ["High Risk", "Medium Risk", "Low Risk"]
    sns.countplot(data=df, x="risk", order=risk_order, palette=["#d62728", "#ff8c00", "#2ca02c"], ax=ax)
    ax.set_title("Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Student Count")
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_cluster_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="learning_cluster_name", palette="Blues", ax=ax)
    ax.set_title("Cluster Distribution")
    ax.set_xlabel("Learning Cluster")
    ax.set_ylabel("Student Count")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def risk_pct(df: pd.DataFrame, label: str) -> float:
    return (df["risk"].eq(label).sum() / len(df)) * 100


df = prepare_data(DATA_PATH)

st.sidebar.header("Student Selection")
student_index = st.sidebar.slider("Select student index", 0, len(df) - 1, 0)
student = df.iloc[student_index]

st.subheader("Individual Student Data")
left, right = st.columns(2)

with left:
    col1, col2, col3 = st.columns(3)
    col1.metric("Study Time", int(student["studytime"]))
    col2.metric("Absences", int(student["absences"]))
    col3.metric("Failures", int(student["failures"]))

    col4, col5 = st.columns(2)
    col4.metric("Learning Cluster", student["learning_cluster_name"])
    col5.metric("Risk Level", student["risk"])

    st.caption("Cluster indicates learning behavior pattern identified via unsupervised learning.")

with right:
    risk = student["risk"]
    color = risk_color(risk)
    st.markdown(
        f"<h4 style='color:{color}; margin-bottom:0;'>Risk Indicator: {risk}</h4>",
        unsafe_allow_html=True,
    )
    st.write("Teacher Recommendation")
    st.info(teacher_recommendation(risk))

st.subheader("Class-Level Overview")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Total Students", len(df))
metric_col2.metric("% High Risk", f"{risk_pct(df, 'High Risk'):.1f}%")
metric_col3.metric("% Medium Risk", f"{risk_pct(df, 'Medium Risk'):.1f}%")
metric_col4.metric("% Low Risk", f"{risk_pct(df, 'Low Risk'):.1f}%")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.pyplot(plot_risk_distribution(df))

with chart_right:
    st.pyplot(plot_cluster_distribution(df))
