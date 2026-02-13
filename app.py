import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Learning Patterns Dashboard", layout="wide")

st.title("Learning Patterns & Risk Dashboard")
st.write("Minimal interactive dashboard for student-level insights.")

DATA_PATH = "dataset/student_data.csv"


@st.cache_data
def prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Base preprocessing for model pipeline.
    df_model = df.drop(columns=["G1", "G2"]).copy()

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
col1, col2, col3 = st.columns(3)
col1.metric("Study Time", student["studytime"])
col2.metric("Absences", student["absences"])
col3.metric("Failures", student["failures"])

st.metric("Learning Cluster", int(student["learning_cluster"]))
