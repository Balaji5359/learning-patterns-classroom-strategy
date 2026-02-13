import pandas as pd
import streamlit as st

st.set_page_config(page_title="Learning Patterns Dashboard", layout="wide")

st.title("Learning Patterns & Risk Dashboard")
st.write("Minimal interactive dashboard for student-level insights.")

DATA_PATH = "dataset/student_data.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_data(DATA_PATH)

st.sidebar.header("Student Selection")
student_index = st.sidebar.slider("Select student index", 0, len(df) - 1, 0)
student = df.iloc[student_index]

st.subheader("Student Snapshot")
col1, col2, col3 = st.columns(3)
col1.metric("Study Time", student["studytime"])
col2.metric("Absences", student["absences"])
col3.metric("Failures", student["failures"])
