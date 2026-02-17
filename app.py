import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Learning Patterns Dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-soft: #f3f8ff;
        --bg-surface: #ffffff;
        --text-main: #1f2a37;
        --text-muted: #5b6472;
        --brand: #0f766e;
        --brand-soft: #ccfbf1;
        --border: #dbe4f0;
        --shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }
    .stApp {
        background: radial-gradient(circle at 15% 10%, #e6f4ff 0%, #f7fbff 45%, #f3f8ff 100%);
        color: var(--text-main);
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem 1.3rem 1.6rem 1.3rem;
        box-shadow: var(--shadow);
    }
    h1, h2, h3 {
        color: #123049 !important;
        letter-spacing: 0.2px;
    }
    .stCaption, p {
        color: var(--text-muted);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
        border: 1px solid #d9e7f7;
        border-radius: 14px;
        padding: 0.45rem 0.75rem;
        box-shadow: 0 5px 14px rgba(16, 24, 40, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #3b4b5f !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0f2438 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #0f2438 !important;
    }
    .stMarkdown, .stText, .stSubheader, .stCaption {
        color: #243447 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5a0 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0a5d57 0%, #0c8f8b 100%);
    }
    [data-baseweb="popover"] {
        border-radius: 14px !important;
        border: 1px solid #d6e4f5 !important;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16) !important;
        background: #ffffff !important;
    }
    [data-testid="stDialog"] [role="dialog"] {
        border-radius: 16px !important;
        border: 1px solid #d6e4f5 !important;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.2) !important;
    }
    .ai-feedback-card {
        border: 1px solid #cde6ff;
        border-radius: 12px;
        padding: 0.85rem;
        background: linear-gradient(180deg, #f8fcff 0%, #eef7ff 100%);
        color: #15324d;
        line-height: 1.55;
    }
    .section-title {
        margin-top: 0.45rem;
        margin-bottom: 0.5rem;
        padding: 0.45rem 0.7rem;
        border-left: 5px solid #0f766e;
        background: linear-gradient(90deg, #ecfdf5 0%, #f8fffe 100%);
        border-radius: 8px;
        color: #123049;
        font-weight: 700;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: radial-gradient(circle at 15% 10%, #1b2a3a 0%, #15202d 45%, #101923 100%);
            color: #e8eef7;
        }
        .main .block-container {
            background: rgba(16, 25, 35, 0.86);
            border: 1px solid #334155;
            box-shadow: 0 10px 25px rgba(2, 6, 23, 0.45);
        }
        h1, h2, h3, .stMarkdown, .stText, .stSubheader, .stCaption, p {
            color: #e7eef9 !important;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #1b2b3b 0%, #172434 100%);
            border: 1px solid #32475f;
        }
        div[data-testid="stMetricLabel"] {
            color: #c3d3e8 !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {
            color: #f5f9ff !important;
        }
        .ai-feedback-card {
            border: 1px solid #325d85;
            background: linear-gradient(180deg, #1b2f42 0%, #172838 100%);
            color: #eaf4ff;
        }
        .section-title {
            color: #e7eef9;
            border-left: 5px solid #2dd4bf;
            background: linear-gradient(90deg, #1d3b37 0%, #162830 100%);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Learning Patterns & Risk Dashboard")
st.write("This system provides interpretable learning pattern and risk insights for educators.")

DATA_PATH = "dataset/student_data.csv"
AI_FEEDBACK_API_URL = "https://nrkg7cmta3.execute-api.ap-south-1.amazonaws.com/dev/praxishackthon-agent-api"
AI_FEEDBACK_ERROR_MSG = "AI feedback is temporarily unavailable. Please refer to the standard recommendations."
LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
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


def yes_no_value(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "1", "true", "t"}:
        return "yes"
    return "no"


def yes_no_to_int(value: object) -> int:
    return 1 if yes_no_value(value) == "yes" else 0


def predict_cluster_from_partial(profile: dict, df_model: pd.DataFrame) -> str:
    feature_cols = ["studytime", "failures", "absences", "G3", "freetime", "goout", "part_time_job", "internet"]
    cluster_df = df_model[feature_cols + ["learning_cluster_name"]].copy()
    cluster_df["part_time_job"] = cluster_df["part_time_job"].apply(yes_no_to_int)
    cluster_df["internet"] = cluster_df["internet"].apply(yes_no_to_int)

    centroids = cluster_df.groupby("learning_cluster_name")[feature_cols].mean()
    point = pd.Series(
        {
            "studytime": profile["studytime"],
            "failures": profile["failures"],
            "absences": profile["absences"],
            "G3": profile["G3"],
            "freetime": profile["freetime"],
            "goout": profile["goout"],
            "part_time_job": yes_no_to_int(profile["part_time_job"]),
            "internet": yes_no_to_int(profile["internet"]),
        }
    )
    distances = ((centroids - point) ** 2).sum(axis=1) ** 0.5
    return str(distances.idxmin())


def build_ai_payload(profile: dict) -> dict:
    return {
        "level": "teacher",
        "student_profile": {
            "academic_profile": {
                "study_time": int(profile["studytime"]),
                "failures": int(profile["failures"]),
                "absences": int(profile["absences"]),
                "final_grade": int(profile["G3"]),
            },
            "engagement_profile": {
                "free_time": int(profile["freetime"]),
                "social_outings": int(profile["goout"]),
                "part_time_job": yes_no_value(profile["part_time_job"]),
                "internet_access": yes_no_value(profile["internet"]),
            },
            "ml_insights": {
                "learning_cluster": str(profile["learning_cluster_name"]),
                "risk_level": str(profile["risk"]),
            },
        },
    }


def fetch_ai_teacher_feedback(payload: dict) -> tuple[str | None, str | None, str, str]:
    raw_response_text = ""
    try:
        LOGGER.info("AI feedback request payload: %s", json.dumps(payload, ensure_ascii=True))
        response = requests.post(
            AI_FEEDBACK_API_URL,
            headers={"Content-Type": "application/json"},
            json={"body": payload},
            timeout=20,
        )
        raw_response_text = response.text
        LOGGER.info("AI feedback HTTP status: %s", response.status_code)
        LOGGER.info("AI feedback raw response: %s", response.text)
        response.raise_for_status()

        outer_json = response.json()
        if not isinstance(outer_json, dict):
            LOGGER.error("AI feedback response is not a JSON object.")
            return None, None, "Response is not a JSON object.", raw_response_text

        status_code = outer_json.get("statusCode", 200)
        if str(status_code) != "200":
            LOGGER.error("AI feedback API returned non-200 statusCode in payload: %s", status_code)
            return None, None, f"API statusCode in body: {status_code}", raw_response_text

        # Some gateways return feedback directly; some wrap it in body JSON string.
        body_data = outer_json.get("body", outer_json)
        inner_json: dict
        if isinstance(body_data, str):
            try:
                inner_json = json.loads(body_data)
            except json.JSONDecodeError:
                LOGGER.error("AI feedback body field is not valid JSON string.")
                return None, None, "Body string is not valid JSON.", raw_response_text
        elif isinstance(body_data, dict):
            inner_json = body_data
        else:
            LOGGER.error("AI feedback body has unsupported type: %s", type(body_data).__name__)
            return None, None, f"Unsupported body type: {type(body_data).__name__}", raw_response_text

        feedback = inner_json.get("feedback") or inner_json.get("message")
        session_id = inner_json.get("sessionId") or outer_json.get("sessionId")
        if isinstance(feedback, str) and feedback.strip():
            LOGGER.info("AI feedback parsed successfully. sessionId=%s", session_id)
            return feedback, session_id, "OK", raw_response_text
        LOGGER.error("AI feedback text missing in parsed response. Parsed keys: %s", list(inner_json.keys()))
        return None, session_id, f"Missing feedback field. Keys: {list(inner_json.keys())}", raw_response_text
    except Exception:
        LOGGER.exception("AI teacher feedback request failed.")
        return None, None, "Request failed. Check terminal logs for traceback.", raw_response_text


@st.dialog("AI Teacher Feedback (GenAI)", width="large")
def show_ai_feedback_dialog(feedback_text: str, session_id: str | None) -> None:
    st.caption("AI-assisted advisory explanation. Use alongside ML insights and teacher judgment.")
    st.markdown("<div class='ai-feedback-card'>", unsafe_allow_html=True)
    st.markdown(feedback_text.replace("\n", "  \n"))
    st.markdown("</div>", unsafe_allow_html=True)
    if session_id:
        st.caption(f"Session ID: {session_id}")
    if st.button("Close Feedback"):
        st.session_state["show_ai_feedback_dialog"] = False
        st.rerun()


df = prepare_data(DATA_PATH)

st.sidebar.header("Student Selection")
profile_source = st.sidebar.radio("Profile Source", ["Dataset Student", "New Student Input"], index=0)

student_index = 0
active_profile: dict
active_profile_key = "dataset:0"

if profile_source == "Dataset Student":
    student_index = st.sidebar.slider("Select student index", 0, len(df) - 1, 0)
    st.sidebar.divider()
    selected_student = df.iloc[student_index]
    active_profile = {
        "studytime": int(selected_student["studytime"]),
        "absences": int(selected_student["absences"]),
        "failures": int(selected_student["failures"]),
        "learning_cluster_name": str(selected_student["learning_cluster_name"]),
        "risk": str(selected_student["risk"]),
        "freetime": int(selected_student["freetime"]),
        "goout": int(selected_student["goout"]),
        "part_time_job": yes_no_value(selected_student["part_time_job"]),
        "internet": yes_no_value(selected_student["internet"]),
        "G3": int(selected_student["G3"]),
    }
    active_profile_key = f"dataset:{student_index}"
else:
    st.sidebar.divider()
    st.sidebar.subheader("New Student Features")
    studytime_input = st.sidebar.selectbox("Study Time {1=low effort, 4=high effort}", [1, 2, 3, 4], index=1)
    failures_input = st.sidebar.selectbox("Failures {past failed subjects}", [0, 1, 2, 3], index=0)
    absences_input = st.sidebar.number_input("Absences {missed classes}", min_value=0, max_value=120, value=5, step=1)
    final_grade_input = st.sidebar.number_input("Final Grade {0-20 score}", min_value=0, max_value=20, value=10, step=1)
    freetime_input = st.sidebar.selectbox("Free Time {1=low, 5=high}", [1, 2, 3, 4, 5], index=2)
    goout_input = st.sidebar.selectbox("Social Outings {1=rare, 5=frequent}", [1, 2, 3, 4, 5], index=2)
    part_time_job_input = st.sidebar.selectbox("Part-Time Job {yes/no}", ["no", "yes"], index=0)
    internet_input = st.sidebar.selectbox("Internet Access {yes/no}", ["yes", "no"], index=0)
    st.sidebar.caption("Risk thresholds {High: grade<10, Medium: 10-13, Low: >=14}")
    st.sidebar.caption("Click Predict to update cluster + risk for this new profile.")

    base_profile = {
        "studytime": int(studytime_input),
        "absences": int(absences_input),
        "failures": int(failures_input),
        "freetime": int(freetime_input),
        "goout": int(goout_input),
        "part_time_job": part_time_job_input,
        "internet": internet_input,
        "G3": int(final_grade_input),
    }

    custom_signature = (
        f"{base_profile['studytime']}:{base_profile['failures']}:{base_profile['absences']}:"
        f"{base_profile['G3']}:{base_profile['freetime']}:{base_profile['goout']}:"
        f"{base_profile['part_time_job']}:{base_profile['internet']}"
    )
    if st.sidebar.button("Predict Risk & Cluster"):
        predicted_profile = base_profile.copy()
        predicted_profile["risk"] = risk_label(predicted_profile["G3"])
        predicted_profile["learning_cluster_name"] = predict_cluster_from_partial(predicted_profile, df)
        st.session_state["custom_predicted_profile"] = predicted_profile
        st.session_state["custom_predicted_signature"] = custom_signature
        st.session_state["show_ai_feedback_dialog"] = False

    predicted_signature = st.session_state.get("custom_predicted_signature")
    predicted_profile = st.session_state.get("custom_predicted_profile")
    if predicted_signature == custom_signature and isinstance(predicted_profile, dict):
        active_profile = predicted_profile
        active_profile_key = f"custom:{custom_signature}"
    else:
        active_profile = base_profile.copy()
        active_profile["risk"] = "Not Predicted"
        active_profile["learning_cluster_name"] = "Click Predict Risk & Cluster"
        active_profile_key = f"custom-pending:{custom_signature}"

st.markdown("<div class='section-title'>Individual Student Data</div>", unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    col1, col2, col3 = st.columns(3)
    col1.metric("Study Time", int(active_profile["studytime"]))
    col2.metric("Absences", int(active_profile["absences"]))
    col3.metric("Failures", int(active_profile["failures"]))

    col4, col5 = st.columns(2)
    col4.metric("Learning Cluster", active_profile["learning_cluster_name"])
    col5.metric("Risk Level", active_profile["risk"])

    st.caption("Cluster indicates learning behavior pattern identified via unsupervised learning.")

with right:
    risk = active_profile["risk"]
    color = risk_color(risk) if risk in {"High Risk", "Medium Risk", "Low Risk"} else "#5b6472"
    st.markdown(
        f"<h4 style='color:{color}; margin-bottom:0;'>Risk Indicator: {risk}</h4>",
        unsafe_allow_html=True,
    )
    st.write("Teacher Recommendation")
    if risk in {"High Risk", "Medium Risk", "Low Risk"}:
        st.info(teacher_recommendation(risk))
    else:
        st.info("Run Predict Risk & Cluster for new input to generate recommendation.")

    can_call_genai = active_profile["risk"] in {"High Risk", "Medium Risk", "Low Risk"}
    if st.button("Get AI Teacher Feedback", disabled=not can_call_genai):
        with st.spinner("Generating AI teacher feedback..."):
            payload = build_ai_payload(active_profile)
            feedback_text, session_id, debug_reason, debug_raw = fetch_ai_teacher_feedback(payload)
            st.session_state["ai_feedback_text"] = feedback_text
            st.session_state["ai_feedback_session_id"] = session_id
            st.session_state["ai_feedback_profile_key"] = active_profile_key
            st.session_state["ai_feedback_debug_reason"] = debug_reason
            st.session_state["ai_feedback_debug_raw"] = debug_raw
            st.session_state["show_ai_feedback_dialog"] = bool(feedback_text)
    if not can_call_genai:
        st.caption("For new input: click `Predict Risk & Cluster` first, then click `Get AI Teacher Feedback`.")

    st.markdown("### AI Teacher Feedback (GenAI)")
    st.caption("AI-assisted advisory explanation. Use alongside ML insights and teacher judgment.")

    ai_feedback_text = st.session_state.get("ai_feedback_text")
    ai_feedback_session_id = st.session_state.get("ai_feedback_session_id")
    ai_feedback_profile_key = st.session_state.get("ai_feedback_profile_key")
    ai_feedback_debug_reason = st.session_state.get("ai_feedback_debug_reason")
    ai_feedback_debug_raw = st.session_state.get("ai_feedback_debug_raw")
    show_ai_feedback_dialog_flag = st.session_state.get("show_ai_feedback_dialog", False)

    if ai_feedback_text and ai_feedback_profile_key == active_profile_key:
        st.success("AI feedback generated successfully.")
        if show_ai_feedback_dialog_flag:
            show_ai_feedback_dialog(ai_feedback_text, ai_feedback_session_id)
    elif "ai_feedback_text" in st.session_state and ai_feedback_profile_key == active_profile_key:
        st.warning(AI_FEEDBACK_ERROR_MSG)
        with st.expander("API Debug (last call)"):
            st.code(f"Reason: {ai_feedback_debug_reason or 'N/A'}", language="text")
            if ai_feedback_debug_raw:
                st.code(ai_feedback_debug_raw, language="json")

st.divider()
st.markdown("<div class='section-title'>Class-Level Overview</div>", unsafe_allow_html=True)
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
