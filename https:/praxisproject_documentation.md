
## PROJECT DOCUMENTATION

**Learning Patterns & Classroom Strategy Insights**
### 1. Overview

This project is an ML and GenAI-powered educational decision support system designed to help educators understand student learning behaviors, detect academic risk early, and receive actionable teaching insights.

The system combines machine learning for pattern discovery and risk detection with Generative AI (Amazon Bedrock – Nova) for human-like interpretation and feedback.

---

### 2. Motivation for Choosing the Problem

Students exhibit diverse learning behaviors, but educators often lack scalable tools to analyze these patterns effectively. Early disengagement is difficult to detect, leading to delayed interventions.

This project addresses that gap by providing interpretable, ethical, and practical insights that support human decision-making in classrooms.

---

### 3. Dataset Description

The dataset consists of student academic, engagement, and behavioral attributes such as:

* Study time
* Absences
* Failures
* Engagement indicators
* Health and lifestyle factors

No personally identifiable or sensitive attributes are used.

---

### 4. Machine Learning Approach

* Data preprocessing and feature engineering
* Unsupervised clustering to identify learning patterns
* Risk classification based on academic indicators
* Explainable outputs for educators

ML models are used strictly for **analysis and detection**, not decision-making.

---

### 5. GenAI Integration

Amazon Bedrock (Nova) is integrated as an **interpretation layer**.

Role of GenAI:

* Convert ML insights into teacher-friendly feedback
* Explain strengths, concerns, and teaching strategies
* Maintain supportive and ethical tone

GenAI does **not** predict risk or replace ML.



### 6. System Architecture

The system consists of:

* Streamlit-based dashboard (UI)
* AWS EC2 for hosting
* AWS Lambda + API Gateway for GenAI requests
* Amazon Bedrock Agent (Nova)
* Custom domain with HTTPS

(Refer to architecture diagrams in README.)

---

### 7. User Flow

1. Educator selects or inputs student data
2. ML models generate learning cluster and risk level
3. Dashboard visualizes insights
4. GenAI provides human-like teacher feedback
5. Educator takes final decision

---

### 8. Ethics, Bias & Limitations

* No sensitive attributes used
* AI feedback is advisory only
* Human oversight is mandatory
* Model outputs may vary across datasets
* Designed to assist, not automate education

---

### 9. Business & Practical Feasibility

* Scalable across institutions
* Low infrastructure cost
* Can integrate with LMS platforms
* Reduces teacher workload
* Improves early intervention outcomes

---

### 10. Conclusion

The project demonstrates how ML and GenAI can be responsibly combined to enhance educational decision-making while keeping humans in control.


