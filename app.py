
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import json
import cv2
import tempfile
import numpy as np
import pandas as pd
import os
import io
import time

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "deeptrace_efficientnet_b0.pth"
REPORT_PATH = BASE_DIR / "reports" / "evaluation_report.json"
HISTORY_PATH = BASE_DIR / "models" / "training_history.json"
ERROR_IMAGE = BASE_DIR / "reports" / "misclassified_images.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="DeepTrace | AI Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. MODERN UI DESIGN
# ============================================================

st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 10% 0%, #19264b 0%, transparent 45%),
        linear-gradient(135deg, #080d1b, #10182c 60%, #0b1222);
    color: #edf2ff;
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 27, 0.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111b32, #0b1222);
    border-right: 1px solid #293657;
}

[data-testid="stSidebar"] * {
    color: #e2eaff;
}

/* Main content */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Headings */
h1 {
    color: #f3f5ff !important;
    font-weight: 800 !important;
    letter-spacing: -1.3px;
}

h2, h3 {
    color: #e8edff !important;
    font-weight: 700 !important;
}

p, label {
    color: #aebbd4;
}

/* Sidebar brand */
.brand {
    padding: 18px 12px;
    margin-bottom: 22px;
}

.brand-title {
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #8f9cff, #44d7e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 11px;
    color: #8797b8;
    letter-spacing: 2px;
    margin-top: 5px;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 35px;
    border-radius: 22px;
    border: 1px solid rgba(128, 145, 255, 0.3);
    background:
        radial-gradient(circle at 85% 20%, rgba(99,102,241,.35), transparent 35%),
        linear-gradient(120deg, #1a2850, #202b52 55%, #142c47);
    box-shadow: 0 15px 45px rgba(0,0,0,.22);
    margin-bottom: 28px;
}

.hero-tag {
    display: inline-block;
    color: #b8c4ff;
    background: rgba(129,140,248,.16);
    border: 1px solid rgba(165,180,252,.25);
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 45px);
    margin-top: 17px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 15px;
    max-width: 650px;
    line-height: 1.8;
    color: #c2cee7;
}

.hero-icon {
    font-size: 100px;
    text-align: center;
    filter: drop-shadow(0 0 20px rgba(124,145,255,.6));
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1a2540, #141d32);
    border: 1px solid #2b3a5b;
    border-radius: 17px;
    padding: 21px;
    min-height: 130px;
    transition: .25s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,.13);
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #7787ff;
    box-shadow: 0 12px 30px rgba(75,91,210,.18);
}

.metric-icon {
    font-size: 23px;
    margin-bottom: 12px;
}

.metric-label {
    color: #9aaaca;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #f3f5ff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

/* General panels */
.panel {
    background: rgba(23, 34, 57, .88);
    border: 1px solid #2c3a58;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 28px rgba(0,0,0,.12);
}

.panel-title {
    color: #e8edff;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 8px;
}

/* Status */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 13px;
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 30px;
    color: #6ee7b7;
    font-size: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 10px #34d399;
}

/* Buttons */
.stButton > button {
    border: 1px solid #6577ed !important;
    border-radius: 11px !important;
    color: white !important;
    background: linear-gradient(100deg, #5968e9, #397fd5) !important;
    font-weight: 700 !important;
    min-height: 45px;
    transition: .25s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(83,104,235,.3);
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(44,59,94,.35);
    border: 1px dashed #6578bd;
    border-radius: 14px;
    padding: 15px;
}

/* Images */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #394a70;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #344363;
    border-radius: 12px;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6675f5, #38c7e6);
}

/* Info messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #2c3a58 !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 22px;
    color: #7283a4;
    font-size: 12px;
    border-top: 1px solid #293653;
    margin-top: 35px;
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(12px);}
    to {opacity: 1; transform: translateY(0);}
}

.fade-in {
    animation: fadeIn .6s ease;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading EfficientNet-B0...")
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    # Support the actual training checkpoint format.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    classes = checkpoint.get("classes", ["fake", "real"])
    image_size = int(checkpoint.get("image_size", 224))

    if not isinstance(classes, (list, tuple)) or len(classes) != 2:
        raise ValueError(f"Invalid class configuration: {classes}")

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    # Handle optional DataParallel prefixes.
    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Keep this consistent with the training preprocessing.
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, list(classes), preprocess, image_size


# ============================================================
# 4. PREDICTION ENGINE
# ============================================================

def predict_image(image):

    model, classes, preprocess, _ = load_model()

    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    index = int(torch.argmax(probabilities).item())

    probability_data = {
        classes[i]: round(
            float(probabilities[i].item()) * 100, 2
        )
        for i in range(len(classes))
    }

    return {
        "prediction": classes[index],
        "confidence": probability_data[classes[index]],
        "probabilities": probability_data
    }


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def load_json(path):

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def metric_card(icon, label, value):

    st.markdown(
        f"""
        <div class="metric-card fade-in">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):

    st.markdown(f"### {title}")

    if subtitle:
        st.caption(subtitle)


def show_prediction(result):

    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction.lower() == "fake":

        st.error(
            "⚠️ MODEL RESULT: POTENTIALLY FAKE"
        )

    else:

        st.success(
            "✅ MODEL RESULT: PREDICTED REAL"
        )

    st.metric(
        "Prediction confidence",
        f"{confidence:.2f}%"
    )

    st.progress(
        int(confidence)
    )

    st.markdown("#### Class probability scores")

    probability_df = pd.DataFrame(
        {
            "Class": list(result["probabilities"].keys()),
            "Score (%)": list(result["probabilities"].values())
        }
    )

    st.bar_chart(
        probability_df.set_index("Class"),
        horizontal=True
    )

    st.caption(
        "Softmax scores indicate the model's relative output "
        "for each class. They are not calibrated probabilities "
        "of authenticity."
    )


# ============================================================
# 6. LOAD PROJECT DATA
# ============================================================

report = load_json(REPORT_PATH)
history = load_json(HISTORY_PATH)

# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">🔬 DeepTrace</div>
            <div class="brand-subtitle">AI INTELLIGENCE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("##### MAIN NAVIGATION")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Model Evaluation",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("##### SYSTEM STATUS")

    st.markdown(
        """
        <div class="status-online">
            <span class="status-dot"></span>
            Application Active
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"**Device:** {DEVICE}")

    st.write("**Architecture:** EfficientNet-B0")

    st.write("**Classes:** Fake / Real")

    st.divider()

    st.caption("DeepTrace | Version 1.0")


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero fade-in">
            <div class="hero-tag">NEXT-GENERATION MEDIA ANALYSIS</div>
            <h1>DeepTrace Intelligence</h1>
            <p>
                Detect manipulated media with deep learning.
                Analyze images, inspect video frames and
                explore model performance through one platform.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if report:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "🎯", "TEST ACCURACY",
                f"{report.get('accuracy', 0) * 100:.1f}%"
            )

        with c2:
            metric_card(
                "🖼️", "TEST IMAGES",
                report.get("test_images", 0)
            )

        with c3:
            metric_card(
                "📈", "MACRO PRECISION",
                f"{report.get('precision_macro', 0):.3f}"
            )

        with c4:
            metric_card(
                "🛡️", "MACRO F1-SCORE",
                f"{report.get('f1_macro', 0):.3f}"
            )

    else:

        st.warning("Evaluation report not available.")

    st.write("")

    section_title(
        "Platform Modules",
        "Choose a module to begin your analysis."
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### 🖼️ Image Intelligence")

            st.write(
                "Upload images and inspect the model's "
                "fake/real predictions and confidence scores."
            )

            if st.button("Open Image Detection", use_container_width=True):
                st.session_state["navigation"] = "Image Detection"
                st.rerun()

        with st.container(border=True):

            st.markdown("### 🎬 Video Intelligence")

            st.write(
                "Extract and analyze selected video frames "
                "using EfficientNet-B0."
            )

            if st.button("Open Video Detection", use_container_width=True):
                st.session_state["navigation"] = "Video Detection"
                st.rerun()

    with col2:

        with st.container(border=True):

            st.markdown("### 📊 Model Analytics")

            st.write(
                "Explore accuracy, precision, recall, "
                "F1-score and confusion matrix."
            )

            if st.button("Open Model Evaluation", use_container_width=True):
                st.session_state["navigation"] = "Model Evaluation"
                st.rerun()

        with st.container(border=True):

            st.markdown("### 🔍 Error Investigation")

            st.write(
                "Inspect saved misclassified images "
                "and understand model prediction errors."
            )

            if st.button("View Error Analysis", use_container_width=True):
                st.session_state["navigation"] = "Model Evaluation"
                st.rerun()

    st.info(
        "Current evaluation: 20 test images. "
        "The reported metrics are preliminary and need "
        "larger independent testing."
    )


# ============================================================
# 9. IMAGE DETECTION
# ============================================================

elif page == "Image Detection":

    st.title("🖼️ Image Intelligence")

    st.caption(
        "Upload an image and run AI-powered deepfake classification."
    )

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image:

        try:

            image = Image.open(
                io.BytesIO(uploaded_image.getvalue())
            ).convert("RGB")

            left, right = st.columns([1.1, 1])

            with left:

                st.markdown(
                    '<div class="panel-title">Image Preview</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    use_container_width=True
                )

                st.caption(
                    f"{uploaded_image.name} | "
                    f"{image.width} × {image.height}"
                )

            with right:

                st.markdown(
                    '<div class="panel-title">Analysis Configuration</div>',
                    unsafe_allow_html=True
                )

                st.write("Model: EfficientNet-B0")

                st.write(f"Device: {DEVICE}")

                st.write("Task: Binary classification")

                st.write("Classes: Fake / Real")

                st.write("Input: RGB image")

                analyze = st.button(
                    "🚀 RUN DEEP ANALYSIS",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner(
                        "Extracting image features..."
                    ):

                        result = predict_image(image)

                    st.session_state["image_result"] = result

            result = st.session_state.get("image_result")

            if result:

                st.divider()

                section_title(
                    "Analysis Report",
                    "Model prediction and class probability distribution."
                )

                show_prediction(result)

        except Exception as error:

            st.error(
                f"Image processing error: {error}"
            )

    else:

        st.markdown(
            """
            <div class="panel">
                <h3>🔍 Ready for image analysis</h3>
                <p>
                    Upload a JPG or PNG image to start.
                    The model will classify the image into
                    one of its two trained categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 10. VIDEO DETECTION
# ============================================================

elif page == "Video Detection":

    st.title("🎬 Video Intelligence")

    st.caption(
        "Frame extraction and individual frame classification."
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video:

        st.video(uploaded_video)

        st.write(f"File: {uploaded_video.name}")

        sample_count = st.slider(
            "Number of frames to analyze",
            min_value=5,
            max_value=30,
            value=10
        )

        if st.button(
            "🚀 ANALYZE VIDEO",
            use_container_width=True
        ):

            temp_path = None
            cap = None

            try:

                suffix = Path(
                    uploaded_video.name
                ).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.getvalue()
                    )

                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():

                    raise ValueError(
                        "Unable to open uploaded video."
                    )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if total_frames <= 0 or fps <= 0:

                    raise ValueError(
                        "Invalid video metadata or FPS."
                    )

                actual_samples = min(
                    sample_count,
                    total_frames
                )

                indices = np.linspace(
                    0,
                    total_frames - 1,
                    actual_samples,
                    dtype=int
                )

                results = []

                progress = st.progress(0)

                status = st.empty()

                for i, index in enumerate(indices):

                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(index)
                    )

                    success, frame = cap.read()

                    if not success:
                        continue

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_image = Image.fromarray(rgb)

                    result = predict_image(frame_image)

                    results.append({
                        "Frame": int(index),
                        "Time (sec)": round(index / fps, 2),
                        "Prediction": result["prediction"],
                        "Confidence (%)": result["confidence"]
                    })

                    progress.progress(
                        (i + 1) / actual_samples
                    )

                    status.write(
                        f"Analyzing frame {i + 1} of {actual_samples}"
                    )

                if not results:

                    raise ValueError(
                        "No video frames could be analyzed."
                    )

                fake_count = sum(
                    item["Prediction"].lower() == "fake"
                    for item in results
                )

                real_count = sum(
                    item["Prediction"].lower() == "real"
                    for item in results
                )

                st.session_state["video_results"] = results

                status.success(
                    "Video frame analysis completed!"
                )

                st.divider()

                a, b, c = st.columns(3)

                with a:
                    metric_card(
                        "🎞️", "ANALYZED FRAMES",
                        len(results)
                    )

                with b:
                    metric_card(
                        "⚠️", "FAKE PREDICTIONS",
                        fake_count
                    )

                with c:
                    metric_card(
                        "✅", "REAL PREDICTIONS",
                        real_count
                    )

                st.subheader("Frame Analysis")

                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Prediction Distribution")

                st.bar_chart({
                    "Fake": fake_count,
                    "Real": real_count
                })

                st.warning(
                    "These are frame-level predictions only. "
                    "They do not establish the authenticity "
                    "of the entire video."
                )

            except Exception as error:

                st.error(
                    f"Video analysis error: {error}"
                )

            finally:

                # Release OpenCV handle before deleting temp file.
                if cap is not None:
                    cap.release()
                    cap = None

                if temp_path and os.path.exists(temp_path):

                    try:
                        os.remove(temp_path)

                    except PermissionError:

                        st.warning(
                            "Windows is temporarily locking the "
                            "video file. You can remove it later."
                        )

    else:

        st.info(
            "Upload a video to begin frame-level deepfake analysis."
        )

    previous_results = st.session_state.get("video_results")

    if previous_results:

        st.divider()

        st.subheader("Previous Analysis Results")

        st.dataframe(
            pd.DataFrame(previous_results),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("📊 Model Performance Analytics")

    if not report:

        st.error(
            "Evaluation report is missing or invalid."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "🎯", "ACCURACY",
                f"{report.get('accuracy', 0) * 100:.1f}%"
            )

        with c2:
            metric_card(
                "📈", "PRECISION",
                f"{report.get('precision_macro', 0):.3f}"
            )

        with c3:
            metric_card(
                "🔎", "RECALL",
                f"{report.get('recall_macro', 0):.3f}"
            )

        with c4:
            metric_card(
                "🛡️", "F1 SCORE",
                f"{report.get('f1_macro', 0):.3f}"
            )

        st.divider()

        # Classification report

        section_title(
            "Classification Report",
            "Detailed performance for each class."
        )

        class_report = report.get(
            "classification_report", {}
        )

        rows = []

        for name, values in class_report.items():

            if isinstance(values, dict) and "precision" in values:

                rows.append({
                    "Class": name,
                    "Precision": values.get("precision", 0),
                    "Recall": values.get("recall", 0),
                    "F1-score": values.get("f1-score", 0),
                    "Support": values.get("support", 0)
                })

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

        # Confusion matrix

        st.divider()

        section_title(
            "Confusion Matrix",
            "Actual labels compared with predicted labels."
        )

        matrix = report.get("confusion_matrix", [])
        classes = report.get("classes", [])

        if (
            len(matrix) == len(classes)
            and all(len(row) == len(classes) for row in matrix)
        ):

            matrix_df = pd.DataFrame(
                matrix,
                index=[f"Actual {c}" for c in classes],
                columns=[f"Predicted {c}" for c in classes]
            )

            col1, col2 = st.columns([1, 1])

            with col1:

                st.dataframe(
                    matrix_df,
                    use_container_width=True
                )

            with col2:

                st.bar_chart(
                    matrix_df
                )

        else:

            st.warning(
                "Confusion matrix dimensions do not match class labels."
            )

        # Misclassified visualization

        st.divider()

        section_title(
            "Misclassified Images",
            "Saved visualization of incorrect model predictions."
        )

        if ERROR_IMAGE.exists():

            st.image(
                str(ERROR_IMAGE),
                use_container_width=True,
                caption="DeepTrace error analysis"
            )

        else:

            st.warning(
                "Misclassified visualization was not found."
            )

        # Training history

        st.divider()

        section_title(
            "Training History",
            "Training and validation performance across epochs."
        )

        if history:

            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])

            train_acc = history.get("train_accuracy", [])
            val_acc = history.get("val_accuracy", [])

            if train_loss and len(train_loss) == len(val_loss):

                st.subheader("Loss Curve")

                loss_df = pd.DataFrame({
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss
                })

                st.line_chart(loss_df)

            if train_acc and len(train_acc) == len(val_acc):

                st.subheader("Accuracy Curve")

                accuracy_df = pd.DataFrame({
                    "Training Accuracy": train_acc,
                    "Validation Accuracy": val_acc
                })

                st.line_chart(accuracy_df)

        else:

            st.info(
                "Training history file was not found."
            )


# ============================================================
# 12. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("🔬 About DeepTrace")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">COMPUTER VISION PROJECT</div>
            <h1>DeepTrace AI</h1>
            <p>
                An experimental deep learning system for
                fake and real image classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Technology Stack")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        - Python
        - PyTorch
        - EfficientNet-B0
        - Streamlit
        - OpenCV
        """)

    with col2:

        st.markdown("""
        - NumPy
        - Pandas
        - Pillow
        - Matplotlib
        - Deep Learning
        """)

    st.subheader("Core Features")

    st.markdown("""
    1. Deepfake image classification.
    2. Video frame extraction and analysis.
    3. Class probability visualization.
    4. Confusion matrix and classification report.
    5. Misclassified image visualization.
    6. Training and validation performance.
    """)

    st.warning(
        "This is an experimental model. Its current evaluation "
        "uses only 20 test images and should not be used as "
        "a standalone forensic verification system."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🔬 DEEPTRACE AI &nbsp; | &nbsp;
        EfficientNet-B0 &nbsp; | &nbsp;
        Deep Learning Intelligence Platform
        <br><br>
        Experimental research project · 2026
    </div>
    """,
    unsafe_allow_html=True
)
