
# ============================================================
# BRAIN TUMOR DETECTION 
# Frontend: Streamlit
# Backend: FastAPI
# ============================================================

# Import required libraries
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI prediction endpoint
API_URL = "http://127.0.0.1:8000/predict"

# ------------------------------------------------------------
# 2. CUSTOM CSS - DARK NAVY AND TEAL THEME
# ------------------------------------------------------------

st.markdown("""
<style>
/* Main application background */
.stApp {
    background: linear-gradient(
        135deg,
        #071A14 0%,
        #0B2B20 60%,
        #103D2E 100%
    );
    color: #F0FDF4;
}

/* Main content width and spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0B2B20,
        #071A14
    );
    border-right: 1px solid #1B5E45;
}

/* General text */
h1, h2, h3 {
    color: #edf6ff !important;
}

p, label, li {
    color: #b8c8dc;
}

/* Header accent */
.hero-label {
    color: #2dd4bf;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    color: #f0f7ff;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 750;
    line-height: 1.15;
    margin-bottom: 12px;
}

.hero-description {
    color: #a7b9ce;
    font-size: 1.05rem;
    line-height: 1.8;
    max-width: 780px;
}

/* Dashboard cards */
.dashboard-card {
    background: linear-gradient(145deg, #103D2E, #0B2B20);
    border: 1px solid #1B5E45;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
}

/* Small labels inside cards */
.card-label {
    color: #8da5bf;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 650;
    margin-bottom: 12px;
}

/* Prediction result */
.prediction-name {
    color: #5eead4;
    font-size: clamp(1.6rem, 3vw, 2.3rem);
    font-weight: 750;
    margin: 8px 0;
    overflow-wrap: anywhere;
}

.prediction-description {
    color: #a7b9ce;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Teal buttons */
.stButton > button {
    background: linear-gradient(135deg, #059669, #047857);
    color: #FFFFFF;
    border: 1px solid #10B981;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #10B981, #059669);
    border-color: #34D399;
    color: #FFFFFF;
    transform: translateY(-2px);
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0c1b2d;
    border: 1px dashed #2b526b;
    border-radius: 14px;
    padding: 12px;
}

/* Dividers */
hr {
    border-color: #20364c;
}

/* Informational text boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Sidebar brand */
.brand-title {
    color: #5eead4;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.brand-subtitle {
    color: #8199b3;
    font-size: 0.8rem;
    margin-top: 3px;
}

/* Footer */
.footer {
    color: #7088a2;
    font-size: 0.8rem;
    text-align: center;
    border-top: 1px solid #20364c;
    padding-top: 20px;
    margin-top: 35px;
}

/* Mobile spacing */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .dashboard-card {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ------------------------------------------------------------

with st.sidebar:

    # Display application branding
    # Sidebar branding
    st.markdown(
    '<div class="brand-title">Brain MRI Classification</div>'
    '<div class="brand-subtitle">CNN-BASED IMAGE CLASSIFICATION</div>',
    unsafe_allow_html=True
    )
    

    st.markdown("---")

    # Let the user switch between application sections
    page = st.radio(
        "Explore",
        ["Classifier", "About Project"],
        index=0
    )

  

    st.markdown("---")

    # Display the supported categories
    st.markdown("### Supported Categories")
    st.markdown("- Glioma")
    st.markdown("- Meningioma")
    st.markdown("- No Tumor")
    st.markdown("- Pituitary")

    st.markdown("---")
    st.caption("CNN • TensorFlow • FastAPI • Streamlit")

# ------------------------------------------------------------
# 4. MRI SCANNER PAGE
# ------------------------------------------------------------

if page == "Classifier":

    # Dashboard heading
    st.markdown(
        '<div class="hero-label"></div>'
        '<div class="hero-title">Brain MRI Classification System</div>'
        '<div class="hero-description">'
        'Upload a brain MRI image to explore the prediction '
        'generated by the trained convolutional neural network.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("")
    st.markdown("")

    # Divide the main interface into upload and result columns
    upload_col, result_col = st.columns([1.05, 0.95], gap="large")

    # Store the current prediction in the Streamlit session
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    if "prediction_filename" not in st.session_state:
        st.session_state.prediction_filename = None

    with upload_col:

        # MRI upload card
        st.markdown(
            '<div class="dashboard-card">'
            '<div class="card-label">01 / IMAGE INPUT</div>'
            '<h3>Upload MRI Scan</h3>'
            '<p class="prediction-description">'
            'Select a JPG, JPEG, or PNG brain MRI image.'
            '</p></div>',
            unsafe_allow_html=True
        )

        # Accept a brain MRI image from the user
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["jpg", "jpeg", "png"],
            key="mri_uploader"
        )

        if uploaded_file is not None:

            # Read and display the uploaded image
            try:
                image_bytes = uploaded_file.getvalue()
                preview_image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                st.image(
                    preview_image,
                    caption=f"Selected image: {uploaded_file.name}",
                    use_container_width=True
                )

                st.caption(
                    f"Image dimensions: "
                    f"{preview_image.width} × {preview_image.height} pixels"
                )

                # Send the image to the backend when requested
                if st.button(
                    "Analyze MRI Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Processing image through the CNN model..."
                    ):

                        try:
                            # Send the image to the FastAPI prediction endpoint
                            response = requests.post(
                                API_URL,
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        image_bytes,
                                        uploaded_file.type
                                    )
                                },
                                timeout=60
                            )

                            # Raise an exception for HTTP errors
                            response.raise_for_status()

                            # Read the JSON prediction returned by FastAPI
                            result = response.json()

                            # Validate the essential response fields
                            predicted_class = result.get("predicted_class")
                            confidence = float(result.get("confidence"))

                            valid_classes = {
                                "glioma",
                                "meningioma",
                                "notumor",
                                "pituitary"
                            }

                            if (
                                predicted_class not in valid_classes
                                or not 0 <= confidence <= 100
                            ):
                                st.error(
                                    "The backend returned an invalid prediction."
                                )

                            else:
                                # Save the result for display in the result panel
                                st.session_state.prediction_result = {
                                    "predicted_class": predicted_class,
                                    "confidence": confidence
                                }

                                st.session_state.prediction_filename = (
                                    uploaded_file.name
                                )

                                st.success(
                                    "Analysis completed successfully."
                                )

                        # Handle an unavailable backend
                        except requests.exceptions.ConnectionError:
                            st.error(
                                "Cannot connect to FastAPI. "
                                "Start the backend using: "
                                "uvicorn app:app --reload"
                            )

                        # Handle a request timeout
                        except requests.exceptions.Timeout:
                            st.error(
                                "The request timed out. Please try again."
                            )

                        # Handle HTTP and other request errors
                        except requests.exceptions.RequestException as error:
                            st.error(f"API request failed: {error}")

                        # Handle an invalid API response
                        except (ValueError, TypeError):
                            st.error(
                                "The backend returned an invalid response."
                            )

            # Handle an unreadable image
            except Exception:
                st.error("Unable to read this image. Try another image.")

        else:
            st.info(
                "Upload an MRI image to begin. "
                "The prediction will appear in the results panel."
            )

    with result_col:

        # Results panel heading
        st.markdown(
            '<div class="dashboard-card">'
            '<div class="card-label">02 / MODEL OUTPUT</div>'
            '<h3>Prediction Results</h3>'
            '<p class="prediction-description">'
            'Your model output will appear here after analysis.'
            '</p></div>',
            unsafe_allow_html=True
        )

        # Display the latest prediction when available
        prediction = st.session_state.prediction_result

        if prediction is not None:

            predicted_class = prediction["predicted_class"]
            confidence = prediction["confidence"]

            # Format the class name for display
            display_class = predicted_class.replace(
                "_", " "
            ).title()

            # Display the predicted category
            st.markdown(
                '<div class="dashboard-card">'
                '<div class="card-label">PREDICTED CATEGORY</div>'
                f'<div class="prediction-name">{display_class}</div>'
                '<p class="prediction-description">'
                'This is the category selected by the trained CNN model.'
                '</p></div>',
                unsafe_allow_html=True
            )

            # Display the model confidence score
            st.markdown(
                '<div class="dashboard-card">'
                '<div class="card-label">MODEL CONFIDENCE SCORE</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(
                int(round(confidence)),
                text=f"Model output score: {confidence:.2f}%"
            )

            # Display the filename associated with the result
            st.caption(
                f"Analyzed file: {st.session_state.prediction_filename}"
            )

            # Explain how to interpret the score
            st.warning(
                "This score reflects the model's output, not clinical "
                "certainty. This educational application is not a "
                "medical diagnostic tool."
            )
            # Explain which images the model supports
            st.warning(
                "⚠️ Brain MRI images only. This model is designed to "
                "classify brain MRI scans into four categories. "
                "Uploading images of other body parts, X-rays, or unrelated "
                "images may produce incorrect predictions."
            )

        else:
            # Show a placeholder before the first prediction
            st.markdown(
                '<div class="dashboard-card">'
                '<div style="font-size:2.5rem;">⌁</div>'
                '<h3>Awaiting MRI Analysis</h3>'
                '<p class="prediction-description">'
                'Upload an image and select Analyze MRI Image '
                'to view the predicted category and confidence score.'
                '</p></div>',
                unsafe_allow_html=True
            )

# ------------------------------------------------------------
# 5. ABOUT PROJECT PAGE
# ------------------------------------------------------------

elif page == "About Project":

    # Display project information
    st.markdown(
        '<div class="hero-label"></div>'
        '<div class="hero-title">About</div>'
        '<div class="hero-description">'
        'A brain MRI image classification project built with '
        'deep learning and Python web technologies.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Explain the project objective
    st.markdown("### Project Objective")

    st.write(
        "This project uses a Convolutional Neural Network (CNN) "
        "to classify brain MRI images into four dataset categories."
    )

    # Describe the four output classes
    st.markdown("### Classification Categories")

    category_col1, category_col2 = st.columns(2)

    with category_col1:
        st.markdown(
            '<div class="dashboard-card">'
            '<h3>Glioma</h3>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="dashboard-card">'
            '<h3>No Tumor</h3>'
            '</div>',
            unsafe_allow_html=True
        )

    with category_col2:
        st.markdown(
            '<div class="dashboard-card">'
            '<h3>Meningioma</h3>'
            '<p class="prediction-description">'
           
            '</p></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="dashboard-card">'
            '<h3>Pituitary</h3>'
            '<p class="prediction-description">'
            '</p></div>',
            unsafe_allow_html=True
        )

    # Summarize the technology stack
    st.markdown("### Technology Stack")

    st.markdown(
    """
    - **Language:** Python
    - **Deep Learning:** TensorFlow, Keras, CNN
    - **Image Processing:** OpenCV, NumPy
    - **Frontend:** Streamlit
    - **Backend API:** FastAPI
    - **Evaluation & Visualization:** Scikit-learn, Matplotlib, Seaborn
    """
    )

    # Show the reported test performance
    st.markdown("### Model Evaluation")

    st.metric(
        label="Reported Test Accuracy",
        value="95.12%"
    )

    st.caption(
        "Reported evaluation result from the project's test dataset. "
        "Performance on other images may differ."
    )

    # Medical safety disclaimer
    st.markdown("### Important Disclaimer")

    st.warning(
        "This project is intended for educational and research purposes. "
        "It has not been clinically validated and must not be used to "
        "diagnose, exclude, or guide treatment for a medical condition. "
        "Consult a qualified healthcare professional for medical advice."
    )

# ------------------------------------------------------------
# 6. APPLICATION FOOTER
# ------------------------------------------------------------

st.markdown(
    '<div class="footer">'
    '•&nbsp; BRAIN TUMOR CLASSIFICATION<br>'
    'Educational deep learning project built with CNN, FastAPI and Streamlit'
    '</div>',
    unsafe_allow_html=True
)
