import streamlit as st
from ultralytics import YOLO
import os
import tempfile

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Function to add logs
def add_log(message):
    st.session_state.logs.append(message)

# Streamlit UI for user inputs
st.title("YOLOv8 Object Detection, Segmentation, and Pose Estimation")

# Define layout columns
col1, col2 = st.columns([3, 1])  # Adjust column ratio as needed

with col1:
    # Main content and inputs
    st.header("Settings")
    
    # Mode Selection (detect, segment, pose)
    MODE = st.selectbox("Select Mode", ["detect", "segment", "pose"])

    # Weights file upload
    uploaded_file = st.file_uploader("Upload YOLO weights (.pt file)", type=["pt"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location and rename it with a .pt extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            PT_FILE_PATH = tmp_file.name
        add_log(f"Uploaded weights file saved as: {PT_FILE_PATH}")
    else:
        st.warning("Please upload the YOLO weights file to continue.")

    # Input source images upload
    SOURCE_FOLDER = st.file_uploader("Upload image(s)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    # Ensure that the uploaded images are processed correctly
    image_paths = []
    if SOURCE_FOLDER:
        for uploaded_file in SOURCE_FOLDER:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                img_path = tmp_file.name
                image_paths.append(img_path)
        add_log("Images from SOURCE_FOLDER are all uploaded successfully")
    else:
        st.warning("No images uploaded.")

    # Output folder (user specifies the folder path)
    OUTPUT_FOLDER = st.text_input("Enter the path to the desired output folder (e.g., C:/Users/YourName/SharedFolder)")

    # Ensure the output folder exists
    if OUTPUT_FOLDER:
        if os.path.exists(OUTPUT_FOLDER):
            st.success(f"Output folder path is valid: {OUTPUT_FOLDER}")
        else:
            st.error(f"Output folder path is invalid or not accessible: {OUTPUT_FOLDER}")

    # Output folder name
    OUTPUT_FOLDER_NAME = st.text_input("Name of the folder to save results", value="results")

    # Confidence threshold slider
    CONF_THRESHOLD = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25)

    # Option to save cropped images
    IS_SAVE_CROP = st.checkbox("Save cropped images", value=False)

    # Option to save TXT predictions
    IS_SAVE_TXT = st.checkbox("Save TXT predictions", value=False)

    # Start the process when button is clicked and the weights file is uploaded
    if st.button("Start YOLO Processing") and uploaded_file is not None:
        if not image_paths:
            add_log("No valid images to process.")
        elif not os.path.exists(OUTPUT_FOLDER):
            add_log(f"Output folder does not exist: {OUTPUT_FOLDER}")
        else:
            # Creating the YOLOv8 model
            model = YOLO(PT_FILE_PATH)

            # Running prediction on uploaded images
            results = model.predict(
                image_paths,
                save=True,
                conf=CONF_THRESHOLD,
                project=OUTPUT_FOLDER,
                name=OUTPUT_FOLDER_NAME,
                save_crop=IS_SAVE_CROP,
                save_txt=IS_SAVE_TXT,
            )

            # Processing results based on selected mode
            for result in results:
                if MODE == "detect":
                    if len(result) > 0:
                        add_log(f"{len(result.boxes)} objects detected.")
                    else:
                        add_log("No objects detected.")

                elif MODE == "segment":
                    if len(result) > 0:
                        add_log(f"{len(result.masks)} instances segmented.")
                    else:
                        add_log("No objects segmented.")

                elif MODE == "pose":
                    if len(result) > 0:
                        for ind in range(len(result.boxes)):
                            add_log(f"Object detected: {result.names[result.boxes[ind].cls[0].item()]}") 
                            if result.keypoints is not None:
                                coords = result.keypoints.xy[0].tolist()
                                add_log(f"Pose coordinates: {coords}")
                    else:
                        add_log("No pose estimated.")

            add_log(f"Results saved in: {os.path.join(OUTPUT_FOLDER, OUTPUT_FOLDER_NAME)}")

with col2:
    # Text area for log messages
    st.header("Logs")
    log_text_area = st.text_area(
        "Log Messages",
        value="\n".join(st.session_state.logs),
        height=500,  # Adjust height as needed
        max_chars=None
    )

    # To update the log text area, call `st.experimental_rerun()` after appending to the logs
    if st.button("Refresh Logs"):
        log_text_area = "\n".join(st.session_state.logs)
        # st.experimental_rerun()
