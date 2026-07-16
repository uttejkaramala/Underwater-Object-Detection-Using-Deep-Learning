import streamlit as st
import hashlib
import sqlite3
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import tempfile
import tensorflow as tf
import google.generativeai as genai
from PIL import Image
import io

# ------------------- Database Setup -------------------
DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and check_hashes(password, result[0]):
        return True
    return False

init_db()

def should_dehaze(image):
    """Simple heuristic: low contrast → hazy image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.std() < 40  # you can tune this value


# ------------------- Session State -------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ------------------- Gemini API Key -------------------
GEMINI_API_KEY = "INSERT YOUR API KEY HERE"

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        gemini_model = None
        st.error(f"Failed to initialize Gemini model: {e}")
else:
    st.warning("⚠️ Gemini API key not set! Chatbot will not work.")

# ------------------- Load YOLO Model -------------------
@st.cache_resource
def load_yolo_model():
    model_path = "model/best.pt"
    if not os.path.exists(model_path):
        st.error(f"⚠️ YOLO Model not found! Please place 'best.pt' at: {model_path}")
        return None
    return YOLO(model_path)

yolo_model = load_yolo_model()
class_names = [
    "animal_crab", "animal_eel", "animal_etc", "animal_fish", "animal_shells",
    "animal_starfish", "plant", "rov", "trash_bag", "trash_bottle",
    "trash_branch", "trash_can", "trash_clothing", "trash_container", "trash_cup",
    "trash_net", "trash_pipe", "trash_rope", "trash_snack_wrapper", "trash_tarp",
    "trash_unknown_instance", "trash_wreckage"
]

# ------------------- Load Dehazing Model -------------------
@st.cache_resource
def load_dehazing_model():
    model_path1 = "trained_model"
    if not os.path.exists(model_path1):
        st.error(f"⚠️ Dehazing model not found! Please place SavedModel at: {model_path1}")
        return None
    try:
        model = tf.saved_model.load(model_path1)
        return model
    except Exception as e:
        st.error(f"Failed to load dehazing model: {e}")
        return None

dehazing_model = load_dehazing_model()

# ------------------- Grad-CAM Function -------------------
def generate_gradcam_overlay(model, img_path):
    if model is None:
        return None, None, None
    
    orig_bgr = cv2.imread(img_path)
    orig_h, orig_w = orig_bgr.shape[:2]
    rgb_img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    results = model.predict(source=img_path, conf=0.25, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    
    target_layer = model.model.model[-4]
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    letterbox = LetterBox(640, stride=model.model.stride)
    preprocessed = letterbox(image=rgb_img.copy())
    input_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(model.device)
    
    with torch.no_grad():
        _ = model.model(input_tensor)
    
    handle.remove()
    
    annotated = orig_bgr.copy()
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if activations is not None and len(results.boxes) > 0:
        cam = activations[0].mean(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (orig_w, orig_h))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(annotated, 0.6, heatmap, 0.4, 0)
    else:
        overlay = annotated.copy()
    
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, overlay_rgb, results

# ------------------- Dehazing Function -------------------
def dehaze_image(img_path):
    if dehazing_model is None:
        return None, None
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    original_size = tf.shape(img)[:2]
    
    img = tf.image.resize(img, [384, 384])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    
    dehazed = dehazing_model(img)
    dehazed = tf.squeeze(dehazed, axis=0)
    dehazed = tf.clip_by_value(dehazed, 0.0, 1.0)
    dehazed = dehazed * 255.0
    dehazed = tf.cast(dehazed, tf.uint8)
    dehazed = tf.image.resize(dehazed, original_size)
    
    dehazed_np = dehazed.numpy()
    dehazed_np = np.clip(dehazed_np, 0, 255).astype(np.uint8)
    
    original_np = cv2.imread(img_path)
    original_np = cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB)
    
    return original_np, dehazed_np

# ------------------- NEW CLEAN & BEAUTIFUL CSS -------------------
st.markdown("""
    <style>
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* Beautiful title styling */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        color: #0b5394;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }

    /* Subheaders */
    h2, h3 {
        color: #0b5394;
        font-weight: 600;
    }

    /* Buttons - modern blue gradient */
    .stButton > button {
        background: linear-gradient(to right, #2980b9, #3498db);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #3498db, #5dade2);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }

    /* Text inputs - clean and modern */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #bdc3c7;
        padding: 12px;
        font-size: 16px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3498db;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: #f8fbff;
    }

    /* Chat input - REMOVE blue border */
    div[data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        border: 2px solid #bdc3c7 !important;  /* Changed from #3498db */
        padding: 14px !important;
        font-size: 16px !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fbff;
        border-right: 1px solid #e0e0e0;
    }

    /* Success, info, warning messages */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px;
        padding: 15px;
    }

    /* Image captions */
    .stImage > div > div > img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- Title with Emoji -------------------
st.title("🌊 Underwater Detection System")

# ------------------- Sidebar with Emojis -------------------
with st.sidebar:
    st.markdown("### 🌊 Navigation")
    
    if not st.session_state.logged_in:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
        if st.button("🔐 Login", use_container_width=True):
            st.session_state.page = "Login"
        if st.button("📝 Register", use_container_width=True):
            st.session_state.page = "Register"
    else:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
        if st.button("🔍 Detection", use_container_width=True):
            st.session_state.page = "Detection"
        if st.button("🖼️ Image Dehazing", use_container_width=True):
            st.session_state.page = "Image_Denoising"
        if st.button("💬 Chatbot", use_container_width=True):
            st.session_state.page = "Chatbot"
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.page = "Home"
            st.session_state.chat_history = []
            st.rerun()

# ------------------- Page Routing -------------------
if st.session_state.page == "Home":
    st.header("Welcome to Underwater Detection System")
    st.write("Advanced object detection and image enhancement for underwater environments.")
    if st.session_state.logged_in:
        st.success(f"Logged in as **{st.session_state.username}**")
    else:
        st.info("Login or register to access all features.")

elif st.session_state.page == "Login":
    st.header("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("Incorrect username or password")

elif st.session_state.page == "Register":
    st.header("📝 Register New Account")
    new_user = st.text_input("Choose Username")
    new_pass = st.text_input("Choose Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")
   
    if st.button("Create Account"):
        if not new_user or not new_pass:
            st.error("Please fill all fields")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match")
        elif len(new_pass) < 6:
            st.error("Password must be at least 6 characters")
        elif add_user(new_user, new_pass):
            st.success(f"Account '{new_user}' created successfully!")
            st.info("You can now login.")
        else:
            st.error("Username already exists")


elif st.session_state.page == "Detection":
    st.header("🔍 Underwater Object Detection")
    st.write("Upload an underwater image to detect objects and visualize model attention using Grad-CAM.")
   
    if not st.session_state.logged_in:
        st.warning("Please login to use this feature.")
        st.stop()
    
    uploaded_detection = st.file_uploader(
        "📁 Upload image for object detection",
        type=["jpg", "jpeg", "png", "bmp"],
        key="detection_uploader"
    )
   
    if uploaded_detection and yolo_model:
        with st.spinner("🔄 Running YOLO detection + generating Grad-CAM heatmap..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_detection.name)[1]) as tmp_file:
                tmp_file.write(uploaded_detection.getvalue())
                tmp_path = tmp_file.name
            
            annotated_img, gradcam_img, results = generate_gradcam_overlay(yolo_model, tmp_path)
            
            # ------------------- NEW: Dehazing Options -------------------
            st.markdown("### ⚙️ Preprocessing Options")
            use_dehaze = st.checkbox("Apply Dehazing", value=True)
            auto_mode = st.checkbox("Auto-detect haze", value=False)

            # Read image
            img = cv2.imread(tmp_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Decide whether to dehaze
            if auto_mode:
                apply_dehaze = should_dehaze(img_rgb)
            else:
                apply_dehaze = use_dehaze

            # Apply dehazing if needed
            if apply_dehaze:
                st.info("🌫️ Dehazing applied")
                _, dehazed_img = dehaze_image(tmp_path)

                temp_dehaze_path = tmp_path + "_dehazed.png"
                cv2.imwrite(temp_dehaze_path, cv2.cvtColor(dehazed_img, cv2.COLOR_RGB2BGR))
                input_path = temp_dehaze_path
            else:
                st.info("🟢 Using original image")
                input_path = tmp_path

            # Run YOLO + GradCAM
            annotated_img, gradcam_img, results = generate_gradcam_overlay(yolo_model, input_path)

            # Cleanup
            if apply_dehaze and os.path.exists(input_path):
                os.unlink(input_path)
            
            if annotated_img is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(annotated_img, caption="🟢 Detected Objects (with bounding boxes)", use_container_width=True)
               
                with col2:
                    st.image(gradcam_img, caption="🔥 Grad-CAM Heatmap (Model Attention)", use_container_width=True)
                
                st.success(f"✅ Found {len(results.boxes)} object(s)!")
                if len(results.boxes) > 0:
                    st.markdown("### 📋 Detection Details")
                    for i, box in enumerate(results.boxes):
                        cls_id = int(box.cls.item())
                        conf = box.conf.item()
                        st.write(f"**{i+1}.** {class_names[cls_id]} — Confidence: **{conf:.2f}**")
            else:
                st.error("Failed to process the image.")
    elif uploaded_detection and not yolo_model:
        st.error("Model not loaded. Check path to best.pt")

elif st.session_state.page == "Image_Denoising":
    st.header("🖼️ Underwater Image Dehazing")
    st.write("Upload a hazy underwater image to enhance clarity and remove haze using deep learning.")
    
    if not st.session_state.logged_in:
        st.warning("Please login to use this feature.")
        st.stop()
    
    uploaded_dehazing = st.file_uploader(
        "📁 Upload image for dehazing",
        type=["jpg", "jpeg", "png", "bmp"],
        key="dehazing_uploader"
    )
    
    if uploaded_dehazing is not None:
        with st.spinner("🔄 Processing image with dehazing model..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_dehazing.name)[1]) as tmp_file:
                tmp_file.write(uploaded_dehazing.getvalue())
                tmp_path = tmp_file.name
            
            original_img, dehazed_img = dehaze_image(tmp_path)
            os.unlink(tmp_path)
            
            if original_img is not None and dehazed_img is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="🌫️ Original Hazy Image", use_container_width=True)
                with col2:
                    st.image(dehazed_img, caption="✨ Dehazed Enhanced Image", use_container_width=True, clamp=True)
                
                st.success("✅ Image successfully dehazed!")
                
                # ----- Download button for dehazed image -----
                dehazed_pil = Image.fromarray(dehazed_img)
                buf = io.BytesIO()
                dehazed_pil.save(buf, format="PNG")
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Dehazed Image",
                    data=buf,
                    file_name=f"dehazed_{uploaded_dehazing.name}",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.error("Failed to dehaze the image. Check model path or input format.")
    elif dehazing_model is None:
        st.error("Dehazing model not loaded. Check path to 'trained_model'")

elif st.session_state.page == "Chatbot":
    st.header("💬 Gemini-Powered Chatbot")
    st.write("Chat with Google's latest Gemini 2.5 Flash model (fast & efficient)!")
    
    if not st.session_state.logged_in:
        st.warning("Please login to use this feature.")
        st.stop()
    
    if gemini_model is None:
        st.error("Gemini model not available. Check your API key in the code.")
        st.stop()
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = gemini_model.generate_content(prompt)
                    response_text = response.text
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
