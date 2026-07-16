import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import cv2
import random
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ------------------- Load YOLO -------------------
@st.cache_resource
def load_yolo_model():
    model_path = "model/best.pt"
    if not os.path.exists(model_path):
        st.error(f"⚠️ YOLO Model not found! {model_path}")
        return None
    return YOLO(model_path)

# ------------------- Load Dehazing -------------------
@st.cache_resource
def load_dehazing_model():
    model_path = "trained_model"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Dehazing model not found! {model_path}")
        return None
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load dehazing model: {e}")
        return None

# Initialize
yolo_model = load_yolo_model()
dehazing_model = load_dehazing_model()

# ------------------- Dehazing -------------------
def process_dehazing(image, model):
    if model is None:
        return image, False

    try:
        h_orig, w_orig = image.shape[:2]

        input_img = cv2.resize(image, (384, 384))
        input_img = input_img.astype(np.float32) / 255.0

        input_tensor = tf.convert_to_tensor(input_img, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        # ✅ Correct call
        output = model(input_tensor, training=False)

        output = tf.squeeze(output, axis=0)
        output = tf.clip_by_value(output, 0.0, 1.0)
        output = (output * 255.0).numpy().astype(np.uint8)

        output = cv2.resize(output, (w_orig, h_orig))

        return output, True

    except Exception as e:
        import traceback
        print("❌ Dehazing failed:")
        traceback.print_exc()
        return image, False

# ------------------- Confidence Helper -------------------
def get_conf_stats(results):
    if len(results[0].boxes) == 0:
        return 0, 0.0, 0.0

    confs = results[0].boxes.conf.cpu().numpy()
    avg_conf = float(np.mean(confs))
    max_conf = float(np.max(confs))

    return len(confs), avg_conf, max_conf

# ------------------- UI -------------------
st.set_page_config(page_title="Underwater Validation", layout="wide")
st.title("🌊 Automated Pipeline Evaluation")
st.info("Compare YOLO detection on RAW vs DEHAZED images with confidence analysis")

TEST_FOLDER = r"C:\underwater object detection\TK201616 - Under water object Detection\CODE\BACKEND\valid_folder\images"

if st.button("🔍 Run Comparative Test (Random 10)"):

    if not os.path.exists(TEST_FOLDER):
        st.error(f"Folder not found: {TEST_FOLDER}")

    else:
        files = [f for f in os.listdir(TEST_FOLDER)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(files) == 0:
            st.warning("No images found.")

        else:
            selected_files = random.sample(files, min(10, len(files)))
            stats = []

            for filename in selected_files:
                path = os.path.join(TEST_FOLDER, filename)
                img = cv2.imread(path)

                # -------- RAW --------
                res_raw = yolo_model.predict(img, conf=0.25, verbose=False)
                count_raw, avg_raw, max_raw = get_conf_stats(res_raw)

                # -------- DEHAZED --------
                img_dehazed, success = process_dehazing(img, dehazing_model)

                if not success:
                    st.warning(f"⚠️ Dehazing failed for {filename}")

                res_dehaze = yolo_model.predict(img_dehazed, conf=0.25, verbose=False)
                count_dehaze, avg_dehaze, max_dehaze = get_conf_stats(res_dehaze)

                # -------- DISPLAY --------
                st.write(f"## {filename}")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.image(img[:, :, ::-1], caption="Original")

                with c2:
                    st.image(img_dehazed[:, :, ::-1], caption="Dehazed")

                with c3:
                    st.image(res_dehaze[0].plot()[:, :, ::-1],
                             caption=f"Detected: {count_dehaze}")

                # -------- METRICS --------
                st.write("### 📊 Metrics")
                st.write(f"Objects → Raw: {count_raw} | Dehazed: {count_dehaze}")
                st.write(f"Avg Confidence → Raw: {avg_raw:.3f} | Dehazed: {avg_dehaze:.3f}")
                st.write(f"Max Confidence → Raw: {max_raw:.3f} | Dehazed: {max_dehaze:.3f}")

                stats.append({
                    "File": filename,
                    "Raw_Count": count_raw,
                    "Dehazed_Count": count_dehaze,
                    "Gain": count_dehaze - count_raw,
                    "Raw_AvgConf": round(avg_raw, 3),
                    "Dehazed_AvgConf": round(avg_dehaze, 3),
                    "Conf_Improvement": round(avg_dehaze - avg_raw, 3)
                })

                st.divider()

            # -------- SUMMARY --------
            st.header("📊 Performance Summary")
            st.table(stats)

            total_gain = sum(s['Gain'] for s in stats)
            avg_conf_gain = np.mean([s['Conf_Improvement'] for s in stats])

            st.subheader("📈 Overall Results")

            if total_gain > 0:
                st.success(f"✅ Object Improvement: +{total_gain}")
            elif total_gain < 0:
                st.warning(f"⚠️ Object Drop: {total_gain}")
            else:
                st.info("No change in object count")

            if avg_conf_gain > 0:
                st.success(f"✅ Confidence Improved: +{avg_conf_gain:.3f}")
            elif avg_conf_gain < 0:
                st.warning(f"⚠️ Confidence Dropped: {avg_conf_gain:.3f}")
            else:
                st.info("No change in confidence")