import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

st.set_page_config(page_title="Safety & Triple-Riding Detector", page_icon="🪖", layout="wide")

st.title("🪖 Real-Time Safety & Helmet Detection")
st.subheader("Helmet detection + triple-riding detection from an uploaded photo")
st.write(
    "Upload a JPG, JPEG, or PNG image. The COCO model detects people and motorcycles; "
    "an optional custom helmet model detects helmet/no-helmet classes."
)

@st.cache_resource
def load_coco_model():
    # Downloads yolov8n.pt automatically the first time if it is not already present.
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_helmet_model(model_path):
    return YOLO(model_path)

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    helmet_model_path = st.text_input(
        "Custom helmet model path (optional)",
        value="best.pt",
        help="Use a trained YOLO model whose classes include helmet/no_helmet. Leave blank if unavailable."
    )
    st.caption(
        "Triple-riding is an estimate based on how many detected people overlap a motorcycle box. "
        "Camera angle and occlusion can affect accuracy."
    )

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def box_iou_area_ratio(person_box, bike_box):
    # Returns the fraction of the person's box that overlaps the motorcycle box.
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    person_area = max(1, (px2 - px1) * (py2 - py1))
    return intersection / person_area

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    left, right = st.columns(2)
    with left:
        st.markdown("### Original image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        coco_model = load_coco_model()
        coco_result = coco_model.predict(image_np, conf=confidence, verbose=False)[0]

        people = []
        motorcycles = []
        annotated = image_np.copy()

        # COCO class IDs: person=0, motorcycle=3
        if coco_result.boxes is not None:
            for box in coco_result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                coords = (x1, y1, x2, y2)
                if cls_id == 0:
                    people.append({"box": coords, "conf": conf})
                elif cls_id == 3:
                    motorcycles.append({"box": coords, "conf": conf})

        # Draw detected people and motorcycles.
        for p in people:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 180, 40), 2)
            cv2.putText(annotated, f"Person {p['conf']:.2f}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 180, 40), 2)

        triple_flags = []
        for i, bike in enumerate(motorcycles, start=1):
            bx1, by1, bx2, by2 = bike["box"]
            # Expand bike box upward and sideways to include riders above/on the bike.
            bw, bh = bx2 - bx1, by2 - by1
            rider_zone = (
                max(0, int(bx1 - 0.35 * bw)),
                max(0, int(by1 - 1.35 * bh)),
                min(image_np.shape[1] - 1, int(bx2 + 0.35 * bw)),
                min(image_np.shape[0] - 1, int(by2 + 0.10 * bh)),
            )
            rider_count = 0
            for p in people:
                px1, py1, px2, py2 = p["box"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                rx1, ry1, rx2, ry2 = rider_zone
                # Person center in the expanded motorcycle zone, or enough overlap with bike.
                if (rx1 <= pcx <= rx2 and ry1 <= pcy <= ry2) or box_iou_area_ratio(p["box"], bike["box"]) > 0.12:
                    rider_count += 1

            is_triple = rider_count >= 3
            triple_flags.append((i, rider_count, is_triple))
            color = (0, 0, 255) if is_triple else (255, 160, 0)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)
            label = f"Motorcycle {i}: {rider_count} person(s)" + (" - TRIPLE RIDING" if is_triple else "")
            cv2.putText(annotated, label, (bx1, max(25, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            rx1, ry1, rx2, ry2 = rider_zone
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 1)

        helmet_status = "Not checked (custom helmet model not configured)"
        if helmet_model_path.strip() and Path(helmet_model_path.strip()).is_file():
            helmet_model = load_helmet_model(helmet_model_path.strip())
            helmet_result = helmet_model.predict(image_np, conf=confidence, verbose=False)[0]
            names = helmet_result.names
            helmet_detections = []
            if helmet_result.boxes is not None:
                for box in helmet_result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
                    helmet_detections.append((class_name, conf))
                    lower_name = class_name.lower().replace("-", "_").replace(" ", "_")
                    color = (0, 200, 0) if "helmet" in lower_name and "no" not in lower_name and "without" not in lower_name else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{class_name} {conf:.2f}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            helmet_status = f"Custom model ran: {len(helmet_detections)} helmet-related detection(s)"
        elif helmet_model_path.strip():
            helmet_status = f"Custom helmet model not found at: {helmet_model_path.strip()}"

    with right:
        st.markdown("### Detection result")
        st.image(annotated, use_container_width=True)

    st.markdown("### Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("People detected", len(people))
    m2.metric("Motorcycles detected", len(motorcycles))
    m3.metric("Triple-riding flags", sum(1 for _, _, flag in triple_flags if flag))

    if motorcycles:
        for bike_id, count, flag in triple_flags:
            if flag:
                st.error(f"⚠️ Motorcycle {bike_id}: possible triple riding ({count} people associated).")
            else:
                st.success(f"Motorcycle {bike_id}: no triple-riding flag ({count} people associated).")
    else:
        st.info("No motorcycle detected. Try a clear image where the full motorcycle and riders are visible.")

    st.write("**Helmet status:**", helmet_status)
    st.warning(
        "This is a student-project prototype, not a reliable law-enforcement or workplace-safety decision system. "
        "Validate it on varied images before real-world use."
    )
else:
    st.info("Upload a photo above to run detection.")
