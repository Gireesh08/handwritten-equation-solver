import streamlit as st
import numpy as np
import cv2
import pickle
import json
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Apple-style UI
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Title styling */
    h1 {
        color: #1d1d1f !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    /* Subheader styling */
    h3 {
        color: #1d1d1f !important;
        font-weight: 600 !important;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #d2d2d7 !important;
        color: #1d1d1f !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0071e3 !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        padding: 8px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #0077ed !important;
    }

    /* Success message */
    .stSuccess {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        color: #1d1d1f !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    /* Warning message */
    .stWarning {
        border-radius: 12px !important;
    }

    /* Error message */
    .stError {
        border-radius: 12px !important;
    }

    /* Divider */
    hr {
        border-color: #d2d2d7 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# loading the trained CNN model
model = load_model('handwritten_equation_solver.h5')

# loading the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# loading the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# app title
st.title("✏️ Handwritten Equation Solver")
st.markdown("Drawn equations solved instantly using AI!")
st.markdown("---")

# instructions panel
st.info("""
📝 **How to draw symbols:**

| Symbol | Draw |
|--------|------|
| Addition | **+** |
| Subtraction | **-** |
| Multiplication | **×** |
| Division | **÷** (dots above and below!) |
| Digits | **0 to 9** |

⚠️ **Tips:**
- Draw **1** with a horn on top + vertical line!
- Draw **÷** with clear dots above and below!
- Keep symbols separated with small gaps!
- Use clear, bold strokes!
""")

st.markdown("---")
st.subheader("🎨 Draw your equation below!")

# canvas width matches container
canvas_result = st_canvas(
    fill_color = "black",
    stroke_width = 15,
    stroke_color = "white",
    background_color = "#1d1d1f",
    height = 200,
    width = 700,
    drawing_mode = "freedraw",
    key = "canvas"
)

st.markdown("<br>", unsafe_allow_html=True)

# buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗑️ Clear"):
        st.rerun()
with col2:
    predict_button = st.button("🔮 Predict Equation")

st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = img.astype('uint8')

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # original gray → for contour detection
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # inverted gray → for model prediction
        gray_inverted = cv2.bitwise_not(gray)

        # find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get all bounding boxes
        all_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            all_boxes.append((x, y, w, h))

        # sort left to right
        all_boxes = sorted(all_boxes, key=lambda b: b[0])

        # merge nearby contours
        boxes = []
        skip_indices = []

        for i, (x1, y1, w1, h1) in enumerate(all_boxes):
            if i in skip_indices:
                continue

            merged_x = x1
            merged_y = y1
            merged_w = w1
            merged_h = h1

            for j, (x2, y2, w2, h2) in enumerate(all_boxes):
                if i == j or j in skip_indices:
                    continue

                horizontal_gap = abs(x1 - x2)

                if horizontal_gap < 50:
                    merged_x = min(merged_x, x2)
                    merged_y = min(merged_y, y2)
                    merged_w = max(merged_x + merged_w, x2 + w2) - merged_x
                    merged_h = max(merged_y + merged_h, y2 + h2) - merged_y
                    skip_indices.append(j)

            boxes.append((merged_x, merged_y, merged_w, merged_h))

        # sort final boxes
        boxes = sorted(boxes, key=lambda b: b[0])

        # store predictions
        predicted_symbols = []

        for i, (x, y, w, h) in enumerate(boxes):

            char_img = gray_inverted[y:y+h, x:x+w]

            # make square
            size = max(w, h)
            square = np.ones((size, size), dtype=np.uint8) * 255
            x_offset = (size - w) // 2
            y_offset = (size - h) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

            # add padding
            padding = 20
            char_img = cv2.copyMakeBorder(
                square,
                padding, padding, padding, padding,
                cv2.BORDER_CONSTANT,
                value=255
            )

            # resize normalize reshape
            char_img = cv2.resize(char_img, (32, 32))
            char_img = char_img / 255.0
            char_img = char_img.reshape(1, 32, 32, 1)

            # predict
            prediction = model.predict(char_img)
            predicted_index = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_index])[0]
            predicted_symbols.append(predicted_label)

        # symbol map
        symbol_map = {
            'add' : '+',
            'sub' : '-',
            'mul' : '*',
            'div' : '/'
        }

        # build equation
        equation = ''
        for symbol in predicted_symbols:
            if symbol in symbol_map:
                equation += symbol_map[symbol]
            else:
                equation += symbol

        # solve equation
        try:
            result = eval(equation)
            st.success(f"✅   {equation} = {result}")
        except:
            st.error("❌ Could not solve! Try drawing more clearly!")

    else:
        st.warning("⚠️ Please draw something first!")