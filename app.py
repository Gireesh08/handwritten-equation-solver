import streamlit as st
import numpy as np
import cv2
import pickle
import json
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    canvas {
        border: 2px solid #cccccc;
        border-radius: 10px;
    }
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

canvas_result = st_canvas(
    fill_color = "black",
    stroke_width = 15,
    stroke_color = "white",
    background_color = "black",
    height = 200,
    width = 600,
    drawing_mode = "freedraw",
    key = "canvas"
)

# clear and predict buttons side by side
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🗑️ Clear"):
        st.rerun()
with col2:
    predict_button = st.button("🔮 Predict")

## Adding the Prediction Button
if predict_button:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = img.astype('uint8')

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # original gray → for contour detection (white on black!)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # inverted gray → for model prediction (black on white!)
        gray_inverted = cv2.bitwise_not(gray)

        ## find contours from ORIGINAL gray
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ## get ALL bounding boxes without filtering
        all_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            all_boxes.append((x, y, w, h))

        # sort all boxes left to right
        all_boxes = sorted(all_boxes, key=lambda b: b[0])

        # merge nearby contours that belong together (like ÷)
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

                # check if boxes are close horizontally
                horizontal_gap = abs(x1 - x2)

                if horizontal_gap < 50:
                    # merge these boxes!
                    merged_x = min(merged_x, x2)
                    merged_y = min(merged_y, y2)
                    merged_w = max(merged_x + merged_w, x2 + w2) - merged_x
                    merged_h = max(merged_y + merged_h, y2 + h2) - merged_y
                    skip_indices.append(j)

            boxes.append((merged_x, merged_y, merged_w, merged_h))

        # sort final boxes left to right
        boxes = sorted(boxes, key=lambda b: b[0])

        # store all predictions
        predicted_symbols = []

        for i, (x, y, w, h) in enumerate(boxes):

            # crop from INVERTED image (black on white = matches training!)
            char_img = gray_inverted[y:y+h, x:x+w]

            # make it SQUARE first!
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

            # resize to 32×32
            char_img = cv2.resize(char_img, (32, 32))

            # normalize
            char_img = char_img / 255.0

            # reshape for model
            char_img = char_img.reshape(1, 32, 32, 1)

            # predict
            prediction = model.predict(char_img)
            predicted_index = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_index])[0]

            # store prediction
            predicted_symbols.append(predicted_label)

        # symbol map
        symbol_map = {
            'add' : '+',
            'sub' : '-',
            'mul' : '*',
            'div' : '/'
        }

        # build equation string
        equation = ''
        for symbol in predicted_symbols:
            if symbol in symbol_map:
                equation += symbol_map[symbol]
            else:
                equation += symbol

        # solve equation
        try:
            result = eval(equation)
            st.success(f"✅ {equation} = {result}")
        except:
            st.error("❌ Could not solve equation! Try drawing more clearly!")

    else:
        st.warning("⚠️ Please draw something first!")