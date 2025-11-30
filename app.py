import os
from datetime import datetime

import cv2
import streamlit as st


# Function to load the cascade classifier safely
def load_cascade():
    # Uses the internal OpenCV data path so you don't need to download the XML manually
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_cascade()

def detect_faces(rect_color, min_neighbors, scale_factor):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        st.error("Could not access the webcam.")
        return

    st.toast("Webcam started! Check the popup window.")

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame.")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the user-defined parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors
        )

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
            
            # Optional: Add a label
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)

        # Add instruction text overlay on the video feed itself
        cv2.putText(frame, "Press 's' to Save | 'q' to Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frames in an external OpenCV window
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Key press handler
        key = cv2.waitKey(1) & 0xFF
        
        # Exit the loop when 'q' is pressed
        if key == ord('q'):
            break
        
        # Save the image when 's' is pressed
        if key == ord('s'):
            # Create a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_face_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}") # Print to console/terminal

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Face Detection App (Viola-Jones)")

    # --- INSTRUCTIONS ---
    st.markdown("""
    ### How to use this app:
    1. **Configure Settings:** Use the sidebar to adjust sensitivity and colors.
    2. **Start Detection:** Press the **'Detect Faces'** button below.
    3. **Save Images:** While the video window is active, press **'s'** on your keyboard to save the current frame.
    4. **Quit:** Press **'q'** inside the video window to stop.
    """)
    
    st.divider()

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("Configuration")
    
    # 1. Color Picker
    color_hex = st.sidebar.color_picker("Rectangle Color", "#00FF00")
    # Convert Hex (#RRGGBB) to RGB tuple, then to BGR for OpenCV
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    rect_color = (b, g, r) # OpenCV uses BGR
    
    # 2. Adjust minNeighbors
    # Higher value = fewer detections but higher quality (less false positives)
    min_neighbors = st.sidebar.slider(
        "Min Neighbors (Quality)", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Higher values result in fewer detections but with higher quality."
    )

    # 3. Adjust scaleFactor
    # How much the image size is reduced at each image scale
    scale_factor = st.sidebar.slider(
        "Scale Factor", 
        min_value=1.1, 
        max_value=2.0, 
        value=1.3,
        step=0.1,
        help="Specifies how much the image size is reduced at each image scale."
    )

    # --- MAIN EXECUTION ---
    st.write("Current Settings: Color:", color_hex, "| Neighbors:", min_neighbors, "| Scale:", scale_factor)

    if st.button("Detect Faces"):
        # Pass the sidebar parameters into the function
        detect_faces(rect_color, min_neighbors, scale_factor)

if __name__ == "__main__":
    main()