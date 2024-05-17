import streamlit as st
from PIL import Image
import cv2
import numpy as np
from sendboundarymail import send_email

# page config
st.set_page_config(
    page_title="Boundary",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title('🎥Boundary Detection')


st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #000000;
    }
</style>
""", unsafe_allow_html=True)
top_image = Image.open(r'static\2.png')
st.sidebar.image(top_image,use_column_width='auto')

# checkboxes
st.info('✨ The Live Feed from Camera will take some time to load up 🎦')
col1, col2 ,col3  = st.columns([1,6,4],gap='large')
with col2:
    live_feed = st.checkbox('Start Camera ✅')

with col3:
    available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
    cam_id = st.selectbox(
    "Select which camera signal to use", list(available_cameras.keys()))
# camera section    
col1, col2 ,col3 = st.columns([1,5,1])
with col2:
    frame_placeholder = st.image('static/live.png')
    
email = st.sidebar.text_input('Enter Email ID', 'mathewannette9@gmail.com')   

def main():
    # Create background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Define boundary coordinates
    # boundary = [(10, 10), (100, 10), (100, 450), (10, 450)]  
    boundary = [(10, 10), (100, 10), (100, 450), (10, 450)]  

    # Initialize webcam
    cap = cv2.VideoCapture(available_cameras[cam_id])
    bflag = 0
    while cap.isOpened() and live_feed:
        ret, frame = cap.read()
        if not ret:
            break
        
    # Apply background subtraction
        fgmask = fgbg.apply(frame)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundary
        cv2.polylines(frame, [np.array(boundary)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Detect trespassing
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 12500:  # Adjust area threshold 
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Check if centroid is inside boundary
                if cv2.pointPolygonTest(np.array(boundary), (cx, cy), False) == 1:
                    # trespass detected, draw bounding box and alert
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # print("trespassing detected at ({}, {})".format(cx, cy))
                    cv2.putText(frame, 'trespassing detected', (int(cx),int(cy)), 0, 1, [255, 0,0], thickness=1, lineType=cv2.LINE_AA)
                    bflag = bflag+1
                    if bflag < 3:
                        cv2.imwrite("boundary.png",frame)
                    if bflag>5:
                        bflag = 0
                        filename = 'boundary.png'
                        receiver_email = email
                        send_email(filename,receiver_email)
                        
                        
                        
        # Display the resulting frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)
        
        # Exit on 'q' press
        if (cv2.waitKey(1) & 0xFF ==ord("q")) or not(live_feed):
            break
    # Release the capture 
    cap.release()

if __name__ == "__main__":
    main()