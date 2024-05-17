import streamlit as st
from PIL import Image
# with st.spinner('Loading,.....'):
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model('model.h5')
import os
import math
from sendemailimage import send_email



# page config
st.set_page_config(
    page_title="Fall",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #000000;
    }
</style>
""", unsafe_allow_html=True)
top_image = Image.open(r'static\2.png')
st.sidebar.image(top_image,use_column_width='auto')
email = st.sidebar.text_input('Enter Email ID', 'mathewannette9@gmail.com')

st.title('🏠 FALL Detection')
st.divider()

col1, col2 = st.columns([3, 3])

    

with col1:
    st.markdown("""
               
        <p style="text-align:justify; font-size:18px">Please upload the video for FALL detection here. The output will be visible in right.</p>
""", unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Upload the Video (should be of format '.mp4')", type=["mp4"])
    
with col2:
    x = st.image([])

y1, y2, y3 = st.columns([1, 1, 1])
with y2:
    detect_but = st.button("DETECT FALL", use_container_width=True)
sendmail = 1
if detect_but:
    try:

        with st.spinner('Processing the Video,.....'):
            if not os.path.exists("./temp"):
                os.makedirs("./temp")

            with open(os.path.join("temp", "uploaded_media.mp4"), "wb") as f:
                f.write(video_file.getbuffer())
            
                
            cap = cv2.VideoCapture(r"temp/uploaded_media.mp4")
            fall_detected = False
            while cap.isOpened():
                ret, frame = cap.read() 
                if not ret:
                    break
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                results = pose.process(image)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Render detections
                img = np.zeros(image.shape,dtype=np.uint8)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=3), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=3) 
                                                ) 
                img = cv2.resize(img, (360, 200))
                result = model(np.expand_dims(img,axis=0))
                clear = lambda: os.system('cls')
                clear()
                pred = np.argmax(result)
                flag1 = pred


                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark
                    
                    x_values = [landmark.x*w for landmark in landmarks]
                    y_values = [landmark.y*h for landmark in landmarks]
                    xmin, ymin = min(x_values),min(y_values)
                    xmax, ymax = max(x_values),max(y_values)
                    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    left_body_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                    left_body_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
                    right_body_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
                    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
                    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
                    dx = int(xmax) - int(xmin)
                    dy = int(ymax) - int(ymin)
                    difference = dy - dx
                    
                    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                            len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                            or difference < 0:
                            flag2 = 1
                            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255),
                                    thickness=2, lineType=cv2.LINE_AA)
                            
                    else:
                        flag2 = 0
                        
                    if flag2 and flag1:
                        cv2.putText(image, 'fall detected', (int(xmin)+5,int(ymin)+5), 0, 1, [255, 0,0], thickness=1, lineType=cv2.LINE_AA)
                        fall_detected=True
                        if sendmail:
                            sendmail=0
                            cv2.imwrite("images.png",frame)       
                        
                    x.image( image)

            # x.image(r"static\2.png")

            
            x1, x2 = st.columns([5, 1])

            with x1:
                st.success(
                    "FALL detection completed ", icon="✅")
                if fall_detected:
                     st.error("FALL DETECTED")
                     
                else:
                    st.success("NO FALL DETECTED")
                    

    except Exception as e:
        print(e)
        st.error(
            "There has been some issue, make sure a file was uploaded and the type is .mp4 only. Please Retry !!!", icon="⚠️")


if not sendmail:
    filename = 'images.png'
    receiver_email = email
    send_email(filename,receiver_email)