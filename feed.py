import cv2
import mediapipe as mp
import numpy as np
mp_drawing =mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap=cv2.VideoCapture(0)
##setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)as pose:
    while cap.isOpened():
        ret,frame=cap.read()
    
        #Detect stuff and render
        #1 Recolor the image to RGB
        image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Make Detection
        results=pose.process(image)

        #Recolor back to BGR
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #RENDER DETECTION
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(185,82,66),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(37,155,57),thickness=2,circle_radius=2),
                                 
                                 
                                 )
        
        #Extreact LAndmarks
        try:
            landmarks=results.pose_landmarks.landmark
            
            #get coordinates
            shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW .value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            
            #calclute angle
            angle=calculate_angle(shoulder,elbow,wrist)
            
            #visulize
            cv2.putText(image, str(angle),
                          tuple(np.multiply(elbow,[640,480]).astype(int)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA
                       )
            
            print(landmarks)
        except:
            pass


        cv2.imshow('Mediapipe Feed',image)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 
