import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get pose landmarks
    results = pose.process(rgb_frame)

    # If pose landmarks are found, draw them on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract key body points (shoulders, hips, etc.)
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

        # Calculate angles and distances for posture analysis
        # Example: Check for slouching by comparing the alignment of the shoulders with the hips

        # Get the x, y coordinates for shoulder and hip joints
        shoulder_distance = np.linalg.norm([np.array([left_shoulder.x, left_shoulder.y]) - np.array([right_shoulder.x, right_shoulder.y])])
        hip_distance = np.linalg.norm([np.array([left_hip.x, left_hip.y]) - np.array([right_hip.x, right_hip.y])])

        # Simple posture analysis: If the shoulder distance > hip distance, user may be slouching
        if shoulder_distance > hip_distance:
            feedback = "Straighten your back!"
        else:
            feedback = "Your posture looks good!"

        # Display the feedback on the frame
        cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with pose landmarks and feedback
    cv2.imshow("Posture Analysis", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
