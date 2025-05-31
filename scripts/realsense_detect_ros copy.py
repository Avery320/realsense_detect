import cv2
import numpy as np
import roslibpy
import time

# 1. rosbridge 連線（連到你的 ROS Docker 容器的 IP 與 port）
ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()
pose_pub = roslibpy.Topic(ros, '/aruco/pose', 'geometry_msgs/PoseStamped')

# 2. 相機參數
fx = 800
fy = 800
cx = 320
cy = 240
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
marker_length = 50  # mm

# 3. 開啟攝影機
cap = cv2.VideoCapture(0)

while ros.is_connected:
    ret, frame = cap.read()
    if not ret:
        print("攝影機無訊號")
        break

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        for i in range(len(ids)):
            # 顯示結果與 OpenCV 標註畫面
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length/2)
            c = corners[i][0].mean(axis=0).astype(int)
            tvec = tvecs[i][0]
            text = f"ID:{ids[i][0]} Pos:{tvec.round(1)}"
            cv2.putText(frame, text, (c[0] - 50, c[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            
            # ======== 新增這段：publish pose 到 ROS ========
            # rvec 是旋轉向量，可以轉成四元數，這裡暫以無旋轉處理（或用 tf 轉換）
            pose_msg = {
                'header': {
                    'stamp': {
                        'secs': int(time.time()),
                        'nsecs': int((time.time() % 1) * 1e9)
                    },
                    'frame_id': 'realsense_link'
                },
                'pose': {
                    'position': {
                        'x': float(tvec[0]) / 1000,
                        'y': float(tvec[1]) / 1000,
                        'z': float(tvec[2]) / 1000
                    },
                    'orientation': {
                        'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0  # 可轉換 rvec 為四元數
                    }
                }
            }
            pose_pub.publish(pose_msg)
            # ===========================================

    cv2.imshow("ArUco 3D Pose Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ros.terminate()
