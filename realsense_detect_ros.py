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

def rotation_matrix_to_quaternion(R):
    # 從旋轉矩陣轉換為四元數
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

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
            
            # 將旋轉向量轉換為旋轉矩陣
            R, _ = cv2.Rodrigues(rvecs[i])
            # 將旋轉矩陣轉換為四元數
            quaternion = rotation_matrix_to_quaternion(R)
            
            # ======== 更新 pose 訊息，加入旋轉資訊 ========
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
                        'x': float(quaternion[0]),
                        'y': float(quaternion[1]),
                        'z': float(quaternion[2]),
                        'w': float(quaternion[3])
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
