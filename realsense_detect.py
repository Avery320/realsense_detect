import cv2
import numpy as np

# 1. 臨時 camera_matrix 與 dist_coeffs 設定（假設 640x480 解析度）
fx = 800
fy = 800
cx = 320
cy = 240
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# 2. 選用的 ArUco 字典與碼邊長（以你實際列印長度為準，單位 mm 或 cm 均可，但要一致）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
marker_length = 50  # 這裡假設 ArUco code 一邊為 50 mm

# 3. 開啟攝影機
cap = cv2.VideoCapture(0)  # 0 為預設攝影機，需根據實際情況調整

while True:
    ret, frame = cap.read()
    if not ret:
        print("攝影機無訊號")
        break

    # 4. 偵測 ArUco
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        # 5. 姿態估測
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        for i in range(len(ids)):
            # 在畫面上畫出座標軸
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length/2)
            # 標示ID與3D座標
            c = corners[i][0].mean(axis=0).astype(int)
            tvec = tvecs[i][0]  # (X, Y, Z)，單位與 marker_length 一致
            
            # 計算距離（Z軸距離）
            distance = np.linalg.norm(tvec)
            
            # 顯示ID、位置和距離
            text = f"ID:{ids[i][0]} Pos:{tvec.round(1)}"
            distance_text = f"Distance: {distance:.1f}mm"
            
            # 調整文字位置
            text_position = (c[0] - 50, c[1] - 20)  # 向左偏移並向上移動
            distance_position = (c[0] - 50, c[1] + 10)  # 在位置資訊下方顯示距離
            
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, distance_text, distance_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("ArUco 3D Pose Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
