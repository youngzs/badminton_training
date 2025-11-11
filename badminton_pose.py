import cv2
import mediapipe as mp
import math

# 初始化姿势检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 读取视频
cap = cv2.VideoCapture('badminton_training.mp4')

# 初始化计数器和列表存储运动员坐标
counter = 0
coords = []


# 定义运动姿势判断函数
def check_pose(coords):
    # 设定关键关节点坐标阈值
    thresholds = {
        'shoulder': 30,
        'elbow': 20,
        'wrist': 15,
        'hip': 30,
        'knee': 20,
        'ankle': 15
    }

    # 关键点索引
    keypoints = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28
    }

    # 判断关键点间的距离
    for key, threshold in thresholds.items():
        key_parts = key.split('_')
        left_keypoint = keypoints[f"left_{key_parts[0]}"]
        right_keypoint = keypoints[f"right_{key_parts[0]}"]
        if abs(coords[left_keypoint][0] - coords[right_keypoint][0]) > threshold:
            print(f'{key_parts[0].capitalize()}距离过大，姿势不标准')
        else:
            print(f'{key_parts[0].capitalize()}距离正常')


# 计算运动员移动距离
def calculate_distance(coords):
    if len(coords) >= 2:
        last_frame = coords[-2]
        current_frame = coords[-1]
        hip_center_last = [(last_frame[23][0] + last_frame[24][0]) / 2, (last_frame[23][1] + last_frame[24][1]) / 2]
        hip_center_current = [(current_frame[23][0] + current_frame[24][0]) / 2,
                              (current_frame[23][1] + current_frame[24][1]) / 2]
        distance = math.sqrt((hip_center_current[0] - hip_center_last[0]) ** 2 + (
                hip_center_current[1] - hip_center_last[1]) ** 2)
        return distance
    else:
        return 0


# 循环检测视频中的每一帧
while cap.isOpened():
    # 读取帧
    ret, frame = cap.read()

    # 如果读取帧成功,进行检测
    if ret:

        # 将当前帧发送到Pose检测模型
        results = pose.process(frame)

        # 如果检测到人体,获取人体关节点的坐标
        if results.pose_landmarks:
            # 获取当前帧中人体关节点的相关坐标
            landmark_coords = []
            for _, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_coords.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # 将当前人体坐标添加到总坐标列表中
            coords.append(landmark_coords)

            # 根据坐标列表判断姿势
            check_pose(coords)

            # 计算移动距离
            distance = calculate_distance(coords)
            print(f'移动距离: {distance}')

            # 显示当前帧并标注关键点
            mp_pose.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 显示图像
        cv2.imshow('MediaPipe Pose', frame)

        # 如果q键被按下,退出循环
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 增加帧计数器
        counter += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()
