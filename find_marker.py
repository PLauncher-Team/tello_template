import time
import cv2
from djitellopy import Tello
import numpy as np

camera_matrix = np.array(
    [[920, 0, 480], [0, 920, 360], [0, 0, 1]], dtype=np.float32
)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_length = 0.1

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

obj_points = np.array(
    [
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0],
    ],
    dtype=np.float32,
)


def find_markers(valid_ids, frame):
    if frame is None:
        return None, None, None

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    corners, ids, rejected = detector.detectMarkers(gray)

    marker_pose = None
    detected_id = None

    if ids is not None:
        ids = ids.flatten()

        for i, marker_id in enumerate(ids):
            if marker_id in valid_ids:
                cv2.aruco.drawDetectedMarkers(
                    frame, [corners[i]], np.array([[marker_id]])
                )

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners[i],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if success:
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        length=marker_length * 0.5,
                    )

                    x, y, z = tvec.flatten()
                    marker_pose = (float(x), float(y), float(z))
                    detected_id = int(marker_id)  

                break

    return frame, marker_pose, detected_id


target_ids = [4, 13]
current_id = None
found_id = set()

tello.takeoff()

while True:
    frame_raw = tello.get_frame_read().frame
    if frame_raw is None:
        continue

    frame, pose, detected_id = find_markers(target_ids, frame_raw)
    
    if pose is not None and detected_id is not None:
        current_id = detected_id

        x, y, z = pose
        print(
            f"\r Нашли {current_id} | Координаты: x: {x:.2f}, y: {y:.2f}, z: {z:.2f}",
            end="",
            flush=True,
        )

        if current_id not in found_id:
            if current_id == 4:
                tello.land()
                found_id.add(4)

            elif current_id == 13:
                tello.move_right(100)
                found_id.add(13)
                time.sleep(2)
                tello.move_forward(150)
                time.sleep(2)
                tello.rotate_clockwise(200)
                for _ in range(5):
                    tello.send_expansion_command("led 255 0 0")
                    time.sleep(0.5)
                    tello.send_expansion_command("led 0 0 0")
                    time.sleep(0.5)
                break

            # elif current_id == 3:
            #     tello.move_left(100)
            #     found_id.add(3)

            # elif current_id == 4:
            #     tello.move_left(100)
            #     found_id.add(4)

    if frame is not None:
        cv2.imshow("Tello ArUco", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print(f"Battery: {tello.get_battery()}%")
tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
