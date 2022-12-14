import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import mediapipe as mp
import pandas as pd
from traitlets import default


class ImageProcessor():
    def __init__(self, image):
        self.sol = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.sol.Pose()
        self.original_image = image
        self.landmark_names = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
        self.limb_names = {(12,14):"right_arm", (14,16):"right_forearm", (11,13):"left_arm", (13,15):"left_forearm", (23,25):"left_leg", (25,27):"left_calf", (24,26):"right_leg", 
                            (26,28):"right_calf", (29,31): "left_sole", (30,32): "right_sole", (28,32): "right_forefoot", (27,31): "left_forefoot", (11,23): "left_torso", (12,24): "right_torso",
                            (11,12): "shoulders", (23,24): "hips"}
        self.valid_angles = ['right_arm|right_forearm', 'left_arm|left_forearm', 
                             'right_calf|right_leg', 'left_calf|left_leg', 
                             'right_calf|right_forefoot', 'left_calf|left_forefoot', 
                             'right_arm|right_torso', 'left_arm|left_torso',
                             'right_leg|right_torso', 'left_leg|left_torso']
        self.results = self.pose.process(self.original_image)
        self.create_landmarks()
        self.create_limbs()
        self.create_angles()

    # calculations 

    def create_landmarks(self):
        landmarks = {}
        image = self.original_image
        for index, name in enumerate(self.landmark_names):
            landmarks[name] = {'id' : index}

        for id, landmark in enumerate(self.results.pose_landmarks.landmark):
            height, width, channel = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            landmarks[self.landmark_names[id]]["image_coords"] = (cx, cy)

        for id, landmark in enumerate(self.results.pose_world_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            landmarks[self.landmark_names[id]]["world_coords"] = (x, y, z)

        self.landmarks_df = pd.DataFrame(landmarks).T

    def create_limbs(self):
        limbs = {}
        for tuple in self.sol.POSE_CONNECTIONS:
            if tuple in self.limb_names.keys():
                first_node = tuple[0]
                second_node = tuple[1]
                first_node = self.landmarks_df.loc[self.landmarks_df['id'] == first_node].iloc[0]
                second_node = self.landmarks_df.loc[self.landmarks_df['id'] == second_node].iloc[0]
                name = self.limb_names[tuple]
                # vector
                vector = np.array(second_node["image_coords"]) - np.array(first_node["image_coords"])
                limbs[name] = {"vector": vector, "first_node": first_node, "second_node": second_node}
        self.limbs = limbs

    def create_angles(self):
        self.angles = {}
        for angle in self.valid_angles:
            limb_0, limb_1 = angle.split("|")
            angle = self.calc_angle(self.limbs, limb_0, limb_1)
            self.angles[f"{limb_0}|{limb_1}"] = angle

    # helper functions

    @staticmethod
    def calc_gradient(point_0, point_1):
        return (point_1[1] - point_0[1]) / (point_1[0] - point_0[0])

    @staticmethod
    def calc_angle(limbs, limb_0, limb_1):
        if limb_0 in limbs.keys() and limb_1 in limbs.keys():
            limb_0_0 = limbs[limb_0]["first_node"]["image_coords"]
            limb_0_1 = limbs[limb_0]["second_node"]["image_coords"]
            limb_1_0 = limbs[limb_1]["first_node"]["image_coords"]
            limb_1_1 = limbs[limb_1]["second_node"]["image_coords"]

            if limb_0_0 == limb_1_0:
                a = limb_0_1
                b = limb_1_1
                common = limb_0_0
            elif limb_0_0 == limb_1_1:
                a = limb_0_1
                b = limb_1_0
                common = limb_0_0
            elif limb_0_1 == limb_1_0:
                a = limb_0_0
                b = limb_1_1
                common = limb_0_1
            elif limb_0_1 == limb_1_1:
                a = limb_0_0
                b = limb_1_0
                common = limb_0_1
            else:
                return None

            try:
                m0 = ImageProcessor.calc_gradient(a, common)
                m1 = ImageProcessor.calc_gradient(b, common)
                angle_rad = math.atan((m1 - m0) / (1 + m0 * m1))
                angle_deg = (math.degrees(angle_rad))
                angle_deg = angle_deg if angle_deg > 0 else 180 + angle_deg
                return angle_deg
            except:
                return None
        else:
            return None

    # visualization
    def draw_limbs(self, thickness=2, colour=(0,255,0)):
        image = self.original_image.copy()
        for limb in self.limbs:
            first_node = self.limbs[limb]["first_node"]["image_coords"]
            second_node = self.limbs[limb]["second_node"]["image_coords"]
            cv.line(image, first_node, second_node, colour, thickness)
        return image

    def draw_landmarks(self, radius=3, colour=(0, 0, 255)):
        image = self.original_image.copy()
        for row in self.landmarks_df.iterrows():
            landmark = row[1]
            cv.circle(image, landmark["image_coords"], radius, colour, -1)
        return image

    def label_limbs(self):
        image = self.original_image.copy()
        for limb in self.limbs.keys():
            first_node = self.limbs[limb]["first_node"]
            second_node = self.limbs[limb]["second_node"]
            x = (first_node["image_coords"][0] + second_node["image_coords"][0]) / 2
            y = (first_node["image_coords"][1] + second_node["image_coords"][1]) / 2
            cv.putText(image, limb, (int(x) - 20, int(y) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        return image

    def label_angles(self):
        image = self.original_image.copy()
        for index, angle in enumerate(self.valid_angles):
            limb_0, limb_1 = angle.split("|")
            limb_0_0 = self.limbs[limb_0]["first_node"]
            limb_0_1 = self.limbs[limb_0]["second_node"]
            limb_1_0 = self.limbs[limb_1]["first_node"]
            limb_1_1 = self.limbs[limb_1]["second_node"]
            # find the common node
            if limb_0_0["id"] == limb_1_0["id"]:
                common_node = limb_0_0
            elif limb_0_0["id"] == limb_1_1["id"]:
                common_node = limb_0_0
            elif limb_0_1["id"] == limb_1_0["id"]:
                common_node = limb_0_1
            elif limb_0_1["id"] == limb_1_1["id"]:
                common_node = limb_0_1
            else:
                continue
            angle = self.calc_angle(self.limbs, limb_0, limb_1)
            self.angles[f"{limb_0}|{limb_1}"] = angle
            x = common_node["image_coords"][0]
            y = common_node["image_coords"][1]
            cv.putText(image, str(round(angle, 1)), (int(x) + 20, int(y) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
        return image

    # getters
    def get_landmarks(self):
        return self.landmarks_df
    
    def get_limbs(self):
        return self.limbs

    def get_angles(self):
        return self.angles

    def get_specific_angle(self, limb_0, limb_1):
        if f"{limb_0}|{limb_1}" in self.angles.keys():
            return self.angles[f"{limb_0}|{limb_1}"]
        elif f"{limb_1}|{limb_0}" in self.angles.keys():
            return self.angles[f"{limb_1}|{limb_0}"]
        else:
            raise Exception("Angle not found")

    def display(self, option = 0):
        match option:
            case 0:
                return self.original_image
            case 1:
                return self.label_limbs()
            case 2:
                return self.label_angles()
            case 3:
                return self.draw_landmarks()
            case 4:
                return self.draw_limbs()
            case default:
                return self.original_image

class ResultProcessor():
    def __init__(self, result):
        self.result = result

    def count_landmarks(self):
        for index, landmark in enumerate(self.result.pose_landmarks.landmark):
            if landmark.visibility < 0.5:
                return index

def main():
    running_image = cv.imread("../inputs/run_0.jpg")
    # running = process_image(running_image)
    # test_img = running.label_angles()
    cv.imshow("test", running_image)
    cv.waitKey(0)

if __name__ == '__main__':
    main()