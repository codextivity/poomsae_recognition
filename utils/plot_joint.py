import numpy as np
import matplotlib.pyplot as plt

npz_path = r"compare\references_mediapipe\frames\14_1\keypoints.npz"
data = np.load(npz_path, allow_pickle=True)

kp = data["keypoints_norm"]   # (N, 26, 3)

std_x = np.std(kp[:, :, 0], axis=0)
std_y = np.std(kp[:, :, 1], axis=0)
spatial_std = np.sqrt(std_x**2 + std_y**2)

joint_names = [
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_sho","r_sho","l_elb","r_elb","l_wri","r_wri",
    "l_hip","r_hip","l_knee","r_knee","l_ank","r_ank",
    "head","neck","hip","l_big_toe","r_big_toe",
    "l_small_toe","r_small_toe","l_heel","r_heel"
]

plt.figure(figsize=(12, 4))
plt.bar(range(26), spatial_std)
plt.xticks(range(26), joint_names, rotation=60, ha="right")
plt.ylabel("STD in normalized coordinates")
plt.title("Per-joint spatial STD")
plt.tight_layout()
plt.show()
