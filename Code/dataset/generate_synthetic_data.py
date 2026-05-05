import argparse
import os
import cv2
import numpy as np
from tqdm import trange
from utils import quaternion_from_matrix, quaternion_to_rotation_matrix


def random_texture(size=1024):
    pattern = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.float32) / 25.0
    pattern = cv2.filter2D(pattern, -1, kernel)
    return pattern


def make_camera_intrinsics(width, height, fx=220.0, fy=220.0):
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K


def quat_from_euler(roll, pitch, yaw):
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


def world_to_camera_extrinsics(position, quaternion):
    R = quaternion_to_rotation_matrix(quaternion)
    t = -R @ position
    return R, t


def compute_homography(K, Rcw, tcw):
    H = K @ np.hstack([Rcw[:, :2], tcw.reshape(3, 1)])
    return H


def synthesize_image(texture, pose, K, width, height):
    Rcw, tcw = world_to_camera_extrinsics(pose[:3], pose[3:])
    H = compute_homography(K, Rcw, tcw)
    H /= H[2, 2]
    img = cv2.warpPerspective(texture, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img


def integrate_pose(prev_pose, linear_velocity, angular_velocity, dt):
    p = prev_pose[:3]
    q = prev_pose[3:]
    roll, pitch, yaw = 0.0, 0.0, 0.0
    w = angular_velocity * dt
    norm = np.linalg.norm(w)
    if norm > 1e-8:
        axis = w / norm
        theta = norm
        dq = np.concatenate([[np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis])
    else:
        dq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q_new = quaternion_from_matrix(quaternion_to_rotation_matrix(q) @ quaternion_to_rotation_matrix(dq))
    p_new = p + linear_velocity * dt
    return np.concatenate([p_new, q_new])


def generate_trajectory(num_samples, dt, initial_height=2.0):
    poses = []
    pose = np.array([0.0, 0.0, initial_height, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(num_samples):
        if i % 100 == 0:
            velocity = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
            velocity[2] = np.random.uniform(-0.02, 0.02)
        yaw = 0.02 * i
        roll = np.random.uniform(-0.2, 0.2)
        pitch = np.random.uniform(-0.2, 0.2)
        q = quat_from_euler(roll, pitch, yaw)
        pose = np.concatenate([pose[:3] + velocity * dt, q])
        pose[2] = max(1.0, pose[2])
        poses.append(pose.copy())
    return np.stack(poses, axis=0)


def imu_from_poses(poses, dt, g=np.array([0.0, 0.0, -9.81], dtype=np.float32)):
    imu = []
    for i in range(1, len(poses)):
        p_prev, q_prev = poses[i - 1][:3], poses[i - 1][3:]
        p_curr, q_curr = poses[i][:3], poses[i][3:]
        accel_w = (p_curr - p_prev) / dt - g
        R_prev = quaternion_to_rotation_matrix(q_prev)
        acc_body = R_prev.T @ accel_w
        q_delta = q_curr
        omega = np.zeros(3, dtype=np.float32)
        imu.append(np.concatenate([acc_body, omega]))
    return np.stack(imu, axis=0)


def save_sequence(out_dir, seq_id, images, imu, rel_poses):
    np.savez_compressed(
        os.path.join(out_dir, f"sequence_{seq_id:02d}.npz"),
        images=images,
        imu=imu,
        rel_poses=rel_poses,
    )


def build_dataset(output_dir, num_sequences, sequence_length, cam_rate=100, imu_rate=1000, image_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    dt_imu = 1.0 / imu_rate
    dt_cam = 1.0 / cam_rate
    pts_per_frame = imu_rate // cam_rate
    K = make_camera_intrinsics(image_size[0], image_size[1])
    for seq in trange(num_sequences, desc="Generating sequences"):
        texture = random_texture(1024)
        num_imu = sequence_length * pts_per_frame + 1
        poses = generate_trajectory(num_imu, dt_imu)
        imu = imu_from_poses(poses, dt_imu)
        images = []
        rel_poses = []
        for frame_index in range(sequence_length):
            image_pose = poses[frame_index * pts_per_frame]
            images.append(synthesize_image(texture, image_pose, K, *image_size))
            next_pose = poses[(frame_index + 1) * pts_per_frame]
            rel_t = next_pose[:3] - image_pose[:3]
            rel_q = quaternion_from_matrix(quaternion_to_rotation_matrix(next_pose[3:]) @ quaternion_to_rotation_matrix(image_pose[3:]).T)
            rel_poses.append(np.concatenate([rel_t, rel_q]))
        images = np.stack(images, axis=0)
        imu_windows = np.stack([imu[i * pts_per_frame : (i + 1) * pts_per_frame] for i in range(sequence_length)], axis=0)
        save_sequence(output_dir, seq, images, imu_windows, np.stack(rel_poses, axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--num-sequences", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--cam-rate", type=int, default=100)
    parser.add_argument("--imu-rate", type=int, default=1000)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()
    build_dataset(args.output_dir, args.num_sequences, args.sequence_length, args.cam_rate, args.imu_rate, tuple(args.image_size))
