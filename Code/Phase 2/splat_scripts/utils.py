import math
import numpy as np


def quaternion_to_rotation_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def quaternion_from_matrix(R):
    t = np.trace(R)
    if t > 0.0:
        r = math.sqrt(1.0 + t)
        w = 0.5 * r
        x = (R[2, 1] - R[1, 2]) / (2.0 * w)
        y = (R[0, 2] - R[2, 0]) / (2.0 * w)
        z = (R[1, 0] - R[0, 1]) / (2.0 * w)
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            r = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            x = 0.5 * r
            y = (R[0, 1] + R[1, 0]) / (2.0 * x)
            z = (R[0, 2] + R[2, 0]) / (2.0 * x)
            w = (R[2, 1] - R[1, 2]) / (2.0 * x)
        elif i == 1:
            r = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            y = 0.5 * r
            x = (R[0, 1] + R[1, 0]) / (2.0 * y)
            z = (R[1, 2] + R[2, 1]) / (2.0 * y)
            w = (R[0, 2] - R[2, 0]) / (2.0 * y)
        else:
            r = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            z = 0.5 * r
            x = (R[0, 2] + R[2, 0]) / (2.0 * z)
            y = (R[1, 2] + R[2, 1]) / (2.0 * z)
            w = (R[1, 0] - R[0, 1]) / (2.0 * z)
    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quaternion_rotate(q, v):
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    v_q = np.concatenate([[0.0], v])
    return quaternion_multiply(quaternion_multiply(q, v_q), q_conj)[1:]


def pose_mul(pose_a, pose_b):
    t_a, q_a = pose_a[:3], pose_a[3:]
    t_b, q_b = pose_b[:3], pose_b[3:]
    rot_b = quaternion_rotate(q_a, t_b)
    t = t_a + rot_b
    q = quaternion_multiply(q_a, q_b)
    return np.concatenate([t, q])


def dead_reckon(rel_poses):
    traj = [np.zeros(3, dtype=np.float32)]
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for rel in rel_poses:
        t, dq = rel[:3], rel[3:]
        z = quaternion_rotate(q, t)
        traj.append(traj[-1] + z)
        q = quaternion_multiply(q, dq)
    return np.stack(traj, axis=0)


def rotation_loss(pred_q, gt_q):
    dot = np.abs(np.sum(pred_q * gt_q, axis=1))
    return np.mean(1.0 - np.clip(dot, -1.0, 1.0))
