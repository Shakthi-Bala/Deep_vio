"""
DeepVIO Data Generation Script - Unified & UV Fixed
===================================================
Run inside Blender -> Scripting workspace -> Run Script.
"""

import bpy
import sys
import os
import math
import csv
import numpy as np
from mathutils import Matrix

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# Point this to the directory containing oystersim_imuutlils.py
SCRIPT_DIR = "/home/adipat/Documents/Spring_26/CV/p4/DeepVIO"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from oystersim_imuutlils import acc_gen, gyro_gen, accel_mid_accuracy, gyro_mid_accuracy

TEXTURE_PATH = os.path.join(SCRIPT_DIR, "textures/max-bottinger-8VL7meb_k7U-unsplash.jpg")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "output")
SEQUENCE_IDX = 1       # Change this to generate new sequence folders (seq_001, seq_002, etc.)

IMU_HZ    = 1000       # IMU sample rate (Hz)
CAM_HZ    = 10        # Camera rate (Hz)
DURATION  = 1.0        # Duration of the sequence in seconds

PLANE_SIZE  = 200.0    # Floor plane size
TILE_LONG_M = 200.0     # Metres per longest side of the texture tile (lower = higher density)

RENDER_W    = 640      
RENDER_H    = 480      

# Trajectory choices: 'lissajous', 'spiral', 'figure8', 'linear'
TRAJECTORY  = 'figure8'

# ════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
CAM_STEP = IMU_HZ // CAM_HZ
N_IMU    = int(DURATION * IMU_HZ)
DT       = 1.0 / IMU_HZ
GRAVITY  = np.array([0., 0., -9.81])

def _Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
def _Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
def _Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
def R_from_rpy(roll, pitch, yaw): return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)

def R_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s

# ════════════════════════════════════════════════════════════════════════════
# 3. TRAJECTORY GENERATION
# ════════════════════════════════════════════════════════════════════════════
def make_trajectory(traj_type, n_steps, dt):
    t = np.arange(n_steps) * dt

    if traj_type == 'lissajous':
        x     = 10.0 * np.sin(2*np.pi * 0.10 * t)
        y     =  8.0 * np.sin(2*np.pi * 0.15 * t + np.pi/4)
        z     = 15.0 +  2.0 * np.sin(2*np.pi * 0.05 * t)
        roll  =  0.40 * np.sin(2*np.pi * 0.12 * t)
        pitch =  0.40 * np.cos(2*np.pi * 0.08 * t)
        yaw   =  0.30 * t
    elif traj_type == 'spiral':
        r     = np.linspace(2, 30, n_steps)
        angle = 2*np.pi * 0.10 * t
        x, y  = r * np.cos(angle), r * np.sin(angle)
        z     = 10.0 + 3.0 * np.sin(2*np.pi * 0.05 * t)
        roll  =  0.30 * np.sin(2*np.pi * 0.10 * t)
        pitch =  0.30 * np.cos(2*np.pi * 0.07 * t)
        yaw   = angle
    elif traj_type == 'figure8':
        a, f  = 15.0, 0.08
        x     =  a      * np.sin(2*np.pi * f * t)
        y     = (a/2)   * np.sin(2*np.pi * 2*f * t)
        z     = 12.0 + 2.5 * np.sin(2*np.pi * 0.04 * t)
        roll  =  0.35 * np.sin(2*np.pi * 0.12 * t)
        pitch =  0.35 * np.sin(2*np.pi * 0.09 * t)
        yaw   =  0.25 * t
    elif traj_type == 'linear':
        x     = np.linspace(-30, 30, n_steps)
        y     =  5.0 * np.sin(2*np.pi * 0.10 * t)
        z     = 10.0 + 1.5 * np.sin(2*np.pi * 0.06 * t)
        roll  =  0.20 * np.sin(2*np.pi * 0.15 * t)
        pitch =  0.30 * np.sin(2*np.pi * 0.06 * t)
        yaw   =  0.40 * np.sin(2*np.pi * 0.04 * t)
    else:
        raise ValueError("Options: lissajous, spiral, figure8, linear")

    pos = np.stack([x, y, z], axis=1)
    rpy = np.stack([roll, pitch, yaw], axis=1)
    # Clip roll & pitch to max 45 degrees as required by instructions
    rpy[:, 0] = np.clip(rpy[:, 0], -math.radians(45), math.radians(45))
    rpy[:, 1] = np.clip(rpy[:, 1], -math.radians(45), math.radians(45))
    return pos, rpy

def compute_imu_ideal(pos, rpy, dt):
    vel_w  = np.gradient(pos, dt, axis=0)
    acc_w  = np.gradient(vel_w, dt, axis=0)
    rpy_d  = np.gradient(rpy,   dt, axis=0)

    n = len(pos)
    acc_body  = np.zeros((n, 3))
    gyro_body = np.zeros((n, 3))

    for i in range(n):
        r, p, _y = rpy[i]
        dr, dp, dy = rpy_d[i]
        R_wb = R_from_rpy(r, p, _y)
        
        # Accelerometer
        acc_body[i] = R_wb.T @ (acc_w[i] - GRAVITY)
        # Gyroscope
        gyro_body[i, 0] = dr - dy * math.sin(p)
        gyro_body[i, 1] = dp * math.cos(r) + dy * math.sin(r) * math.cos(p)
        gyro_body[i, 2] = -dp * math.sin(r) + dy * math.cos(r) * math.cos(p)

    return acc_body, gyro_body

# ════════════════════════════════════════════════════════════════════════════
# 4. BLENDER SCENE SETUP
# ════════════════════════════════════════════════════════════════════════════
def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Create Floor
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = 'Floor'

    # Texture & Material Setup (with UV aspect-ratio scaling!)
    mat = bpy.data.materials.new(name="FloorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_nd   = nt.nodes.new('ShaderNodeOutputMaterial')
    emis_nd  = nt.nodes.new('ShaderNodeEmission')
    tex_nd   = nt.nodes.new('ShaderNodeTexImage')
    map_nd   = nt.nodes.new('ShaderNodeMapping')
    coord_nd = nt.nodes.new('ShaderNodeTexCoord')
    
    tex_nd.extension  = 'REPEAT'
    map_nd.vector_type = 'POINT'

    try:
        img = bpy.data.images.load(TEXTURE_PATH)
        tex_nd.image = img
        
        # Automatic UV scaling magic
        W, H = img.size
        long_px, short_px = max(W, H), min(W, H)
        tile_short = TILE_LONG_M * short_px / long_px
        tile_w = TILE_LONG_M if W >= H else tile_short
        tile_h = tile_short  if W >= H else TILE_LONG_M
        
        su = PLANE_SIZE / tile_w
        sv = PLANE_SIZE / tile_h
        map_nd.inputs['Scale'].default_value    = (su, sv, 1.0)
    except Exception as e:
        print(f"Failed to load image for UV mapping: {e}")

    # Link nodes
    nt.links.new(coord_nd.outputs['UV'],      map_nd.inputs['Vector'])
    nt.links.new(map_nd.outputs['Vector'],    tex_nd.inputs['Vector'])
    nt.links.new(tex_nd.outputs['Color'],     emis_nd.inputs['Color'])
    nt.links.new(emis_nd.outputs['Emission'], out_nd.inputs['Surface'])
    floor.data.materials.append(mat)

    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 10))
    cam_obj = bpy.context.active_object
    cam_obj.name = 'DroneCamera'
    bpy.context.scene.camera = cam_obj

    cam = cam_obj.data
    cam.type         = 'PERSP'
    cam.sensor_fit   = 'HORIZONTAL'
    cam.lens         =20.0  # Wide angle lens so more texture feature tracking matches are available!

    # EEVEE setup
    sc = bpy.context.scene
    
    try:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        sc.render.engine = 'BLENDER_EEVEE'
        
    sc.render.resolution_x = RENDER_W
    sc.render.resolution_y = RENDER_H
    sc.render.image_settings.file_format = 'PNG'

    return cam_obj

def set_camera_pose(cam_obj, pos_np, R_wb):
    mat = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            mat[i][j] = float(R_wb[i][j])
    mat[0][3], mat[1][3], mat[2][3] = float(pos_np[0]), float(pos_np[1]), float(pos_np[2])
    cam_obj.matrix_world = mat

# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════
print(f"\n--- Generating Data: Trajectory={TRAJECTORY}, Duration={DURATION}s ---")
cam_obj = setup_scene()

pos, rpy = make_trajectory(TRAJECTORY, N_IMU, DT)
acc_ideal, gyro_ideal = compute_imu_ideal(pos, rpy, DT)

print("Applying OysterSim noise...")
acc_noisy  = acc_gen(IMU_HZ, acc_ideal, accel_mid_accuracy)
gyro_noisy = gyro_gen(IMU_HZ, gyro_ideal, gyro_mid_accuracy)

timestamps = np.arange(N_IMU) * DT

seq_dir = os.path.join(OUTPUT_DIR, f"seq_{SEQUENCE_IDX:03d}")
img_dir = os.path.join(seq_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# Write IMU data
imu_csv = os.path.join(seq_dir, "imu.csv")
with open(imu_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])
    for i in range(N_IMU):
        w.writerow([f"{timestamps[i]:.6f}",
                    *[f"{v:.8f}" for v in gyro_noisy[i]],
                    *[f"{v:.8f}" for v in acc_noisy[i]]])

print("Rendering camera frames and saving Ground Truth poses...")
cam_frames = list(range(0, N_IMU, CAM_STEP))
gt_rows, rel_rows = [], []
prev_R = prev_p = None

for frame_idx, step_i in enumerate(cam_frames):
    t = timestamps[step_i]
    p = pos[step_i]
    r, pv, yv = rpy[step_i]
    R_wb = R_from_rpy(r, pv, yv)
    qx, qy, qz, qw = R_to_quat(R_wb)

    gt_rows.append([f"{t:.6f}", f"{p[0]:.8f}", f"{p[1]:.8f}", f"{p[2]:.8f}",
                     f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}", f"{qw:.8f}"])

    if prev_R is not None:
        R_rel = prev_R.T @ R_wb
        t_rel = prev_R.T @ (p - prev_p)
        rqx, rqy, rqz, rqw = R_to_quat(R_rel)
        rel_rows.append([f"{t:.6f}",
                         f"{t_rel[0]:.8f}", f"{t_rel[1]:.8f}", f"{t_rel[2]:.8f}",
                         f"{rqx:.8f}", f"{rqy:.8f}", f"{rqz:.8f}", f"{rqw:.8f}"])
    prev_R, prev_p = R_wb.copy(), p.copy()

    set_camera_pose(cam_obj, p, R_wb)
    bpy.context.view_layer.update()

    bpy.context.scene.render.filepath = os.path.join(img_dir, f"{frame_idx:05d}.png")
    bpy.ops.render.render(write_still=True)

with open(os.path.join(seq_dir, "groundtruth.csv"), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    w.writerows(gt_rows)

with open(os.path.join(seq_dir, "relative_poses.csv"), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'dtx', 'dty', 'dtz', 'dqx', 'dqy', 'dqz', 'dqw'])
    w.writerows(rel_rows)

print(f"DONE! Saved to {seq_dir}")