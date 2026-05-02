# Understanding S-MSCKF: Visual-Inertial Odometry from the Ground Up

## The Big Picture: What Are We Solving?

You have a drone flying through a room. It has:
- **Two cameras** (stereo) taking pictures at 20 Hz
- **An IMU** (accelerometer + gyroscope) measuring at 200 Hz
- **No GPS**

**Goal:** Estimate where the drone is (position + orientation) at every moment in time.

**Why is this hard?**
- Camera alone: can't measure scale (is that a small box nearby or a big box far away?)
- IMU alone: integrating acceleration gives position, but bias errors make it drift to garbage within seconds
- Together: cameras fix IMU drift, IMU fills gaps between frames and provides scale

---

## The Two Fundamental Problems

### Problem 1: IMU Integration (Prediction)
The IMU gives you angular velocity ω and linear acceleration a at 200 Hz. Between camera frames (every 50ms = 10 IMU readings), you integrate these to predict where you are:

```
orientation += ω × dt          (integrate gyro)
velocity += R^T × a + g × dt   (integrate accelerometer, add gravity)  
position += velocity × dt       (integrate velocity)
```

**But:** The gyro has a slowly-changing bias b_g. If your gyro says ω = 0.01 rad/s but the true value is 0, after 10 seconds you've accumulated 0.1 rad (5.7°) of error. Same problem with accelerometer bias.

**Solution:** Estimate the biases as part of your state, and let the camera measurements correct them.

### Problem 2: Visual Measurement Update (Correction)
When a camera frame arrives, you:
1. Detect features (corners) in the image
2. Track them across frames
3. Use their positions to compute "how wrong was my prediction?"
4. Correct the state

**The key insight of MSCKF:** Don't put feature 3D positions in your state vector. Instead, keep a sliding window of past camera poses, and when a feature disappears, use ALL its observations across multiple frames to compute one correction.

---

## The State Vector: What Are We Estimating?

```python
# msckf.py: IMUState class
class IMUState:
    orientation     # quaternion [x,y,z,w] — rotation from world to IMU
    position        # 3D position of IMU in world frame
    velocity        # 3D velocity in world frame
    gyro_bias       # 3D gyroscope bias (slowly drifts)
    acc_bias        # 3D accelerometer bias
    R_imu_cam0      # rotation from IMU to camera (extrinsic calibration)
    t_cam0_imu      # translation from camera to IMU
```

That's **21 dimensions** for the error state:
- 3 orientation error (small angle approximation)
- 3 gyro bias
- 3 velocity
- 3 accel bias
- 3 position
- 3 extrinsic rotation
- 3 extrinsic translation

Plus a **sliding window** of camera poses (max 20), each with 6 DOF (3 orientation + 3 position). Total: **21 + 6N** dimensions.

**Why is this efficient?** EKF-SLAM would have 21 + 3L dimensions where L = number of landmarks (could be thousands). MSCKF caps at 21 + 120 = 141.

---

## Step-by-Step Pipeline

### Step 0: Gravity Initialization

```python
# msckf.py: initialize_gravity_and_bias()
# Wait for 200 IMU readings while the drone is stationary
sum_angular_vel = 0    # average should ≈ gyro bias
sum_linear_acc = 0     # average should ≈ gravity (since stationary, a = -g)

gyro_bias = average(angular_velocities)     # set initial gyro bias
gravity_imu = average(linear_accelerations)  # gravity direction in IMU frame

# Set initial orientation so that IMU z-axis aligns with gravity
q = from_two_vectors([0,0,1], normalize(gravity_imu))
```

**Intuition:** When stationary, the accelerometer measures -g (gravity pointing down). The gyro output is pure bias. Average both to initialize.

---

### Step 1: IMU Propagation (between camera frames)

For each IMU measurement (every 5ms), we do two things:

#### 1a. Propagate the actual state (Runge-Kutta)

```python
# msckf.py: predict_new_state()
# Remove bias from measurements
gyro = measured_gyro - gyro_bias    # true angular velocity
acc = measured_acc - acc_bias        # true acceleration

# 4th-order Runge-Kutta for orientation
# The quaternion derivative: dq/dt = 0.5 * Omega(ω) * q
Omega = [[-skew(gyro), gyro],      # 4x4 matrix
         [-gyro^T,      0  ]]

dq_dt = (cos(|ω|dt/2)*I + sin(|ω|dt/2)/|ω| * Omega) @ q

# RK4 for velocity and position
k1_v = R^T @ acc + g              # acceleration in world frame
k1_p = v                           # velocity
k2_v = R_half^T @ acc + g         # at midpoint
k2_p = v + k1_v * dt/2
k3_v = R_half^T @ acc + g
k3_p = v + k2_v * dt/2
k4_v = R_full^T @ acc + g         # at endpoint
k4_p = v + k3_v * dt

v_new = v + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
p_new = p + dt/6 * (k1_p + 2*k2_p + 2*k3_p + k4_p)
```

**Why RK4?** Simple Euler integration (x += dx*dt) accumulates numerical error fast. RK4 evaluates the derivative at multiple points within each timestep for much higher accuracy. Critical when integrating 200 Hz IMU data over 50ms gaps.

#### 1b. Propagate the covariance (how uncertain are we?)

```python
# msckf.py: process_model()
# Linearized error dynamics: dẋ = F*x̃ + G*n
# F is the Jacobian of the dynamics w.r.t. the state error
R = to_rotation(orientation)

F = zeros(21, 21)
F[0:3, 0:3] = -skew(gyro)      # orientation error depends on gyro
F[0:3, 3:6] = -I               # orientation affected by gyro bias
F[6:9, 0:3] = -R^T @ skew(acc) # velocity error depends on orientation error
F[6:9, 9:12] = -R^T            # velocity affected by accel bias
F[12:15, 6:9] = I              # position error grows with velocity error

# Discrete transition matrix (matrix exponential, approximated)
Phi = I + F*dt + 0.5*(F*dt)^2 + (1/6)*(F*dt)^3

# Covariance propagation
Q = Phi @ G @ noise_cov @ G^T @ Phi^T * dt
P_new = Phi @ P @ Phi^T + Q
```

**What F tells us:** Each row says "how does the error in this state affect the error derivatives?"
- Row 0-2 (orientation): errors grow from gyro noise and gyro bias
- Row 6-8 (velocity): errors come from orientation error (wrong rotation of acceleration) and accel bias
- Row 12-14 (position): errors grow from velocity error

**The covariance P grows during propagation** — we become less certain of our state between camera frames.

---

### Step 2: State Augmentation (new camera frame arrives)

When a new image arrives, add the camera pose to the sliding window:

```python
# msckf.py: state_augmentation()
# Camera pose = IMU pose transformed by extrinsic calibration
R_world_cam = R_imu_cam @ R_world_imu
p_cam = p_imu + R_world_imu^T @ t_cam_imu

# Augment the covariance matrix
# J is the Jacobian: how does camera pose relate to IMU state?
J = zeros(6, 21)
J[0:3, 0:3] = R_imu_cam          # cam orientation ← IMU orientation
J[0:3, 15:18] = I                 # cam orientation ← extrinsic rotation
J[3:6, 0:3] = skew(R^T @ t)      # cam position ← IMU orientation
J[3:6, 12:15] = I                 # cam position ← IMU position
J[3:6, 18:21] = R^T              # cam position ← extrinsic translation

# Expand covariance: new rows/cols for camera state
P_new = [P,        P @ J^T    ]
        [J @ P,    J @ P @ J^T]
```

**Intuition:** The new camera pose is just a copy of the current IMU pose (transformed by calibration), so its uncertainty is derived from the current IMU uncertainty through the Jacobian J.

---

### Step 3: Feature Tracking (Image Processing)

```python
# image.py: stereo_callback()
# For each frame:
# 1. Track existing features with optical flow
prev_pts → curr_pts using Lucas-Kanade (KLT)

# 2. Match to right camera (stereo)  
left_pts → right_pts using horizontal optical flow

# 3. Detect new features in empty grid cells
# Image divided into 4×5 grid, detect FAST corners where tracking is sparse

# 4. Outlier rejection
# RANSAC on the essential matrix to remove bad tracks
```

**Why grid-based detection?** Without it, features cluster in textured areas (e.g., a poster on the wall) while large regions have no features. The grid ensures spatial coverage.

**Why stereo?** The horizontal disparity between left and right gives depth:
```
depth = baseline * focal_length / disparity
```
This makes metric scale directly observable — a huge advantage over monocular VIO.

---

### Step 4: Measurement Update (The Core of MSCKF)

When a feature is lost (no longer tracked), we process all its observations:

#### 4a. Triangulate the feature

```python
# feature.py: initialize_position()
# Given observations from multiple camera poses, estimate 3D position
# Uses inverse-depth parameterization + Levenberg-Marquardt optimization

# Initial guess from two-view triangulation (first and last observation)
# Then refine with all views using least-squares
```

#### 4b. Compute the Jacobian for each observation

```python
# msckf.py: measurement_jacobian()
# For feature at 3D position p_w, observed in camera i:

# Project to camera frame
p_c0 = R_w_c @ (p_w - t_c_w)     # world → left camera
p_c1 = R_cam01 @ p_c0 + t_cam01  # left → right camera

# Predicted measurement (normalized image coordinates)
z_hat = [p_c0.x/p_c0.z, p_c0.y/p_c0.z,    # left camera
         p_c1.x/p_c1.z, p_c1.y/p_c1.z]    # right camera

# Residual
r = z_measured - z_hat

# Jacobians
H_x = dz/d(camera_state)  # (4×6) how measurement changes with cam pose
H_f = dz/d(feature_pos)   # (4×3) how measurement changes with feature position
```

#### 4c. Nullspace projection (THE key trick)

```python
# msckf.py: feature_jacobian()
# Stack all observations of this feature:
# H_fj has shape (4M × 3) where M = number of observations
# 
# We DON'T want feature position in our state!
# Project into the left nullspace of H_f:
U, S, V = svd(H_fj)
A = U[:, 3:]          # columns of U beyond the first 3

H_projected = A^T @ H_xj    # eliminates feature position
r_projected = A^T @ r_j     # residual in nullspace
```

**Why this works:** H_f tells us "how does the residual depend on the feature position." By projecting into the nullspace of H_f, we keep ONLY the part of the residual that depends on the camera poses (our state), not on where the feature is. This is what lets MSCKF avoid putting features in the state.

**The math:** If H = [H_x | H_f], the full residual is r = H_x * δx + H_f * δp_f + noise. By left-multiplying by A^T (nullspace of H_f), A^T * H_f = 0, so: A^T * r = A^T * H_x * δx + noise. Feature position eliminated.

#### 4d. Gating test

```python
# msckf.py: gating_test()
# Chi-squared test: is this measurement consistent with our state?
gamma = r^T @ (H @ P @ H^T + R)^{-1} @ r

if gamma < chi2_threshold(dof, 0.95):
    accept    # consistent measurement, use it
else:
    reject    # outlier, skip it
```

#### 4e. EKF Update (Kalman Gain)

```python
# msckf.py: measurement_update()
# Optional: QR decomposition to reduce dimensions
if num_rows > num_cols:
    Q, R = qr(H)
    H = R           # smaller matrix
    r = Q^T @ r     # projected residual

# Standard Kalman update
S = H @ P @ H^T + σ²_obs * I    # innovation covariance
K = P @ H^T @ inv(S)             # Kalman gain
δx = K @ r                        # state correction

# Apply correction
orientation *= small_angle_quaternion(δx[0:3])
gyro_bias += δx[3:6]
velocity += δx[6:9]
acc_bias += δx[9:12]
position += δx[12:15]
# ... also update camera states

# Update covariance (Joseph form for stability)
P = (I - K @ H) @ P
```

**The Kalman gain K balances:**
- When P is large (uncertain state), K is large → trust the measurement more
- When σ²_obs is large (noisy measurement), K is small → trust the state more

---

### Step 5: Sliding Window Management

```python
# msckf.py: prune_cam_state_buffer()
# If window exceeds 20 camera states:
# 1. Find redundant states (similar pose to neighbors)
# 2. Process any features observed at those states
# 3. Remove the camera state from the window
# 4. Delete corresponding rows/cols from covariance P
```

**Why prune?** The state dimension is 21 + 6N. Without pruning, N grows forever. With N_max = 20, we cap at 141 dimensions.

---

## The Observability Trick

The code has mysterious `orientation_null`, `position_null`, `velocity_null` variables. What are they?

**The problem:** In VIO, 4 things are truly unobservable:
1. Global yaw (which way is "north")
2. Global x position
3. Global y position  
4. Global z position

A naive EKF can accidentally "observe" these through linearization errors, making the filter overconfident (thinks it knows absolute position when it actually can't). This causes inconsistency and divergence.

**The fix (from Hesch et al. 2014):** Modify the transition matrix Φ so that its nullspace matches the true unobservable directions. The `_null` variables store the linearization point used for this correction:

```python
# msckf.py: process_model()
# Modify Phi to maintain correct observability
u = R^T @ gravity
A1 = Phi[6:9, 0:3]   # velocity-orientation block
w = skew(v_null - v) @ gravity
Phi[6:9, 0:3] = A1 - (A1 @ u - w)[:,None] * u^T / (u @ u)
```

---

## Data Flow Summary

```
IMU @ 200Hz ──→ buffer ──→ batch_imu_processing() ──→ predict state + propagate P
                                    │
Camera @ 20Hz ──→ feature detection/tracking
                        │
                        ▼
                state_augmentation() ──→ add camera pose to window, expand P
                        │
                        ▼
                add_feature_observations() ──→ store measurements per feature
                        │
                        ▼
                remove_lost_features() ──→ for each lost feature:
                  │                          triangulate → Jacobians → nullspace
                  │                          → gating test → Kalman update
                  ▼
                prune_cam_state_buffer() ──→ remove old camera states
                        │
                        ▼
                    publish pose
```

---

## Key Equations Cheat Sheet

| What | Equation | Code location |
|------|----------|---------------|
| Quaternion → rotation | R = (2w²-1)I - 2w[ω]× + 2ωω^T | `utils.py: to_rotation()` |
| IMU dynamics | dq/dt = ½Ω(ω)q, dv/dt = R^T a + g | `msckf.py: predict_new_state()` |
| Error state propagation | P = ΦPΦ^T + Q | `msckf.py: process_model()` |
| State augmentation | J relates camera to IMU state | `msckf.py: state_augmentation()` |
| Nullspace projection | A^T H_f = 0, so A^T r depends only on x | `msckf.py: feature_jacobian()` |
| Kalman gain | K = PH^T(HPH^T + R)^{-1} | `msckf.py: measurement_update()` |
| State correction | x ← x + Kr, P ← (I-KH)P | `msckf.py: measurement_update()` |

---

## Common Confusions

**Q: Why JPL quaternion convention?**  
A: Historical convention in NASA/aerospace. q = [vector, scalar] and represents rotation from global→body. The Trawny tech report defines all the algebra. Your code uses `[x, y, z, w]` where `w` is scalar.

**Q: Why not just use feature positions in the state?**  
A: You could (that's EKF-SLAM). But with 200 features, your state would be 21 + 600 = 621 dimensions. The covariance P would be 621×621. Matrix inversion is O(n³). MSCKF keeps it at ~141 dimensions regardless of feature count.

**Q: Why does position drift but orientation doesn't?**  
A: Orientation has a natural "anchor" — gravity. The accelerometer always measures g, which constrains roll and pitch. Only yaw drifts. Position has no anchor without GPS, so it drifts in all 3 axes.

**Q: What if there are no features (blank wall)?**  
A: Pure IMU dead-reckoning. The covariance grows rapidly, but the state doesn't jump. When features reappear, the filter corrects. This is why VIO is more robust than pure visual odometry.
