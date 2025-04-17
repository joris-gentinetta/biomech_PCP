import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------
# 1) Load raw CyberGlove angles
# ------------------------------------------------------
mat_path = r"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/pretraining/CyberGlove2/s_1_angles/s_1_angles/S64_E1_A1.mat"
data     = scipy.io.loadmat(mat_path)
angles_raw = data['angles']   # shape (n_frames, n_channels)

# ------------------------------------------------------
# 2) Zero‑and‑stretch each channel into [0°,90°]
# ------------------------------------------------------
# mins   = angles_raw.min(axis=0)
# n_col  = angles_raw.shape[1]
# mins   = np.zeros(n_col)
mins   = angles_raw.min(axis=0)
maxs   = angles_raw.max(axis=0)
ranges = maxs - mins + 1e-8

n_channels = angles_raw.shape[1]
multipliers = np.full(n_channels, 110.0)
for idx in [4, 7, 9, 13]:
    multipliers[idx] = 110.0
for idx in [6, 8, 11, 15]:
    multipliers[idx] = 110.0
for idx in [16, 17, 18, 19]:
    multipliers[idx] = 80.0


angles_deg = (angles_raw - mins[np.newaxis, :]) / ranges[np.newaxis, :] * multipliers[np.newaxis, :]

# ------------------------------------------------------
# 3) (Optional) Flip “reversed” channels here
#    If channel i *decreases* when you flex, add i to this list:
# ------------------------------------------------------
reversed_channels = []  # e.g. [17,18,19] if those go backwards
if reversed_channels:
    angles_deg[:,reversed_channels] = 90.0 - angles_deg[:,reversed_channels]

# ------------------------------------------------------
# 4) (Optional) Smooth jitter: Savitzky–Golay filter
#    window_length must be odd; here 5 frames (~80ms @60Hz)
# ------------------------------------------------------
angles_smooth = savgol_filter(angles_deg, window_length=5, polyorder=2, axis=0)

# choose which to animate:
angles_for_anim = angles_smooth  # or switch to angles_deg for no smoothing

# ------------------------------------------------------
# 5) Animation setup (your original kinematics & plotting)
# ------------------------------------------------------
thumb_indices  = [0, 1, 2]
index_indices  = [4, 6, 16]
middle_indices = [7, 8, 17]
ring_indices   = [9, 11, 18]
pinky_indices  = [13,15,19]

base_pts = {
    'thumb':  np.array([0, 0, 0]),
    'index':  np.array([2, 0, 0]),
    'middle': np.array([4, 0, 0]),
    'ring':   np.array([6, 0, 0]),
    'pinky':  np.array([8, 0, 0]),
}

fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_zlim(-10,10)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

finger_lines = {
    finger: ax.plot([], [], [], 'o-', linewidth=3, label=finger)[0]
    for finger in base_pts
}
ax.legend()

def finger_kinematics(base, angles_deg, lengths=[4,3,2]):
    angles_rad = np.radians(angles_deg)
    pts = [base.copy()]
    direction = np.array([0,0,1])
    for angle, L in zip(angles_rad, lengths):
        Rx = np.array([[1,0,0],
                       [0,np.cos(angle),-np.sin(angle)],
                       [0,np.sin(angle), np.cos(angle)]])
        direction = Rx @ direction
        pts.append(pts[-1] + direction * L)
    return np.array(pts)

def thumb_kinematics(base, angles_deg, lengths=[3,2,1.5]):
    cmc_abd, mcp_flex, ip_flex = np.radians(angles_deg)
    pts = [base.copy()]
    Rz = np.array([[ np.cos(cmc_abd), -np.sin(cmc_abd), 0],
                   [ np.sin(cmc_abd),  np.cos(cmc_abd), 0],
                   [ 0,                0,               1]])
    direction = Rz @ np.array([0,1,0])
    Rx_mcp = np.array([[1,0,0],
                       [0,np.cos(mcp_flex), -np.sin(mcp_flex)],
                       [0,np.sin(mcp_flex),  np.cos(mcp_flex)]])
    direction = Rx_mcp @ direction
    pt1 = pts[-1] + direction * lengths[0]; pts.append(pt1)
    Rx_ip = np.array([[1,0,0],
                      [0,np.cos(ip_flex), -np.sin(ip_flex)],
                      [0,np.sin(ip_flex),  np.cos(ip_flex)]])
    direction = Rx_ip @ direction
    pt2 = pts[-1] + direction * lengths[1]; pts.append(pt2)
    return np.array(pts)

def update(frame):
    fa = angles_for_anim[frame]
    # thumb
    th_pts = thumb_kinematics(base_pts['thumb'], fa[thumb_indices])
    line  = finger_lines['thumb']
    line.set_data(th_pts[:,0], th_pts[:,1]); line.set_3d_properties(th_pts[:,2])
    # other fingers
    for name, idxs in zip(['index','middle','ring','pinky'],
                          [index_indices, middle_indices, ring_indices, pinky_indices]):
        pts = finger_kinematics(base_pts[name], fa[idxs])
        ln = finger_lines[name]
        ln.set_data(pts[:,0], pts[:,1]); ln.set_3d_properties(pts[:,2])
    return list(finger_lines.values())

# ------------------------------------------------------
# 6) Fire off the animation
#    - step=20 skips ahead 20 frames per draw (speeds it up)
#    - interval=30 sets ~33fps
# ------------------------------------------------------
ani = FuncAnimation(fig, update,
                    frames=np.arange(0, angles_for_anim.shape[0], 200),
                    interval=50, blit=False)
plt.show()
