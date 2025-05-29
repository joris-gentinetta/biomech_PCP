import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def load_and_preprocess(mat_path):
    data      = scipy.io.loadmat(mat_path)
    angles_raw = data['angles']  # (n_frames, n_channels)
    # normalize 0–90°
    mins   = angles_raw.min(axis=0)
    maxs   = angles_raw.max(axis=0)
    ranges = maxs - mins + 1e-8
    multipliers = np.full(angles_raw.shape[1], 110.0)
    for i in [4,7,9,13,6,8,11,15]:
        multipliers[i] = 110.0
    for i in [16,17,18,19]:
        multipliers[i] = 80.0
    for i in [1]:
        multipliers[i] = 160.0
    angles_deg = (angles_raw - mins) / ranges * multipliers
    # smooth
    angles_smooth = savgol_filter(angles_deg, window_length=5, polyorder=2, axis=0)
    return angles_smooth

def finger_kin(base, ang, lengths=[4,3,2]):
    θ = np.radians(ang)
    pts = [base.copy()]; d = np.array([0,0,1])
    for th,L in zip(θ,lengths):
        Rx = np.array([[1,0,0],
                       [0,np.cos(th),-np.sin(th)],
                       [0,np.sin(th), np.cos(th)]])
        d = Rx @ d
        pts.append(pts[-1] + d*L)
    return np.array(pts)

def thumb_kin(base, ang, lengths=[6,3,2]):
    # ang = [cmc_flex, cmc_abd, mcp_flex, ip_flex]
    cmc_f, cmc_a, mcp_f, ip_f = np.radians(ang)
    pts = [base.copy()]

    # CMC segment (hidden)
    d0 = np.array([1.0,0.0,0.0])
    Rx0 = np.array([[1,0,0],
                    [0,np.cos(cmc_f),-np.sin(cmc_f)],
                    [0,np.sin(cmc_f), np.cos(cmc_f)]])
    Ry0 = np.array([[ np.cos(cmc_a),0, np.sin(cmc_a)],
                    [             0,1,              0],
                    [-np.sin(cmc_a),0, np.cos(cmc_a)]])
    d1 = Ry0 @ (Rx0 @ d0)
    pts.append(pts[-1] + d1*lengths[0])

    # MCP flexion
    Rx1 = np.array([[1,0,0],
                    [0,np.cos(mcp_f),-np.sin(mcp_f)],
                    [0,np.sin(mcp_f), np.cos(mcp_f)]])
    d2 = Rx1 @ d1
    pts.append(pts[-1] + d2*lengths[1])

    # IP flexion
    Rx2 = np.array([[1,0,0],
                    [0,np.cos(ip_f),-np.sin(ip_f)],
                    [0,np.sin(ip_f), np.cos(ip_f)]])
    d3 = Rx2 @ d2
    pts.append(pts[-1] + d3*lengths[2])

    return np.array(pts)  # shape=(4,3)

def setup_plot():
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-10,10), ylim=(-10,10), zlim=(-10,10),
           xlabel='X', ylabel='Y', zlabel='Z')
    return fig, ax

def main():
    # file and data
    mat_path = r"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/pretraining/CyberGlove2/s_1_angles/s_1_angles/S64_E1_A1.mat"
    angles = load_and_preprocess(mat_path)

    # indices
    thumb_idx  = [0,1,2,3]
    index_idx  = [4,6,16]
    middle_idx = [7,8,17]
    ring_idx   = [9,11,18]
    pinky_idx  = [13,15,19]

    # base points
    base_pts = {
        'thumb':  np.array([2.5, 5.0, -1.0]),  # CMC joint location
        'index':  np.array([2.0,  0.0, 0.0]),
        'middle': np.array([4.0,  0.0, 0.0]),
        'ring':   np.array([6.0,  0.0, 0.0]),
        'pinky':  np.array([8.0,  0.0, 0.0]),
    }

    # plot
    fig, ax = setup_plot()
    lines = {f: ax.plot([],[],[], 'o-', lw=3, label=f)[0] for f in base_pts}
    lines['thumb_dbg'] = ax.plot([],[],[], '--', color='gray', lw=1, label='thumb_dbg')[0]
    ax.legend()

    alpha = 110   # rotate about X
    beta  = -60  # rotate about Y
    gamma = 110  # rotate about Z

    # convert to radians
    a = np.radians(alpha)
    b = np.radians(beta)
    g = np.radians(gamma)

    # build the three rotation matrices
    Rx = np.array([[ 1,      0,       0   ],
                [ 0,  np.cos(a), -np.sin(a)],
                [ 0,  np.sin(a),  np.cos(a)]])
    Ry = np.array([[ np.cos(b), 0, np.sin(b)],
                [       0,    1,       0   ],
                [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[ np.cos(g), -np.sin(g), 0],
                [ np.sin(g),  np.cos(g), 0],
                [      0,           0,    1]])
    
    R_global = Rz @ Ry @ Rx

    def update(frame):
        a = angles[frame]
        # thumb → compute full chain, but only draw segments 1→3
        full = thumb_kin(base_pts['thumb'], a[thumb_idx], lengths=[5, 4, 3])
        base = full[0]
        full = (R_global @  (full - base).T).T + base
        dbg = full[:2]
        lines['thumb_dbg'].set_data(dbg[:,0], dbg[:,1])
        lines['thumb_dbg'].set_3d_properties(dbg[:,2])
        seg  = full[1:]  # MCP→..., tip
        ln  = lines['thumb']
        ln.set_data(seg[:,0], seg[:,1])
        ln.set_3d_properties(seg[:,2])

        # other fingers
        for name, idx in zip(['index','middle','ring','pinky'],
                             [index_idx, middle_idx, ring_idx, pinky_idx]):
            pts = finger_kin(base_pts[name], a[idx])
            ln  = lines[name]
            ln.set_data(pts[:,0], pts[:,1])
            ln.set_3d_properties(pts[:,2])

        return list(lines.values())

    ani = FuncAnimation(fig, update,
                        frames=np.arange(0, angles.shape[0], 400),
                        interval=50, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
