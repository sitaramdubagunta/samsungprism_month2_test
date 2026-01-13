import numpy as np

# Camera center from .tka
C = np.array([-737.599055, -606.224646, 662.894624], dtype=np.float64)

# distance
dist = np.linalg.norm(C)

# elevation
elev = np.arcsin(C[1] / dist)   # y / r

# azimuth
azim = np.arctan2(C[0], C[2])   # x, z  (THIS IS CRITICAL)

# convert to degrees (look_at_view_transform expects degrees by default)
elev_deg = np.degrees(elev)
azim_deg = np.degrees(azim)

print("dist =", dist)
print("elev =", elev_deg)
print("azim =", azim_deg)
