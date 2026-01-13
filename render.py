import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    BlendParams
)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CAMERA FROM .TKA (GROUND TRUTH — DO NOT TOUCH)
# ============================================================
R = np.array([
    [ 0.65577368,  0.00995449,  0.75489191],
    [ 0.38487108, -0.86463110, -0.32293545],
    [ 0.64948837,  0.50230863, -0.57083351]
], dtype=np.float32)

t = np.array([
   -10.68108671,
   -26.20796508,
 1161.97634518
], dtype=np.float32)

K = np.array([
    [4647.36043, 0.0, 777.613852],
    [0.0, 4647.36043, 590.841058],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

W, H = 1600, 1200

# ============================================================
# INTRINSICS → FOV (PyTorch3D expects FoV)
# ============================================================
fx = K[0, 0]
fov_x = 2.0 * np.arctan(W / (2.0 * fx))
fov_deg = np.degrees(fov_x)
fov_tensor = torch.tensor([fov_deg], dtype=torch.float32, device=device)

# ============================================================
# LOAD MESH (.PLY)
# ============================================================
verts, faces = load_ply("wrinkle_nose.000193.ply")
verts = verts.to(device)
faces = faces.to(device)

# ============================================================
# NORMALIZE MESH (UNIT SCALE)
# ============================================================
center = verts.mean(dim=0)
verts = verts - center
scale = torch.max(torch.norm(verts, dim=1))
verts = verts / scale

# Simple gray material (texture irrelevant for this task)
textures = TexturesVertex(
    verts_features=torch.ones_like(verts)[None] * 0.7
)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# ============================================================
# CAMERA (TEMP SCALE: t / 500 AS REQUESTED)
# ============================================================
R_torch = torch.from_numpy(R).unsqueeze(0).to(device)

t_scaled = t / 500.0   # <-- YOU ASKED FOR THIS
t_torch = torch.from_numpy(t_scaled).unsqueeze(0).to(device)

cameras = FoVPerspectiveCameras(
    R=R_torch,
    T=t_torch,
    fov=fov_tensor,
    device=device
)

# ============================================================
# RENDERER
# ============================================================
raster_settings = RasterizationSettings(
    image_size=(H, W),
    blur_radius=0.0,
    faces_per_pixel=1
)

# Light slightly in front of the camera
lights = PointLights(
    device=device,
    location=t_torch + torch.tensor([[0.0, 0.0, 1.0]], device=device)
)

shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights,
    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=shader
)

# ============================================================
# RENDER
# ============================================================
image = renderer(mesh)[0, ..., :3].cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("GT Render from .tka Camera")
plt.show()
