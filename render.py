import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
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
# CAMERA FROM .TKA (GROUND TRUTH)
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
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# ============================================================
# LOAD MESH (.PLY)
# ============================================================
verts, faces = load_ply("wrinkle_nose.000193.ply")
verts = verts.to(device)
faces = faces.to(device)

# ============================================================
# DATASET-LEVEL MESH ALIGNMENT  (CRITICAL STEP)
# ============================================================
# The mesh is NOT in the same world frame as the .tka camera.
# We apply ONE fixed rigid transform to align mesh "front"
# with the camera-facing direction.
#
# This is NOT camera correction.
# This is NOT learning-time logic.
# This is dataset canonicalization.

# 180° rotation about Y axis (front/back correction)
ALIGN_Y_180 = torch.tensor([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0]
], dtype=torch.float32, device=device)

verts = verts @ ALIGN_Y_180.T

# ============================================================
# NORMALIZE MESH (SCALE ONLY, NO ROTATION)
# ============================================================
center = verts.mean(dim=0)
verts = verts - center
scale = torch.max(torch.norm(verts, dim=1))
verts = verts / scale

textures = TexturesVertex(
    verts_features=torch.ones_like(verts)[None] * 0.7
)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# ============================================================
# PERSPECTIVE CAMERA (SCREEN SPACE, NO AXIS HACKS)
# ============================================================
cameras = PerspectiveCameras(
    focal_length=((fx, fy),),
    principal_point=((cx, cy),),
    image_size=((H, W),),
    in_ndc=False,
    R=torch.from_numpy(R).unsqueeze(0).to(device),
    T=torch.from_numpy(t / 500.0).unsqueeze(0).to(device),  # TEMP SCALE
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

# Light near camera
lights = PointLights(
    device=device,
    location=torch.from_numpy(t / 500.0).unsqueeze(0).to(device)
             + torch.tensor([[0.0, 0.0, 1.0]], device=device)
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
    )
)

# ============================================================
# RENDER
# ============================================================
image = renderer(mesh)[0, ..., :3].cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("GT Render from .tka Camera (dataset-aligned mesh)")
plt.show()
