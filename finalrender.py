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
    BlendParams,
)
from pytorch3d.renderer import look_at_view_transform

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CAMERA (derived from .tka via sanity-checked conversion)
# ============================================================
dist = 1
elev = -31.437
azim = -48.053

# Intrinsics → FOV
W, H = 1600, 1200
fx = 4647.36043
fov_x = 2.0 * np.arctan(W / (2.0 * fx))
fov = torch.tensor([float(np.degrees(fov_x))], device=device)

# ============================================================
# LOAD MESH (NO NORMALIZATION — REAL SCALE)
# ============================================================
verts, faces = load_ply("Untitled.ply")
verts = verts.to(device)
faces = faces.to(device)

# Optional: center only (NO scaling)
center = verts.mean(dim=0)
verts = verts - center

textures = TexturesVertex(
    verts_features=torch.ones_like(verts)[None] * 0.7
)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# ============================================================
# CAMERA (LOOK-AT)
# ============================================================
R, T = look_at_view_transform(
    dist=dist,
    elev=elev,
    azim=azim,
    device=device
)

cameras = FoVPerspectiveCameras(
    R=R,
    T=T,
    fov=fov,
    device=device
)

# ============================================================
# LIGHT AT CAMERA CENTER (CRITICAL)
# ============================================================
# Camera center C = -R^T T
C_cam = (-R.transpose(1, 2) @ T.unsqueeze(-1)).squeeze(-1)

lights = PointLights(
    device=device,
    location=C_cam
)

# ============================================================
# RENDERER
# ============================================================
raster_settings = RasterizationSettings(
    image_size=(H, W),
    blur_radius=0.0,
    faces_per_pixel=1
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
plt.title("DEBUG: look_at_view_transform (REAL SCALE)")
plt.show()
