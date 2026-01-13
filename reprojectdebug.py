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
# CAMERA FROM .TKA (VERIFIED CORRECT)
# ============================================================
R = np.array([
    [ 0.655773676,  0.00995448987,  0.754891908],
    [ 0.384871079, -0.864631104,   -0.322935453],
    [ 0.649488366,  0.502308633,   -0.570833513]
], dtype=np.float32)

C = np.array([
    -737.599055,
    -606.224646,
     662.894624
], dtype=np.float32)

t = -R @ C  # world → camera translation

K = np.array([
    [4647.36043, 0.0, 777.613852],
    [0.0, 4647.36043, 590.841058],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

W, H = 1600, 1200
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# ============================================================
# LOAD MESH (RAW SCANNER COORDS)
# ============================================================
verts, faces = load_ply("Untitled.ply")
verts = verts.to(device)
faces = faces.to(device)

# ============================================================
# SCALE ONLY (NO ROTATION)
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

R_torch = torch.from_numpy(R).unsqueeze(0).to(device)
t_torch = torch.from_numpy(t / 500.0).unsqueeze(0).to(device)

# ============================================================
# OpenCV → PyTorch3D (flip X ONLY)
# ============================================================
F_x = torch.tensor(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 1.0, 0.0],
     [ 0.0, 0.0, 1.0]],
    dtype=torch.float32,
    device=device
)

R_torch = torch.from_numpy(R).unsqueeze(0).to(device)
t_torch = torch.from_numpy(t / 500.0).unsqueeze(0).to(device)

R_torch = F_x @ R_torch
t_torch = (F_x @ t_torch.T).T


cameras = PerspectiveCameras(
    focal_length=((fx, fy),),
    principal_point=((cx, cy),),
    image_size=((H, W),),
    in_ndc=False,
    R=R_torch,
    T=t_torch,
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

lights = PointLights(
    device=device,
    location=torch.from_numpy(t / 500.0).unsqueeze(0).to(device)
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
plt.title("GT Render from .tka Camera (scanner world)")
plt.show()
