import torch
import matplotlib.pyplot as plt

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    BlendParams,
)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD PLY MESH
# ============================================================
verts, faces = load_ply("wrinkle_nose.000193.ply")
verts = verts.to(device)
faces = faces.to(device)

print("Mesh loaded")
print("Vertices:", verts.shape)
print("Faces:", faces.shape)

# ============================================================
# NORMALIZE MESH (CENTER + SCALE)  <-- SAME AS REFERENCE
# ============================================================
center = verts.mean(0)
verts = verts - center
scale = torch.max(torch.norm(verts, dim=1))
verts = verts / scale

textures = TexturesVertex(
    verts_features=torch.ones_like(verts).unsqueeze(0) * 0.7
)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# ============================================================
# CAMERA (REFERENCE-CORRECT WAY)
# ============================================================

R, T = look_at_view_transform(
    dist=5,
    elev=-31,
    azim=-48
)


cameras = FoVPerspectiveCameras(
    device=device,
    R=R,
    T=T,
    fov=13.0
)

# ============================================================
# LIGHTS (SAME AS REFERENCE)
# ============================================================
lights = PointLights(
    device=device,
    location=[[0.0, 0.0, 2.0]],
    ambient_color=((0.4, 0.4, 0.4),),
    diffuse_color=((0.4, 0.4, 0.4),),
    specular_color=((0.1, 0.1, 0.1),),
)

# ============================================================
# RENDERER
# ============================================================
raster_settings = RasterizationSettings(
    image_size=1024,
    faces_per_pixel=1,
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
        blend_params=BlendParams(background_color=(0, 0, 0))
    )
)

# ============================================================
# RENDER
# ============================================================
image = renderer(mesh)[0, ..., :3]

plt.figure(figsize=(6, 6))
plt.imshow(image.cpu().numpy())
plt.axis("off")
plt.title("PLY render using reference camera logic")
plt.show()
