# NOCS-Inspired 3D Face Pose Pipeline (PnP Optimized)

This pipeline implements a **Dense Coordinate Regression** strategy. By predicting the "Universal Geography" (NOCS) of a face in 2D, we use a **PnP (Perspective-n-Point)** solver to mathematically "snap" a 3D Master Mesh onto a 2D image with high precision.

---

## Phase 1: Mass Canonicalization & Dataset Gen
**Goal:** Align 100,000 unique meshes into a shared $[0, 1]$ coordinate space.

### 1.1 Global Mesh Normalization
For **each** of the 100,000 meshes:
1.  **Unit Cube Fit:** Calculate the bounding box. Scale the mesh uniformly so the **diagonal of the bounding box = 1.0**.
2.  **Centering:** Translate the mesh so the center is exactly at $[0.5, 0.5, 0.5]$.
3.  **Orientation:** Ensure all meshes share a common forward vector (e.g., $+Z$ facing the camera).
    * **Result:** Every vertex $V(x, y, z)$ now satisfies $0 \le x, y, z \le 1$.

### 1.2 The 5-Shot Render Strategy
Using **PyTorch3D**:
* **Inputs (X):** 5 Realistic RGB renders per mesh (500k total) using random HDRI lighting and varying skin textures.
* **Labels (Y):** 5 NOCS Maps. Assign $RGB = XYZ$ as vertex colors. Render with a **Flat Shader** (no shadows/specular) to get ground truth coordinates.
* **Metadata:** Store the **Camera Intrinsics ($K$)** used for each render.

---

## Phase 2: Training the Generalist CNN
**Goal:** Map 2D pixels to the normalized 3D coordinates of the "Universal Face."

* **Architecture:** **ResNet-50/EfficientNet** Encoder + **U-Net** Decoder.
* **Training Specs:**
    * **Input:** $256 \times 256 \times 3$ (RGB Image).
    * **Output:** $256 \times 256 \times 3$ (Predicted NOCS Map) + 1-channel Face Mask.
* **Loss Functions:**
    1.  **Coordinate Loss:** $Smooth L_1$ loss (masked to face region).
    2.  **Mask Loss:** Binary Cross Entropy (BCE) for background separation.

---

## Phase 3: The 6D PnP Solver (Inference)
**Goal:** Convert the predicted "Rainbow Map" into a Rotation ($R$) and Translation ($T$).

1.  **Prediction:** Feed a query image into the CNN to generate the **Predicted NOCS Map**.
2.  **Correspondence Extraction:**
    * **2D Points ($u, v$):** The pixel coordinates in the image grid.
    * **3D Points ($X, Y, Z$):** The RGB values predicted at those pixels (representing normalized 3D space).
3.  **RANSAC-PnP:**
    * Use `cv2.solvePnPRansac()` to find the pose that minimizes reprojection error.
    * **Input:** $2D$ points, $3D$ points, and Camera Intrinsics ($K$).
    * **Output:** $R$ (Rotation vector) and $T$ (Translation vector).
    * *Benefit:* RANSAC effectively ignores outliers like hair, glasses, or shadows that the CNN mis-regressed.

---

## Phase 4: Refinement & Validation
**Goal:** Achieve sub-pixel alignment accuracy.

1.  **Reprojection Check:** Project the 3D Master Mesh vertices onto the image using the solved $R$ and $T$:
    $$x' = K [R|T] X$$
2.  **Metric (ADD):** Calculate the Average Distance between projected vertices and the predicted coordinates.
3.  **Differentiable Refinement:** Use **PyTorch3D’s Differentiable Renderer** to perform gradient descent on $R$ and $T$ for 5–10 iterations, minimizing the pixel-wise difference between the predicted NOCS and the rendered mesh NOCS.

---

## Implementation Checklist
- [x] **100k Meshes:** Pre-processed and ready.
- [ ] **Step 1:** Run `normalize_mesh` to ensure $XYZ \in [0, 1]$.
- [ ] **Step 2:** Generate 500k RGB/NOCS pairs with saved $K$ matrices.
- [ ] **Step 3:** Train U-Net to regress $RGB$ coordinates.
- [ ] **Step 4:** Implement `solve_pnp_pose(nocs_pred, K)` using OpenCV.
