# NOCS-Inspired 3D Face Pose Pipeline (High-Diversity Scale)

This pipeline leverages a dataset of **100,000 unique meshes** with **5 renders per mesh** (500k total pairs). The goal is to train a CNN that understands the "Universal Geography" of a human face to regress 3D coordinates from any 2D image.

---

## Phase 1: Mass Canonicalization & Dataset Gen
**Goal:** Ensure all 100k meshes share the exact same coordinate space so the CNN learns a consistent "Rainbow Map."

### 1.1 Global Mesh Normalization
For **each** of the 100,000 meshes:
1. **Unit Cube Fit:** Calculate the bounding box. Scale the mesh so the longest side (usually height or depth) equals **1.0**.
2. **Centering:** Translate the mesh so the "Mean Point" or Bounding Box Center is exactly at $[0.5, 0.5, 0.5]$.
3. **Consistency Check:** Ensure all meshes are facing the same direction (e.g., +Z is "forward") before rendering.

### 1.2 The 5-Shot Render Strategy
Using **PyTorch3D** or **BlenderProc**:
* **Inputs (X):** 5 Realistic RGB renders per mesh. Use random HDRI lighting and varying skin textures to prevent the AI from "taking shortcuts" based on color.
* **Labels (Y):** 5 NOCS Maps. Assign $RGB = XYZ$ as vertex colors. Render with a **Flat Shader** (no shadows/specular) to get the ground truth coordinates.
* **Camera:** For the 5 shots, sample positions from a **Upper Hemisphere** to ensure coverage of the face front and profiles.

---

## Phase 2: Training the Generalist CNN
**Goal:** Map 2D pixels to the 100k-mesh-average 3D coordinate.

* **Architecture:** **ResNet-50 Encoder** with a **U-Net Decoder** (to maintain spatial resolution for the Umeyama solver).
* **Training Specs:**
    * **Input:** $256 \times 256 \times 3$ (RGB Image).
    * **Output:** $256 \times 256 \times 3$ (NOCS Map) + 1-channel Binary Mask.
* **Loss Functions:**
    1. **Coordinate Loss:** $\text{Smooth } L_1$ loss between predicted $RGB$ and ground truth NOCS.
    2. **Mask Loss:** Binary Cross Entropy (BCE) to help the AI distinguish "Face" from "Background."
* **Data Strategy:** Use a `WeightedRandomSampler` if certain ethnicities or face shapes are underrepresented in your 100k pool.

---

## Phase 3: The 6D Inference Solver
**Goal:** Turn the "Rainbow Guess" into a rotation matrix $R$ and translation $T$.

1. **Prediction:** Run a "Query Image" (a real person's face) through the CNN.
2. **Point Selection:** Mask the output and pick the top 500–1000 pixels with the highest confidence.
    * **Set A (2D):** The $(u, v)$ pixel coordinates.
    * **Set B (3D):** The $(x, y, z)$ values predicted in those pixels.
3. **Umeyama Alignment (Procrustes):**
    * Align your **Reference Master Mesh** to **Set B**.
    * Use Singular Value Decomposition (SVD) to find the optimal $R$, $T$, and Scale $s$.
    * $$\min_{R, T, s} \sum \| B - (sRA + T) \|^2$$

---

## Phase 4: Refinement & Validation
**Goal:** Pixel-perfect alignment.

1. **Reprojection Error:** Project the 3D mesh back to 2D using the solved $R/T$. If the "projected nose" is $>5$ pixels away from the "image nose," trigger refinement.
2. **Differentiable Refinement:** (Optional) Use **PyTorch3D's Differentiable Renderer** to nudge $R$ and $T$ for 10 iterations to minimize the pixel-wise difference between the predicted NOCS and the rendered mesh NOCS.

---

## Implementation Checklist
- [x] **100k Meshes:** Acquired.
- [ ] **Step 1:** Run `normalize_mesh` in parallel (use `multiprocessing`).
- [ ] **Step 2:** Render 500k pairs (Use a GPU cluster; this is the bottleneck).
- [ ] **Step 3:** Train U-Net (Target: 2–3 days on a modern GPU).
- [ ] **Step 4:** Deploy Umeyama Solver for real-time 6D pose.
