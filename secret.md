# NOCS-Inspired 3D Face Pose Pipeline

This pipeline implements a **Dense Coordinate Regression** strategy. You will use your 3D mesh to teach a CNN how to map 2D pixels to 3D "GPS" coordinates (NOCS), then use a mathematical solver to "snap" the mesh into the correct pose.

---

## Phase 1: Canonicalization & Data Generation
**Goal:** Create a dataset of 20,000 "Rainbow Maps" where every pixel color represents a 3D coordinate.

### 1.1 Mesh Normalization
* **Bounding Box:** Calculate the min/max $x, y, z$ of your Master Mesh.
* **Scale:** Uniformly scale the mesh so the **diagonal of the bounding box = 1.0**.
* **Center:** Translate the mesh so its center is at $[0.5, 0.5, 0.5]$.
* **Result:** Every vertex now lives strictly within the unit cube $\{x, y, z\} \in [0, 1]$.

### 1.2 The Rainbow Map (NOCS)
* **Vertex Coloring:** Assign each vertex an RGB value equal to its normalized XYZ position:
    * $Red = X_{coord}$
    * $Green = Y_{coord}$
    * $Blue = Z_{coord}$
* **Rendering:** Use **PyTorch3D** to render 20,000 images from random camera angles.
    * **Input X:** Realistic render (using textures/lighting from your 5 anchors).
    * **Label Y:** NOCS map (a "flat" shader render showing only the vertex colors).

---

## Phase 2: CNN Training
**Goal:** Train an AI to look at a regular photo and "see" the underlying 3D coordinates.

* **Architecture:** **U-Net** or **ResNet-50 Encoder-Decoder**.
* **Input:** 2D Partial View (RGB).
* **Output:** 3-channel NOCS Map (RGB).
* **Loss Function:** * **Smooth L1 Loss:** For regressing the $x, y, z$ values.
    * **Mask Loss:** Use a Binary Cross Entropy (BCE) mask so the AI only calculates loss on the face, not the background.

---

## Phase 3: The 6D Pose Solver (Inference)
**Goal:** Convert the AI's "Rainbow Guess" into a real Camera Matrix ($R$ and $T$).

1.  **Predict:** Feed the **Query Image** into the CNN to get the **Predicted NOCS Map**.
2.  **Sample Points:** Pick $N$ pixels (e.g., 1000) from the predicted map with the highest confidence.
    * **Set A (2D):** The $(u, v)$ pixel locations in the image.
    * **Set B (3D):** The $(x, y, z)$ values stored in those pixels (the "Rainbow" values).
3.  **The Umeyama Algorithm (SVD):**
    * Perform a rigid Procrustes alignment between the **3D Mesh Vertices** and the **Predicted 3D points (Set B)**.
    * This solves for:
        * **Rotation ($R$):** 3x3 Matrix.
        * **Translation ($T$):** 3x1 Vector.
        * **Scale ($s$):** To match the predicted box to the real-world mesh.

---

## Phase 4: Verification & Refinement
**Goal:** Prove the pose is correct.

1.  **Reprojection:** Using the solved $R$ and $T$, project the 3D Master Mesh back onto the 2D Query Image plane.
2.  **Metric:** * **ADD (Average Distance):** Calculate the mean distance between the projected vertices and the query features.
    * **Visual Check:** Overlay the mesh wireframe on the query.
3.  **Refinement:** If needed, use a **Differentiable Renderer** for 5–10 iterations to "fine-tune" the $R/T$ until the pixels align perfectly.

---

## Implementation Checklist
- [ ] **Step 1:** Write `normalize_mesh(mesh)` script to fit into the $[0,1]$ unit cube.
- [ ] **Step 2:** Setup PyTorch3D `Renderer` to output `nocs_map` and `rgb_image` pairs.
- [ ] **Step 3:** Define U-Net architecture with 3-channel output and $L_1$ loss.
- [ ] **Step 4:** Implement `umeyama_alignment(predicted_nocs, reference_mesh)` using SVD.

**Would you like the specific Python code for the `normalize_mesh` function to get the NOCS coordinates ready?**
