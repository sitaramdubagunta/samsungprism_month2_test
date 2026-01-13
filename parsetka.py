import numpy as np

def parse_tka(path):
    R = None
    params = {}

    with open(path, "r", errors="ignore") as f:
        raw_lines = f.readlines()

    lines = [l.strip() for l in raw_lines if l.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # ----------------------------
        # Rotation matrix (%M or M)
        # ----------------------------
        if line == "M" or line == "%M":
            try:
                R = np.array([
                    list(map(float, lines[i+1].split())),
                    list(map(float, lines[i+2].split())),
                    list(map(float, lines[i+3].split()))
                ], dtype=np.float64)
            except Exception as e:
                raise ValueError("Failed to parse rotation matrix after M/%M") from e

            i += 4
            continue

        # ----------------------------
        # Camera center
        # ----------------------------
        if line.startswith("%X"):
            params["X"] = float(line.split()[1])
        elif line.startswith("%Y"):
            params["Y"] = float(line.split()[1])
        elif line.startswith("%Z"):
            params["Z"] = float(line.split()[1])

        # ----------------------------
        # Intrinsics
        # ----------------------------
        elif line.startswith("%f"):
            params["f"] = float(line.split()[1])
        elif line.startswith("%x"):
            params["px"] = float(line.split()[1])
        elif line.startswith("%y"):
            params["py"] = float(line.split()[1])
        elif line.startswith("%a"):
            params["cx"] = float(line.split()[1])
        elif line.startswith("%b"):
            params["cy"] = float(line.split()[1])

        # ----------------------------
        # Image size
        # ----------------------------
        elif line.startswith("%is"):
            parts = line.split()
            params["width"] = int(parts[1])
            params["height"] = int(parts[2])

        # ----------------------------
        # Distortion
        # ----------------------------
        elif line.startswith("%K "):
            params["k1"] = float(line.split()[1])
        elif line.startswith("%K2"):
            params["k2"] = float(line.split()[1])

        i += 1

    # ----------------------------
    # Validation
    # ----------------------------
    if R is None or R.shape != (3, 3):
        raise ValueError("Rotation matrix R was not parsed correctly")

    required = ["X", "Y", "Z", "f", "px", "py", "cx", "cy", "width", "height"]
    for k in required:
        if k not in params:
            raise ValueError(f"Missing parameter in TKA file: {k}")

    # ----------------------------
    # Extrinsics
    # ----------------------------
    C = np.array([params["X"], params["Y"], params["Z"]], dtype=np.float64)
    t = -R @ C

    # ----------------------------
    # Intrinsics matrix
    # ----------------------------
    fx = params["f"] / params["px"]
    fy = params["f"] / params["py"]

    K = np.array([
        [fx, 0, params["cx"]],
        [0, fy, params["cy"]],
        [0,  0,  1]
    ], dtype=np.float64)

    return {
        "R": R,
        "t": t,
        "C": C,
        "K": K,
        "image_size": (params["width"], params["height"]),
        "distortion": {
            "k1": params.get("k1", 0.0),
            "k2": params.get("k2", 0.0)
        }
    }


# ----------------------------
# TEST BLOCK
# ----------------------------
if __name__ == "__main__":
    data = parse_tka("28_C.tka")

    print("\n=== PARSE SUCCESS ===")
    print("R:\n", data["R"])
    print("\nt:\n", data["t"])
    print("\nK:\n", data["K"])
    print("\nImage size:", data["image_size"])
    print("\nDistortion:", data["distortion"])
