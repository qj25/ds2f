import numpy as np
from typing import Optional, Tuple

# ---------- Utilities ----------

def _arc_length_resample(points: np.ndarray, M: int) -> np.ndarray:
    """Standard chord/arc-length interpolation of a polyline to M samples (endpoints preserved)."""
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    L = cum[-1]
    if L == 0:
        return np.repeat(points[:1], M, axis=0)
    s_targets = np.linspace(0.0, L, M)
    out = np.empty((M, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(s_targets, cum, points[:, k])
    return out

def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Closest point to p on segment [a,b]. Returns (point, t in [0,1])."""
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 0.0:
        return a, 0.0
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab, t

def _closest_point_on_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Closest point to p on a polyline defined by vertices poly (N,3)."""
    best_q = poly[0]
    best_d2 = np.inf
    for i in range(len(poly) - 1):
        q, _ = _closest_point_on_segment(p, poly[i], poly[i+1])
        d2 = np.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_q = q
    return best_q

# ---------- Exact equal-spacing projection (FABRIK with both ends anchored) ----------

def _project_equal_spacing(P: np.ndarray, d: float, proj_order: int, ref_dirs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One forward+backward projection pass enforcing exact distance d and fixed endpoints."""
    M = len(P)
    S, E = P[0].copy(), P[-1].copy()
    step_ratio = 1.0

    def step(a, b, ref):
        v = b - a
        n = np.linalg.norm(v)
        if n < 1e-18:
            v = ref
            n = np.linalg.norm(v)
            if n < 1e-18:
                v = rng.normal(size=3); n = np.linalg.norm(v)
        return a + (v / n) * d * step_ratio

    if proj_order == 0: # forward then backward
        # forward
        P[0] = S
        for i in range(M - 1):
            P[i+1] = step(P[i], P[i+1], ref_dirs[i])
        # backward
        P[-1] = E
        for i in range(M - 2, -1, -1):
            v = P[i] - P[i+1]
            n = np.linalg.norm(v)
            if n < 1e-18:
                ref = -ref_dirs[i if i < len(ref_dirs) else -1]
                nref = np.linalg.norm(ref)
                v = ref if nref > 1e-18 else rng.normal(size=3)
                n = np.linalg.norm(v)
            P[i] = P[i+1] + (v / n) * d * step_ratio
    else: # backward then forward
        # backward
        P[-1] = E
        for i in range(M - 2, -1, -1):
            v = P[i] - P[i+1]
            n = np.linalg.norm(v)
            if n < 1e-18:
                ref = -ref_dirs[i if i < len(ref_dirs) else -1]
                nref = np.linalg.norm(ref)
                v = ref if nref > 1e-18 else rng.normal(size=3)
                n = np.linalg.norm(v)
            P[i] = P[i+1] + (v / n) * d * step_ratio
        # forward
        P[0] = S
        for i in range(M - 1):
            P[i+1] = step(P[i], P[i+1], ref_dirs[i])

    # ensure exact anchors
    P[0], P[-1] = S, E
    return P

# ---------- Main solver ----------

def resample_wire_equal_distance_min_dev(
    points: np.ndarray,
    x_interp: int,
    *,
    iters: int = 200,
    step_size: float = 0.5,
    rng_seed: Optional[int] = None,
    tol: float = 1e-9,
    plot_steps: bool = False,
) -> np.ndarray:
    """
    Resample a 3D polyline to x_interp points with:
      - EXACT equal Euclidean spacing between consecutive points,
      - endpoints fixed to the original ends,
      - deviation from the original polyline minimized by pulling toward nearest points on it.

    Parameters
    ----------
    points : (N,3) array
        Original ordered points.
    x_interp : int
        Number of points in the resampled wire.
    iters : int
        Number of alternating minimize+project iterations.
    step_size : float
        Step size toward nearest points on the original polyline (0..1).
    rng_seed : Optional[int]
        RNG seed for robustness when segments collapse.
    tol : float
        Early-stop tolerance (max point motion per iteration).

    Returns
    -------
    P : (x_interp,3) array
        Resampled points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be (N,3)")
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")
    if x_interp < 2:
        raise ValueError("x_interp must be >= 2.")

    rng = np.random.default_rng(rng_seed)

    # total length & target spacing
    L = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if L == 0.0:
        return np.repeat(points[:1], x_interp, axis=0)
    d = L / (x_interp - 1)

    # initial guess via arc-length resampling (keeps the rough shape)
    Q = _arc_length_resample(points, x_interp)
    P = Q.copy()
    P[0], P[-1] = points[0], points[-1]
    
    # reference directions from initial guess (for stable projection when local segments degenerate)
    ref_dirs = np.diff(Q, axis=0)
    for i in range(len(ref_dirs)):
        if np.linalg.norm(ref_dirs[i]) < 1e-18:
            ref_dirs[i] = rng.normal(size=3)

    proj_order = 0  # alternate projection order for stability

    # Alternate: (1) pull toward nearest original-polyline points, (2) project to equal-spacing exactly
    for _ in range(iters):
        P_prev = P.copy()

        # 1) deviation minimization: move interior points toward nearest points on original polyline
        for i in range(1, x_interp - 1):
            q = _closest_point_on_polyline(P[i], points)
            P[i] = (1.0 - step_size) * P[i] + step_size * q
        # print("After dev minimization:", P)

        # 2) exact equal-spacing projection with endpoints fixed
        P = _project_equal_spacing(P, d, proj_order, ref_dirs, rng)
        proj_order = 1 - proj_order
        # print("After equal spacing:", P)

        # early stop
        max_move = np.max(np.linalg.norm(P - P_prev, axis=1))
        if max_move < tol:
            break

        if plot_steps:
            # Plot original vs resampled
            import matplotlib.pyplot as plt

            def set_axes_equal(ax):
                """Make 3D plot axes have equal scale (so that spheres look like spheres)."""
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                x_middle = np.mean(x_limits)
                y_range = abs(y_limits[1] - y_limits[0])
                y_middle = np.mean(y_limits)
                z_range = abs(z_limits[1] - z_limits[0])
                z_middle = np.mean(z_limits)

                plot_radius = 0.5 * max([x_range, y_range, z_range])

                ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
                ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
                ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(points[:,0], points[:,1], points[:,2], 'o-', label="Original wire", color="tab:blue")
            ax.plot(P[:,0], P[:,1], P[:,2], 'o-', label="Resampled (equal spacing)", color="tab:orange")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            ax.view_init(elev=0, azim=-90)   # azim=180 → +y direction
            set_axes_equal(ax)   # << make equal scale
            ax.set_title("3D Wire Resampling")

            plt.show()

    # # Final safety projection to lock distances exactly
    # P = _project_equal_spacing(P, d, ref_dirs, rng)
    return P

# ---------- Verification helper ----------

def verify_resample(original: np.ndarray, new: np.ndarray, atol: float = 1e-10) -> dict:
    """Report spacing, endpoints, and lengths."""
    L0 = float(np.sum(np.linalg.norm(np.diff(original, axis=0), axis=1)))
    seg = np.linalg.norm(np.diff(new, axis=0), axis=1)
    L1 = float(np.sum(seg))
    d_target = L0 / (len(new) - 1)
    return {
        "endpoints_match": np.allclose(new[0], original[0], atol=atol) and np.allclose(new[-1], original[-1], atol=atol),
        "segment_min": float(seg.min()),
        "segment_max": float(seg.max()),
        "segment_std": float(seg.std()),
        "target_d": d_target,
        "total_length_orig": L0,
        "total_length_new": L1,
        "max_abs_spacing_error": float(np.max(np.abs(seg - d_target))),
    }

# ---------- Example ----------
if __name__ == "__main__":
    pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 1],
        [3, 1, 1.5],
        [4, 0.2, 2.0]
    ], dtype=float)

    # Generate a test "wire": a smooth 3D helix with 60 points
    t = np.linspace(0, 4*np.pi, 60)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2*np.pi)
    pts = np.column_stack((x, y, z))

    pts = np.array([
        [0.0,0,0],
        [0.0,0,-1],
        [3.0,0,-1],
        [3.0,0,0],
    ], dtype=float)


    M = 33
    new_pts = resample_wire_equal_distance_min_dev(pts, M, iters=300, step_size=0.4, rng_seed=0)
    report = verify_resample(pts, new_pts)

    print("New points:", new_pts)
    print("Verification:", report)

    # Plot original vs resampled
    import matplotlib.pyplot as plt

    def set_axes_equal(ax):
        """Make 3D plot axes have equal scale (so that spheres look like spheres)."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pts[:,0], pts[:,1], pts[:,2], 'o-', label="Original wire", color="tab:blue")
    ax.plot(new_pts[:,0], new_pts[:,1], new_pts[:,2], 'o-', label="Resampled (equal spacing)", color="tab:orange")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=180)   # azim=180 → +y direction
    ax.legend()
    set_axes_equal(ax)   # << make equal scale
    ax.set_title("3D Wire Resampling")

    plt.show()