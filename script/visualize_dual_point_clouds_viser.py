import argparse
import os
import time
from glob import glob

import numpy as np


def parse_args():
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize paired random/FPS point clouds side by side with viser."
    )
    parser.add_argument("--random-root", required=True, help="Root folder for random point clouds.")
    parser.add_argument("--fps-root", required=True, help="Root folder for FPS point clouds.")
    parser.add_argument("--scene-count", type=int, default=5, help="Number of scenes to show.")
    parser.add_argument("--pc-name", default="partial_pc_00.npy", help="Point-cloud file name to load per scene.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--scene-spacing", type=float, default=0.75, help="Spacing between scene groups.")
    parser.add_argument("--version-spacing", type=float, default=0.22, help="Spacing between random and FPS clouds.")
    parser.add_argument("--point-size", type=float, default=0.004, help="Viser point size.")
    return parser.parse_args()


def relative_scene_id(pc_path, root, pc_name):
    """Resolve the scene id for one point-cloud path.

    Args:
        pc_path: Absolute or relative point-cloud file path.
        root: Root folder containing scene subdirectories.
        pc_name: Point-cloud file name to strip from the relative path.

    Returns:
        Scene id relative to ``root``.
    """
    rel_path = os.path.relpath(pc_path, root)
    if not rel_path.endswith(pc_name):
        raise ValueError(f"Unexpected point-cloud path: {pc_path}")
    return os.path.dirname(rel_path)


def discover_scene_ids(random_root, fps_root, pc_name, scene_count):
    """Find scene ids present in both random and FPS output roots.

    Args:
        random_root: Root folder for random point clouds.
        fps_root: Root folder for FPS point clouds.
        pc_name: Point-cloud file name to require per scene.
        scene_count: Maximum number of scene ids to return.

    Returns:
        A sorted list of scene ids available in both roots.
    """
    random_paths = sorted(glob(os.path.join(random_root, "**", pc_name), recursive=True))
    scene_ids = []
    for random_path in random_paths:
        scene_id = relative_scene_id(random_path, random_root, pc_name)
        fps_path = os.path.join(fps_root, scene_id, pc_name)
        if os.path.exists(fps_path):
            scene_ids.append(scene_id)
        if len(scene_ids) >= scene_count:
            break
    if len(scene_ids) < scene_count:
        raise RuntimeError(
            f"Found only {len(scene_ids)} paired scenes under {random_root} and {fps_root}; "
            f"expected {scene_count}."
        )
    return scene_ids


def load_point_cloud(path):
    """Load one point cloud as float32.

    Args:
        path: Path to a ``.npy`` point-cloud file with shape ``(N, 3)``.

    Returns:
        A float32 point-cloud array with shape ``(N, 3)``.
    """
    points = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32).reshape(-1, 3)
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shaped (N, 3), got {points.shape}: {path}")
    return points


def add_cloud(server, name, points, color, offset, point_size):
    """Add one translated point cloud to the viser scene.

    Args:
        server: Viser server.
        name: Scene node name.
        points: Point coordinates with shape ``(N, 3)``.
        color: RGB color triplet in 0-255 range.
        offset: Translation offset applied before display.
        point_size: Viser point size.

    Returns:
        None.
    """
    colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (points.shape[0], 1))
    server.scene.add_point_cloud(
        name,
        points=points + np.asarray(offset, dtype=np.float32)[None, :],
        colors=colors,
        point_size=point_size,
    )


def add_label(server, name, text, position):
    """Add a label when the installed viser version supports labels.

    Args:
        server: Viser server.
        name: Scene node name.
        text: Label text.
        position: Label position.

    Returns:
        None.
    """
    if hasattr(server.scene, "add_label"):
        server.scene.add_label(name, text=text, position=np.asarray(position, dtype=np.float32))
    else:
        print(text)


def main():
    """Start a viser server with paired random/FPS point clouds.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "viser is required for this visualizer; install it with `pip install viser`."
        ) from exc

    scene_ids = discover_scene_ids(args.random_root, args.fps_root, args.pc_name, args.scene_count)
    server = viser.ViserServer(port=args.port)

    for scene_idx, scene_id in enumerate(scene_ids):
        group_offset = np.asarray([scene_idx * args.scene_spacing, 0.0, 0.0], dtype=np.float32)
        random_path = os.path.join(args.random_root, scene_id, args.pc_name)
        fps_path = os.path.join(args.fps_root, scene_id, args.pc_name)
        random_pc = load_point_cloud(random_path)
        fps_pc = load_point_cloud(fps_path)

        add_cloud(
            server,
            f"/scene_{scene_idx:02d}/random4096",
            random_pc,
            [64, 148, 255],
            group_offset + np.asarray([0.0, -args.version_spacing, 0.0]),
            args.point_size,
        )
        add_cloud(
            server,
            f"/scene_{scene_idx:02d}/fps4096",
            fps_pc,
            [255, 149, 0],
            group_offset + np.asarray([0.0, args.version_spacing, 0.0]),
            args.point_size,
        )
        compact_id = scene_id.split("/tabletop_ur10e/")[0]
        add_label(
            server,
            f"/scene_{scene_idx:02d}/label",
            f"{scene_idx}: {compact_id}\nblue=random, orange=fps",
            group_offset + np.asarray([0.0, 0.0, 0.35], dtype=np.float32),
        )

    print(f"Loaded {len(scene_ids)} paired scenes.")
    print(f"Viser server running at http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
