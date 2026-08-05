import os
import sys
import time
import fcntl
from contextlib import contextmanager

import pyglet

pyglet.options["headless"] = True
import numpy as np
import warp as wp
import warp.render
import torch
import trimesh
from tqdm import tqdm

os.environ["DISPLAY"] = ":98"  # useless?
import cv2

from .rotation import np_normalize


@contextmanager
def gpu_render_lock(gpu_id):
    lock_path = f"/tmp/meshprocess_render_gpu_{gpu_id}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def camera_spherical(sample_num, radius=1.0):
    points = np.random.randn(sample_num, 3)
    points = radius * np_normalize(points)
    return points


def camera_circular_zaxis(sample_num, radius=0.8, center=[0, 0, 0.8]):
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    theta = np.arange(sample_num) * phi
    pos = np.array(center) + radius * np.stack(
        [np.cos(theta), np.sin(theta), theta * 0], axis=-1
    )
    return pos


def camera_circular_zaxis_uniform_group(
    pc_per_scene,
    views_per_pc,
    radius=0.8,
    height=0.8,
    center_xy=[0.0, 0.0],
    start_angle=0.0,
    global_yaw_mode="random",
    random_start_angle=False,
    random_global_yaw=None,
    **kwargs,
):
    """Create grouped tabletop camera positions around the world z axis.

    Args:
        pc_per_scene: Number of fused point-cloud samples to generate per scene.
        views_per_pc: Number of camera views fused into each point-cloud sample.
        radius: Horizontal distance from the look-at center to each camera.
        height: Absolute z height shared by all cameras.
        center_xy: XY center of the circular camera trajectory.
        start_angle: Base angle offset in radians.
        global_yaw_mode: Strategy for spacing fused samples around z. Supported
            values are ``random``, ``uniform``, ``golden_angle``, and ``fixed``.
        random_start_angle: Whether to add one random scene-level angle offset.
        random_global_yaw: Deprecated boolean alias. ``True`` maps to
            ``global_yaw_mode='random'`` and ``False`` maps to ``fixed``.
        **kwargs: Extra config entries ignored by this position generator.

    Returns:
        A ``(pc_per_scene * views_per_pc, 3)`` array of camera positions.
    """
    if random_global_yaw is not None:
        global_yaw_mode = "random" if random_global_yaw else "fixed"

    start_angle = float(start_angle)
    if random_start_angle:
        start_angle += np.random.random() * 2.0 * np.pi

    if global_yaw_mode == "random":
        group_yaw = np.random.random(pc_per_scene) * 2.0 * np.pi
    elif global_yaw_mode == "uniform":
        group_yaw = np.arange(pc_per_scene) * (2.0 * np.pi / pc_per_scene)
    elif global_yaw_mode == "golden_angle":
        group_yaw = np.arange(pc_per_scene) * np.pi * (3.0 - np.sqrt(5.0))
    elif global_yaw_mode == "fixed":
        group_yaw = np.zeros(pc_per_scene)
    else:
        raise ValueError(f"Unsupported global_yaw_mode: {global_yaw_mode}")

    base_theta = np.arange(views_per_pc) * (2.0 * np.pi / views_per_pc)
    group_yaw = start_angle + group_yaw
    theta = (group_yaw[:, None] + base_theta[None, :]).reshape(-1)
    center_xy = np.asarray(center_xy)
    pos = np.stack(
        [
            center_xy[0] + radius * np.cos(theta),
            center_xy[1] + radius * np.sin(theta),
            np.full_like(theta, height),
        ],
        axis=-1,
    )
    return pos


def camera_view_matrix(
    sample_num,
    pos,
    pos_noise=0,
    lookat=[0, 0, 0.0],
    lookat_noise=0,
    up=None,
    up_noise=0,
    **kwargs,
):
    pos = np.array(pos) + pos_noise * (np.random.random((sample_num, 3)) - 0.5)
    lookat = np.array(lookat) + lookat_noise * (np.random.random((sample_num, 3)) - 0.5)
    front = np_normalize(lookat - pos)

    while 1:
        up = np.array(up) if up is not None else np.random.randn(sample_num, 3)
        up = np_normalize(up + up_noise * (np.random.random((sample_num, 3)) - 0.5))
        up = up - np.sum(up * front, axis=-1, keepdims=True) * front
        up = np_normalize(up)
        if not np.any(np.linalg.norm(up, axis=-1) < 1e-6):
            break

    view_matrix = np.zeros((sample_num, 4, 4))
    view_matrix[:, 0, :3] = np.cross(front, up)
    view_matrix[:, 1, :3] = up
    view_matrix[:, 2, :3] = -front
    view_matrix[:, 3, :3] = pos
    view_matrix[:, 3, 3] = 1
    return view_matrix


def get_camera_matrix(config):
    if config["type"] == "spherical":
        pos = camera_spherical(config["sample_num"], config["radius"])
    elif config["type"] == "circular_zaxis":
        pos = camera_circular_zaxis(
            config["sample_num"], config["radius"], config["center"]
        )
    elif config["type"] == "circular_zaxis_uniform_group":
        pos = camera_circular_zaxis_uniform_group(
            pc_per_scene=config["pc_per_scene"],
            views_per_pc=config["views_per_pc"],
            radius=config["radius"],
            height=config["height"],
            center_xy=config.get("center_xy", [0.0, 0.0]),
            start_angle=config.get("start_angle", 0.0),
            global_yaw_mode=config.get("global_yaw_mode", "random"),
            random_start_angle=config.get("random_start_angle", False),
            random_global_yaw=config.get("random_global_yaw", None),
        )
    else:
        raise NotImplementedError("Unsupported camera type")
    return camera_view_matrix(pos=pos, **config)


def scene_cfg2mesh(scene_cfg, scene_cfg_path):
    tm_lst = []
    for obj in scene_cfg["scene"].values():
        if obj["type"] == "rigid_object":
            tm = trimesh.load(
                os.path.join(os.path.dirname(scene_cfg_path), obj["file_path"]),
                force="mesh",
            )
            tm.vertices *= obj["scale"]
        elif obj["type"] == "plane":
            continue
            plane_thick = 0.01
            delta_transform = trimesh.transformations.translation_matrix(
                [0, 0, -plane_thick / 2]
            )
            tm = trimesh.creation.box(
                extents=[1.0, 1.0, plane_thick], transform=delta_transform
            )
        else:
            raise NotImplementedError("Unsupported object type")

        rotation_matrix = trimesh.transformations.quaternion_matrix(obj["pose"][3:])
        rotation_matrix[:3, 3] = obj["pose"][:3]
        tm.apply_transform(rotation_matrix)
        tm_lst.append(tm)
    scene_mesh = trimesh.util.concatenate(tm_lst)
    return scene_mesh


class WarpRender:
    def __init__(
        self,
        device,
        tile_width=1280,
        tile_height=720,
        z_near=0.1,
        z_far=10.0,
        n_cols=2,
        n_rows=2,
        camera_type="kinect",
        camera_model=None,
    ):
        # tile width and height: the size of each tile
        # n_cols and n_rows: the number of columns and rows
        # screnn_width=tile_width*n_cols, screen_height=tile_height*n_rows
        self.device = device

        self.ncols = n_cols
        self.nrows = n_rows
        self.num_tiles = n_cols * n_rows
        self.tile_width = tile_width
        self.tile_height = tile_height

        self.renderer = wp.render.OpenGLRenderer(
            draw_axis=False,
            draw_grid=False,
            show_info=False,
            draw_sky=False,
            screen_width=tile_width * n_cols,
            screen_height=tile_height * n_rows,
            near_plane=z_near,
            far_plane=z_far,
            vsync=False,
            headless=True,
            device=device,
        )

        # setup intrinsics
        camera_model = dict(camera_model or {})
        camera_type = camera_model.get("type", camera_type)
        if camera_type == "kinect":
            self.cx = tile_width // 2
            self.cy = tile_height // 2
            self.fx = 608.6939697265625
            self.fy = 608.6422119140625
        elif camera_type == "custom":
            self.fx = float(camera_model["fx"])
            self.fy = float(camera_model["fy"])
            self.cx = float(camera_model["cx"])
            self.cy = float(camera_model["cy"])
        else:
            raise NotImplementedError

        self.projection_matrixs = [
            np.array(
                [
                    [2 * self.fx / tile_width, 0, 0, 0],
                    [0, 2 * self.fy / tile_height, 0, 0],
                    # Warp's OpenGL renderer consumes row-vector projection
                    # matrices. Principal-point offsets therefore live in the
                    # third row, not the third column.
                    [
                        1 - 2 * self.cx / tile_width,
                        2 * self.cy / tile_height - 1,
                        -(z_far + z_near) / (z_far - z_near),
                        -1,
                    ],
                    [0, 0, -2 * z_far * z_near / (z_far - z_near), 0],
                ]
            )
        ] * self.num_tiles

        self.setup_tile_flag = False
        self.help_yy, self.help_xx = torch.meshgrid(
            torch.arange(self.tile_height).to(self.device),
            torch.arange(self.tile_width).to(self.device),
            indexing="ij",
        )
        self.help_xx = self.help_xx[None, :, :, None]
        self.help_yy = self.help_yy[None, :, :, None]
        return

    def update_camera_poses(self, view_matrix):
        self.view_matrix = torch.tensor(view_matrix).to(self.device).float()
        inv_view_matrix = torch.inverse(self.view_matrix).tolist()

        if not self.setup_tile_flag:
            self.renderer.setup_tiled_rendering(
                instances=[[0]] * self.num_tiles,
                tile_sizes=[(self.tile_width, self.tile_height)] * self.num_tiles,
                projection_matrices=self.projection_matrixs,
                view_matrices=inv_view_matrix,
                tile_ncols=self.ncols,
                tile_nrows=self.nrows,
            )
            self.setup_tile_flag = True
        else:
            for id in range(self.num_tiles):
                self.renderer.update_tile(
                    tile_id=id,
                    instances=[0],
                    tile_size=(self.tile_width, self.tile_height),
                    projection_matrix=self.projection_matrixs[id],
                    view_matrix=inv_view_matrix[id],
                )
        return

    def get_image(self, mode="depth"):
        if mode == "depth":
            image = wp.zeros(
                (self.num_tiles, self.tile_height, self.tile_width, 1), dtype=wp.float32
            )
        elif mode == "rgb":
            image = wp.zeros(
                (self.num_tiles, self.tile_height, self.tile_width, 3), dtype=wp.float32
            )
        else:
            raise NotImplementedError

        success = self.renderer.get_pixels(image, split_up_tiles=True, mode=mode)
        if not success:
            raise RuntimeError(f"Failed to read {mode} pixels from OpenGL renderer")

        return wp.to_torch(image)

    def render(self, obj_mesh, camera_view_matrix):
        self.renderer.clear()
        self.update_camera_poses(camera_view_matrix)
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)
        self.renderer.render_mesh(
            name="mesh", points=obj_mesh.vertices, indices=obj_mesh.faces
        )
        self.renderer.end_frame()
        return self.view_matrix

    def depth_to_point_cloud(self, depth_image):
        x = (self.help_xx - self.cx) * depth_image / self.fx
        y = -(self.help_yy - self.cy) * depth_image / self.fy
        camera_coords = torch.cat(
            [x, y, -depth_image, torch.ones_like(x, device=x.device)], axis=-1
        )
        world_coords = (
            camera_coords.view(depth_image.shape[0], -1, 4) @ self.view_matrix
        )
        return world_coords[..., :3]


def cfg_get(config, key, default=None):
    """Read a config value from a dict-like object.

    Args:
        config: Dict, OmegaConf node, or object exposing ``get``.
        key: Config key to read.
        default: Value returned when the key is missing or explicitly empty.

    Returns:
        The requested config value, or ``default`` when it is unavailable.
    """
    try:
        value = config.get(key, default)
    except AttributeError:
        value = default
    return default if value is None else value


def build_warp_renderer(device, func_config, n_cols=None, n_rows=None):
    """Create a renderer using global render config and a requested tile layout.

    Args:
        device: CUDA device string used by Warp and OpenGL.
        func_config: Render function config containing optional ``renderer`` keys.
        n_cols: Optional override for tiled rendering columns.
        n_rows: Optional override for tiled rendering rows.

    Returns:
        A ``WarpRender`` instance with the requested tile shape.
    """
    renderer_config = cfg_get(func_config, "renderer", {}) or {}
    camera_model = cfg_get(renderer_config, "camera_model", None)
    return WarpRender(
        device,
        tile_width=cfg_get(renderer_config, "tile_width", 1280),
        tile_height=cfg_get(renderer_config, "tile_height", 720),
        z_near=cfg_get(renderer_config, "z_near", 0.1),
        z_far=cfg_get(renderer_config, "z_far", 10.0),
        n_cols=n_cols if n_cols is not None else cfg_get(renderer_config, "n_cols", 2),
        n_rows=n_rows if n_rows is not None else cfg_get(renderer_config, "n_rows", 2),
        camera_model=camera_model,
    )


def render_scene_views(renderer, obj_mesh, camera_view_matrix, func_config, gpu_id):
    """Render a scene mesh from a batch of camera views and read requested data.

    Args:
        renderer: Initialized ``WarpRender`` whose tile count matches the views.
        obj_mesh: Trimesh scene mesh to render.
        camera_view_matrix: Camera-to-world matrices for all tiled views.
        func_config: Render config controlling RGB/depth/point-cloud outputs.
        gpu_id: GPU id used for the per-GPU render lock.

    Returns:
        Tuple ``(view_matrix, rgb_image, depth_image, all_pc, mask)``. Optional
        entries are ``None`` when the matching output is disabled.
    """
    rgb_image = None
    depth_image = None
    all_pc = None
    mask = None
    view_num = camera_view_matrix.shape[0]

    for retry_id in range(3):
        try:
            with gpu_render_lock(gpu_id):
                view_matrix = renderer.render(obj_mesh, camera_view_matrix)

                if func_config.save_rgb:
                    rgb_image = renderer.get_image(mode="rgb")

                if func_config.save_depth or func_config.save_pc:
                    depth_image = renderer.get_image(mode="depth")
                    if func_config.save_pc:
                        all_pc = renderer.depth_to_point_cloud(depth_image)
                        mask = depth_image.reshape(view_num, -1) < 5
            break
        except RuntimeError:
            if retry_id == 2:
                raise
            time.sleep(0.1)

    return view_matrix, rgb_image, depth_image, all_pc, mask


def sample_point_cloud(pc, max_point_num):
    """Randomly downsample a point cloud without exceeding the available points.

    Args:
        pc: Torch tensor containing point coordinates.
        max_point_num: Maximum number of points to keep.

    Returns:
        A torch tensor containing at most ``max_point_num`` points.
    """
    if pc.shape[0] == 0:
        return pc
    sample_num = min(pc.shape[0], max_point_num)
    return pc[torch.randperm(pc.shape[0], device=pc.device)[:sample_num]]


def farthest_point_sample_torch(pc, max_point_num):
    """Downsample a point cloud with a pure PyTorch FPS implementation.

    Args:
        pc: Torch tensor containing point coordinates.
        max_point_num: Maximum number of points to keep.

    Returns:
        A torch tensor containing at most ``max_point_num`` FPS-selected points.
    """
    sample_num = min(pc.shape[0], max_point_num)
    if sample_num == pc.shape[0]:
        return pc

    selected = torch.empty(sample_num, dtype=torch.long, device=pc.device)
    min_dist = torch.full((pc.shape[0],), float("inf"), device=pc.device)
    farthest = torch.randint(pc.shape[0], (1,), device=pc.device, dtype=torch.long)

    for sample_id in range(sample_num):
        selected[sample_id] = farthest
        centroid = pc[farthest].view(1, 3)
        dist = torch.sum((pc - centroid) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist).view(1)

    return pc[selected]


def farthest_point_sample_pytorch3d(pc, max_point_num):
    """Downsample a point cloud with PyTorch3D's CUDA FPS operator.

    Args:
        pc: Torch tensor containing point coordinates.
        max_point_num: Maximum number of points to keep.

    Returns:
        A torch tensor containing at most ``max_point_num`` FPS-selected points.
    """
    from pytorch3d.ops import sample_farthest_points

    sample_num = min(pc.shape[0], max_point_num)
    if sample_num == pc.shape[0]:
        return pc

    sampled_pc, _ = sample_farthest_points(
        pc[None].contiguous().float(),
        K=sample_num,
        random_start_point=True,
    )
    return sampled_pc[0].to(dtype=pc.dtype)


def farthest_point_sample(pc, max_point_num, pre_sample_num=None, backend="auto"):
    """Downsample a point cloud with farthest point sampling.

    Args:
        pc: Torch tensor containing point coordinates.
        max_point_num: Maximum number of points to keep.
        pre_sample_num: Optional random pre-sample limit used to cap FPS cost.
        backend: FPS backend. Supported values are ``auto``, ``pytorch3d``,
            and ``torch``.

    Returns:
        A torch tensor containing at most ``max_point_num`` FPS-selected points.
    """
    if pc.shape[0] == 0:
        return pc
    if pre_sample_num is not None and pc.shape[0] > pre_sample_num:
        pc = sample_point_cloud(pc, int(pre_sample_num))

    sample_num = min(pc.shape[0], max_point_num)
    if sample_num == pc.shape[0]:
        return pc

    if backend == "torch":
        return farthest_point_sample_torch(pc, max_point_num)

    if backend in ("auto", "pytorch3d"):
        try:
            return farthest_point_sample_pytorch3d(pc, max_point_num)
        except Exception:
            if backend == "pytorch3d":
                raise
            return farthest_point_sample_torch(pc, max_point_num)

    raise ValueError(f"Unsupported FPS backend: {backend}")


def sample_point_cloud_with_method(pc, output_config, max_point_num):
    """Sample a point cloud according to one output configuration.

    Args:
        pc: Torch tensor containing point coordinates.
        output_config: Output config with ``sample_method`` and FPS options.
        max_point_num: Maximum number of points to keep.

    Returns:
        A sampled torch tensor containing at most ``max_point_num`` points.
    """
    sample_method = cfg_get(output_config, "sample_method", "random")
    if sample_method == "random":
        return sample_point_cloud(pc, max_point_num)
    if sample_method == "fps":
        return farthest_point_sample(
            pc,
            max_point_num,
            pre_sample_num=cfg_get(output_config, "fps_pre_sample_num", None),
            backend=cfg_get(output_config, "fps_backend", "auto"),
        )
    raise ValueError(f"Unsupported point cloud sample method: {sample_method}")


def get_point_cloud_outputs(func_config):
    """Build the point-cloud output list while preserving legacy behavior.

    Args:
        func_config: Render config containing ``save_path`` and optional
            ``point_cloud_outputs`` entries.

    Returns:
        A list of output config dictionaries, each with a save path and sampling
        method.
    """
    output_configs = cfg_get(func_config, "point_cloud_outputs", None)
    if not output_configs:
        return [
            {
                "name": "default",
                "sample_method": "random",
                "save_path": func_config.save_path,
            }
        ]

    normalized_outputs = []
    for output_config in output_configs:
        output_config = dict(output_config)
        if not cfg_get(output_config, "enabled", True):
            continue
        output_config.setdefault("name", output_config.get("sample_method", "random"))
        output_config.setdefault("sample_method", "random")
        output_config.setdefault("save_path", func_config.save_path)
        normalized_outputs.append(output_config)

    if not normalized_outputs:
        raise ValueError("At least one point cloud output must be enabled")
    return normalized_outputs


def point_cloud_scene_folder(output_config, scene_id):
    """Resolve one output directory for a rendered scene.

    Args:
        output_config: Output config containing a ``save_path`` template.
        scene_id: Scene id used to replace the ``**`` template marker.

    Returns:
        A concrete scene output directory path.
    """
    return output_config["save_path"].replace("**", scene_id)


def point_cloud_output_root(output_config):
    """Resolve the root folder for one point-cloud output template.

    Args:
        output_config: Output config containing a ``save_path`` template.

    Returns:
        The path before the ``/**`` scene placeholder.
    """
    return output_config["save_path"].split("/**")[0]


def scene_pc_complete(scene_pc_folder, pc_per_scene):
    """Check whether all expected point-cloud files already exist.

    Args:
        scene_pc_folder: Output folder for one rendered scene.
        pc_per_scene: Number of expected ``partial_pc_XX.npy`` files.

    Returns:
        ``True`` when every expected file exists and is non-empty.
    """
    for pc_id in range(pc_per_scene):
        pc_path = os.path.join(scene_pc_folder, f"partial_pc_{str(pc_id).zfill(2)}.npy")
        if not os.path.exists(pc_path):
            return False
        check_pc = np.load(pc_path, allow_pickle=True)
        if len(check_pc) == 0:
            return False
    return True


def scene_outputs_complete(output_configs, scene_id, pc_per_scene):
    """Check whether every configured output has all point-cloud files.

    Args:
        output_configs: List of configured point-cloud outputs.
        scene_id: Scene id used to resolve output folders.
        pc_per_scene: Number of expected ``partial_pc_XX.npy`` files.

    Returns:
        ``True`` only when every output directory is complete.
    """
    return all(
        scene_pc_complete(
            point_cloud_scene_folder(output_config, scene_id),
            pc_per_scene,
        )
        for output_config in output_configs
    )


def save_single_view_outputs(
    output_configs,
    scene_id,
    view_matrix,
    rgb_image,
    depth_image,
    all_pc,
    mask,
    func_config,
):
    """Save legacy one-view-per-file render outputs.

    Args:
        output_configs: List of point-cloud outputs to save.
        scene_id: Scene id used to resolve output folders.
        view_matrix: Camera-to-world matrices for all rendered views.
        rgb_image: Optional RGB image tensor.
        depth_image: Optional depth image tensor.
        all_pc: Optional world-frame point-cloud tensor per view.
        mask: Optional valid-depth mask per view.
        func_config: Render config controlling which outputs are saved.

    Returns:
        None.
    """
    for output_config in output_configs:
        scene_pc_folder = point_cloud_scene_folder(output_config, scene_id)
        os.makedirs(scene_pc_folder, exist_ok=True)
        for b in range(view_matrix.shape[0]):
            data_id = str(b).zfill(2)
            np.save(
                os.path.join(scene_pc_folder, f"cam_ex_{data_id}.npy"),
                view_matrix[b].cpu().numpy(),
            )
            if func_config.save_rgb:
                cv2.imwrite(
                    os.path.join(scene_pc_folder, f"rgb_{data_id}.png"),
                    rgb_image[b].cpu().numpy() * 255,
                )
            if func_config.save_depth:
                cv2.imwrite(
                    os.path.join(scene_pc_folder, f"depth_{data_id}.png"),
                    depth_image[b].cpu().numpy(),
                )
            if func_config.save_pc:
                pc = sample_point_cloud_with_method(
                    all_pc[b, mask[b]],
                    output_config,
                    func_config.max_point_num,
                )
                np.save(
                    os.path.join(scene_pc_folder, f"partial_pc_{data_id}.npy"),
                    pc.cpu().numpy().astype(np.float16),
                )


def save_fused_view_outputs(
    output_configs,
    scene_id,
    pc_id,
    view_slice,
    view_matrix,
    rgb_image,
    depth_image,
    all_pc,
    mask,
    func_config,
):
    """Fuse a group of rendered views and save one point cloud sample.

    Args:
        output_configs: List of point-cloud outputs to save.
        scene_id: Scene id used to resolve output folders.
        pc_id: Index of the fused point-cloud sample within the scene.
        view_slice: Slice selecting the views that belong to this fused sample.
        view_matrix: Camera-to-world matrices for all rendered views.
        rgb_image: Optional RGB image tensor.
        depth_image: Optional depth image tensor.
        all_pc: World-frame point-cloud tensor per rendered view.
        mask: Valid-depth mask per rendered view.
        func_config: Render config controlling output behavior.

    Returns:
        None.
    """
    data_id = str(pc_id).zfill(2)
    selected_view_matrix = view_matrix[view_slice]
    fused_pc = None

    if func_config.save_pc:
        fused_pc = torch.cat(
            [
                all_pc[view_id, mask[view_id]]
                for view_id in range(view_slice.start, view_slice.stop)
            ],
            dim=0,
        )

    for output_config in output_configs:
        scene_pc_folder = point_cloud_scene_folder(output_config, scene_id)
        os.makedirs(scene_pc_folder, exist_ok=True)
        np.save(
            os.path.join(scene_pc_folder, f"cam_ex_{data_id}.npy"),
            selected_view_matrix.cpu().numpy(),
        )

        if func_config.save_pc:
            sampled_pc = sample_point_cloud_with_method(
                fused_pc,
                output_config,
                func_config.max_point_num,
            )
            np.save(
                os.path.join(scene_pc_folder, f"partial_pc_{data_id}.npy"),
                sampled_pc.cpu().numpy().astype(np.float16),
            )

        for local_view_id, view_id in enumerate(
            range(view_slice.start, view_slice.stop)
        ):
            view_id_str = str(local_view_id).zfill(2)
            if func_config.save_pc and cfg_get(
                func_config,
                "save_individual_views",
                False,
            ):
                view_pc = sample_point_cloud_with_method(
                    all_pc[view_id, mask[view_id]],
                    output_config,
                    func_config.max_point_num,
                )
                np.save(
                    os.path.join(scene_pc_folder, f"view_pc_{data_id}_{view_id_str}.npy"),
                    view_pc.cpu().numpy().astype(np.float16),
                )
            if func_config.save_rgb:
                cv2.imwrite(
                    os.path.join(scene_pc_folder, f"rgb_{data_id}_{view_id_str}.png"),
                    rgb_image[view_id].cpu().numpy() * 255,
                )
            if func_config.save_depth:
                cv2.imwrite(
                    os.path.join(scene_pc_folder, f"depth_{data_id}_{view_id_str}.png"),
                    depth_image[view_id].cpu().numpy(),
                )


def batch_warp_render(configs, scene_cfg_path_lst, gpu_id):
    func_config = configs.func
    device = f"cuda:{gpu_id}"
    with wp.ScopedDevice(device):
        point_cloud_outputs = get_point_cloud_outputs(func_config)
        for output_config in point_cloud_outputs:
            os.makedirs(point_cloud_output_root(output_config), exist_ok=True)

        renderer_cache = {}
        cam_in_saved_paths = set()

        def get_renderer(n_cols, n_rows):
            """Return a cached renderer for the requested tile layout.

            Args:
                n_cols: Number of tiled rendering columns.
                n_rows: Number of tiled rendering rows.

            Returns:
                A cached or newly created ``WarpRender`` instance.
            """
            renderer_key = (n_cols, n_rows)
            if renderer_key not in renderer_cache:
                with gpu_render_lock(gpu_id):
                    renderer_cache[renderer_key] = build_warp_renderer(
                        device,
                        func_config,
                        n_cols=n_cols,
                        n_rows=n_rows,
                    )
                for output_config in point_cloud_outputs:
                    output_folder = point_cloud_output_root(output_config)
                    if output_folder in cam_in_saved_paths:
                        continue
                    np.save(
                        os.path.join(output_folder, "cam_in.npy"),
                        renderer_cache[renderer_key].projection_matrixs[0],
                    )
                    cam_in_saved_paths.add(output_folder)
            return renderer_cache[renderer_key]

        for scene_cfg_path in tqdm(scene_cfg_path_lst):
            scene_cfg = np.load(scene_cfg_path, allow_pickle=True).item()
            scene_id = scene_cfg["scene_id"]

            obj_mesh = scene_cfg2mesh(scene_cfg, scene_cfg_path)
            camera_cfg = None
            for camera_name, ccc in func_config["camera"].items():
                if camera_name in scene_cfg["scene_id"]:
                    camera_cfg = dict(ccc)
                    break
            assert camera_cfg is not None

            if camera_cfg.get("fuse_pc", False):
                pc_per_scene = int(
                    camera_cfg.get(
                        "pc_per_scene",
                        cfg_get(func_config, "pc_per_scene", 4),
                    )
                )
                views_per_pc = int(
                    camera_cfg.get(
                        "views_per_pc",
                        cfg_get(func_config, "views_per_pc", 3),
                    )
                )
                render_strategy = camera_cfg.get(
                    "fused_render_strategy",
                    cfg_get(func_config, "fused_render_strategy", "per_pc"),
                )
                if render_strategy not in ("per_pc", "all_at_once"):
                    raise ValueError(
                        f"Unsupported fused render strategy: {render_strategy}"
                    )

                if configs.skip and scene_outputs_complete(
                    point_cloud_outputs,
                    scene_id,
                    pc_per_scene,
                ):
                    continue

                if render_strategy == "per_pc":
                    renderer = get_renderer(views_per_pc, 1)
                    for pc_id in range(pc_per_scene):
                        per_pc_camera_cfg = dict(camera_cfg)
                        per_pc_camera_cfg["pc_per_scene"] = 1
                        per_pc_camera_cfg["views_per_pc"] = views_per_pc
                        per_pc_camera_cfg["sample_num"] = views_per_pc
                        camera_view_matrix = get_camera_matrix(per_pc_camera_cfg)
                        (
                            view_matrix,
                            rgb_image,
                            depth_image,
                            all_pc,
                            mask,
                        ) = render_scene_views(
                            renderer, obj_mesh, camera_view_matrix, func_config, gpu_id
                        )
                        save_fused_view_outputs(
                            point_cloud_outputs,
                            scene_id,
                            pc_id,
                            slice(0, views_per_pc),
                            view_matrix,
                            rgb_image,
                            depth_image,
                            all_pc,
                            mask,
                            func_config,
                        )
                else:
                    renderer = get_renderer(views_per_pc, pc_per_scene)
                    camera_cfg["pc_per_scene"] = pc_per_scene
                    camera_cfg["views_per_pc"] = views_per_pc
                    camera_cfg["sample_num"] = pc_per_scene * views_per_pc
                    camera_view_matrix = get_camera_matrix(camera_cfg)
                    (
                        view_matrix,
                        rgb_image,
                        depth_image,
                        all_pc,
                        mask,
                    ) = render_scene_views(
                        renderer, obj_mesh, camera_view_matrix, func_config, gpu_id
                    )
                    for pc_id in range(pc_per_scene):
                        view_start = pc_id * views_per_pc
                        view_end = view_start + views_per_pc
                        save_fused_view_outputs(
                            point_cloud_outputs,
                            scene_id,
                            pc_id,
                            slice(view_start, view_end),
                            view_matrix,
                            rgb_image,
                            depth_image,
                            all_pc,
                            mask,
                            func_config,
                        )
                continue

            renderer_config = cfg_get(func_config, "renderer", {}) or {}
            renderer = get_renderer(
                cfg_get(renderer_config, "n_cols", 2),
                cfg_get(renderer_config, "n_rows", 2),
            )
            batch = renderer.num_tiles

            if configs.skip and scene_outputs_complete(
                point_cloud_outputs,
                scene_id,
                batch,
            ):
                continue

            camera_cfg["sample_num"] = batch
            camera_view_matrix = get_camera_matrix(camera_cfg)
            view_matrix, rgb_image, depth_image, all_pc, mask = render_scene_views(
                renderer,
                obj_mesh,
                camera_view_matrix,
                func_config,
                gpu_id,
            )
            save_single_view_outputs(
                point_cloud_outputs,
                scene_id,
                view_matrix,
                rgb_image,
                depth_image,
                all_pc,
                mask,
                func_config,
            )
