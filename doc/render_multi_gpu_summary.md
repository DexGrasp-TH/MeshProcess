# Render Concurrency Summary

## Problem

When `thread_per_gpu: 1` in `src/config/func/render.yaml`, rendering worked correctly.
When `thread_per_gpu` was increased, especially to values like `10`, the saved outputs became incorrect.
One visible symptom was that saved point clouds had the wrong scale.

The issue appeared when running:

```bash
script/AnyScaleGrasp_DGN.sh
```

## What Was Observed

- Single render worker on a GPU: stable results
- Multiple render workers on the same GPU: corrupted render outputs
- Multiple GPUs: appears to work correctly

This strongly suggests the bug is tied to **multiple render processes sharing one GPU**, not to the scene configuration itself.

## Relevant Code Path

The rendering pipeline is:

1. `script/AnyScaleGrasp_DGN.sh`
2. `src/main.py`
3. `src/func/render.py`
4. `src/util/warp_render.py`

Key facts from the implementation:

- `src/func/render.py` expands the GPU list with:

  ```python
  gpu_lst = list(cfg.func.gpu_lst) * cfg.func.thread_per_gpu
  ```

  So if `gpu_lst=[2]` and `thread_per_gpu=10`, the code launches 10 processes, all targeting `cuda:2`.

- `src/util/warp_render.py` creates a `warp.render.OpenGLRenderer` inside each worker.
- The renderer runs in headless OpenGL mode and reads back depth/RGB buffers to CUDA tensors.
- The point cloud is reconstructed from the returned depth image.

## Likely Root Cause

The most likely failure mode is **unsafe concurrent OpenGL/CUDA rendering on the same GPU**.

Even though the code uses separate Python processes, those processes still compete for:

- the same GPU
- the same OpenGL stack
- the same headless display/context path
- CUDA/OpenGL interop resources used during pixel readback

This kind of setup often fails in a bad way:

- not a clean crash
- not an obvious exception
- but subtly wrong rendered buffers

If the depth buffer is corrupted, then `depth_to_point_cloud()` will reconstruct the wrong geometry, which matches the observed wrong-scale point clouds.

## Why Multi-GPU Works

When rendering is spread across multiple GPUs, each GPU effectively gets its own rendering workload and device state.
That avoids the high-contention case where many OpenGL renderers all target the same GPU at the same time.

So the fact that multi-GPU works is an important clue:

- scene configs are probably fine
- scale values in the scene are probably fine
- point cloud reconstruction math is probably fine
- the instability is likely in **same-GPU concurrent rendering**

## Changes Made

### 1. Use `spawn` instead of default `fork`

In `src/func/render.py`, worker creation was changed from the default multiprocessing behavior to:

```python
mp_ctx = multiprocessing.get_context("spawn")
```

Why this matters:

- On Linux, the default multiprocessing start method is usually `fork`
- `fork` can inherit partially initialized CUDA/OpenGL state into child processes
- graphics or GPU code is often not fork-safe

Using `spawn` starts a clean interpreter process and reduces inherited-state corruption.

### 2. Bind the renderer explicitly to the target device

In `src/util/warp_render.py`, `OpenGLRenderer` is now created with:

```python
device=device
```

This makes the intended GPU selection explicit instead of relying on implicit preferred-device behavior.

### 3. Add a per-GPU render lock

A file lock was added around renderer creation and render/readback sections:

```python
with gpu_render_lock(gpu_id):
    ...
```

Why this matters:

- multiple workers may still exist for one GPU
- but only one worker is allowed to enter the fragile OpenGL render/readback section at a time
- this serializes the unsafe part while keeping the rest of the worker structure intact

Practically, this means:

- same-GPU workers no longer render concurrently
- corruption risk is reduced
- throughput may still not improve much for `thread_per_gpu > 1` on a single GPU

### 4. Check render readback success

`OpenGLRenderer.get_pixels()` returns a success flag.
Previously that flag was ignored.
The code now raises an error if pixel readback fails, and retries a few times.

This is important because silent readback failure can otherwise produce bad depth buffers, which then become bad point clouds.

## What This Means Operationally

### Safe mental model

For this renderer, assume:

- **one active OpenGL render/readback per GPU is safe**
- **many simultaneous OpenGL render/readbacks on one GPU are unsafe**

### Recommended usage

If you want higher throughput, prefer:

- more GPUs
- one render worker per GPU

Avoid expecting speedup from large `thread_per_gpu` values on a single GPU.
For this codebase and renderer stack, that setting is more likely to create contention than useful parallelism.

## Why Wrong Scale Happens Instead of a Crash

The point cloud is computed from the depth image:

```python
x = (self.help_xx - self.cx) * depth_image / self.fx
y = -(self.help_yy - self.cy) * depth_image / self.fy
...
world_coords = camera_coords @ self.view_matrix
```

This math is straightforward.
If `depth_image` is wrong, the resulting 3D points will also be wrong.

So a scale error does not require the scene scale itself to be wrong.
It is enough for the rendered depth buffer to be corrupted or stale.

## Practical Conclusion

The main bug is not about object scale parsing or scene generation.
It is about **render concurrency on the same GPU**.

The reason multi-GPU works is that it avoids the unsafe contention pattern.

The code changes made so far reduce the risk by:

- avoiding `fork` for render workers
- binding the renderer to the intended device
- serializing same-GPU render/readback
- failing loudly on readback errors instead of silently saving corrupted outputs

## Best Long-Term Design

The cleanest architecture is:

- one render process per GPU
- a queue of scenes assigned to that process
- internal batching inside that one process

That is usually better than launching many independent OpenGL renderers against the same GPU.

## Files Involved

- `src/func/render.py`
- `src/util/warp_render.py`
- `test/print_object_pc_bbox.py`

## Short Version

`thread_per_gpu > 1` was effectively asking multiple OpenGL/CUDA render workers to share one GPU concurrently.
That corrupted depth readback and produced incorrect point clouds.
Multi-GPU works because it avoids that contention pattern.
