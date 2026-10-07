# Vendored `xr_nav`

The dimOS fork's spatial pipeline script (`mac_iphone_spatial_foxglove.py`)
imports an `xr_nav` package unconditionally. In the fork that package lives in
a git submodule; this directory vendors exactly the modules the pipeline
reaches, so the fork can be cloned without it.

- **Source:** the fork's `xr-nav` submodule at commit `b96c95d8`
  (`b96c95d83371c042cd23763bec0f1c8cf3716dc6`), the commit pinned by fork
  revision `0d39b5ad4`.
- **Copied verbatim**, apart from two docstring sentences in `map_io.py` that
  named an unrelated downstream project. The set is the transitive import
  closure of every `xr_nav` import in the pipeline script, including the lazy
  ones inside functions.
- **Wiring:** [`station/dimos_bridge/server.py`](../dimos_bridge/server.py)
  prepends this directory to `sys.path` right after loading the fork script,
  so this copy is used whether the fork's submodule directory is empty or
  populated.

| Module | What the pipeline uses it for |
| --- | --- |
| `cli_args` | map-I/O, keyframe and relocalization argument groups |
| `voxel_map` | sparse hash voxel grid with confidence-weighted centroids and raycast free-space clearing |
| `map_io` | map bundle writer/loader (point cloud, voxel state, provenance) |
| `keyframe` | keyframe selection by pose delta |
| `keyframe_recorder` | saving keyframes alongside a map |
| `icp` | frame-to-map ICP registration and drift tracking (on by default in `server.py`) |
| `reference_map` | loading a saved map as a relocalization reference |
| `relocalize_live` | live relocalization against a reference map |
| `scale_align` | scale recovery for *relative* depth models (Depth-Anything-3 relative only) |
| `mv_window` | multi-view refinement window (`--mv-window`) |

Dependencies are all dimOS core dependencies already: `numpy`, `numba`,
`scipy`, `open3d`, `opencv`.

Not vendored: the Depth-Anything-3 source the submodule also carried. The
default `--depth depthpro` path doesn't use it; for the `da3*` depth models,
install [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-3)
from upstream.

## Tests

```bash
python -m pytest station/vendor/tests -q
```

28 tests covering ICP registration, map provenance round-trips, the multi-view
window, and live relocalization. `test_relocalize_live.py` has one test that
also needs `import dimos`; set `DIMOS_DIR` to your fork checkout to include it.
