from __future__ import annotations

from functools import partial
from typing import Callable

import mitsuba as mi
import numpy as np

from .contexts import KernelContext
from .kernel import (
    DictParameter,
    SceneParameter,
    SearchSceneParameter,
)
from .scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
    XYGrid,
)


def _prepare_grid(
    eval_grid, ctx, spectral_index, unit, dtype, extra_kwargs, shape_xyz
) -> np.ndarray:
    """
    Evaluate, convert and broadcast a grid to the target shape ``(x, y, z)``.

    Scalars and 1-D z-column arrays are broadcast. Any other shape mismatch
    raises a ``ValueError``.
    """
    grid = eval_grid(ctx.si if spectral_index else ctx, **extra_kwargs)
    if unit:
        grid = grid.m_as(unit)
    assert np.size(grid) > 0
    grid = np.asarray(grid, dtype=dtype)
    if grid.size == 1:
        return np.broadcast_to(grid, shape_xyz)
    if np.squeeze(grid).shape == (shape_xyz[2],):
        return np.broadcast_to(grid.reshape(1, 1, -1), shape_xyz)
    if grid.shape != shape_xyz:
        raise ValueError(f"Invalid grid shape, expected {shape_xyz}, got {grid.shape}")
    return grid


def _finalize_grid(
    grid: np.ndarray, extra_dim: bool, to_mi: bool
) -> np.ndarray | mi.VolumeGrid:
    """
    Optionally append a trailing dimension and convert to a Mitsuba
    ``VolumeGrid``.
    """
    if extra_dim:
        grid = grid.reshape(*grid.shape, 1)
    if to_mi:
        return mi.VolumeGrid(grid)
    return grid


def make_volume_grid(
    geometry: SceneGeometry,
    ctx: KernelContext,
    eval_grid=None,
    dtype=None,
    unit=None,
    spectral_index=True,
    to_mi=True,
    extra_dim=False,
    extra_kwargs=None,
) -> np.ndarray | mi.VolumeGrid:
    """
    Evaluate a grid function and reshape it to the layout expected by Mitsuba
    for the given geometry.

    For :class:`.PlaneParallelGeometry` the grid is transposed from
    ``(x, y, z)`` to ``(z, y, x)``. For :class:`.SphericalShellGeometry` the
    ``(x, y, z)`` layout is used as-is.
    """
    shape = (
        *(geometry.xy_resolution if isinstance(geometry, XYGrid) else (1, 1)),
        geometry.zgrid.n_layers,
    )
    grid = _prepare_grid(
        eval_grid, ctx, spectral_index, unit, dtype, extra_kwargs, shape
    )
    if isinstance(geometry, PlaneParallelGeometry):
        grid = grid.T
    return _finalize_grid(grid, extra_dim, to_mi)


class _partial(partial):
    """
    A :class:`functools.partial` subclass with a human-readable ``__repr__``
    and a ``get_bound_instance`` helper that returns the object to which
    ``eval_grid`` is bound.
    """

    def __repr__(self):
        cls_name = self.get_bound_instance().__class__.__name__
        return f"partial({self.func.__name__}, {cls_name}, extra_parameters={list(self.keywords)})"

    def get_bound_instance(self):
        """Return the instance to which the ``eval_grid`` method is bound."""
        return self.keywords["eval_grid"].__self__


def _postprocess_template(
    geometry: SceneGeometry,
    partial_factory: Callable,
    filter_type_kw: str,
    wrap_mode_kw: str,
    include_to_world: bool,
) -> dict:
    """
    Build a Mitsuba kernel dict for a gridvolume plugin, wrapping it in a
    ``sphericalcoordsvolume`` for :class:`.SphericalShellGeometry`.
    """
    to_world = (
        {"to_world": geometry.atmosphere_volume_to_world} if include_to_world else {}
    )
    gridvolume = {
        "type": "gridvolume",
        "grid": DictParameter(partial_factory),
        "filter_type": filter_type_kw,
        "wrap_mode": wrap_mode_kw,
    }
    if isinstance(geometry, SphericalShellGeometry):
        return {
            "type": "sphericalcoordsvolume",
            "volume": gridvolume,
            **to_world,
            "rmin": geometry.atmosphere_volume_rmin,
        }
    return {**gridvolume, **to_world}


def generate_gridvolume(
    geometry: SceneGeometry,
    eval_grid: Callable,
    flag=None,
    search: SearchSceneParameter | None = None,
    filter_type=None,
    wrap_mode=None,
    spectral_index=True,
    unit=None,
    dtype=np.float32,
    extra_kwargs=None,
    include_to_world=True,
) -> dict | SceneParameter:
    """
    Generate a gridvolume kernel dict or a scene parameter update object.

    When ``flag`` is ``None``, a Mitsuba kernel dict suitable for scene
    construction is returned. When ``flag`` is set, a :class:`.SceneParameter`
    intended for scene parameter updates is returned instead.

    Parameters
    ----------
    geometry : .SceneGeometry
        Scene geometry driving the grid shape and coordinate layout.

    eval_grid : callable
        A bound method returning the grid data. It must accept a
        :class:`.KernelContext` (or spectral index if ``spectral_index`` is ``True``)
        as its first argument, followed by any ``extra_kwargs``.

        Expected return shape (in ``(x, y, z)`` order):

        * ``()`` or ``(1,)`` — scalar, broadcast to full shape.
        * ``(n_z,)`` — 1-D vertical profile, broadcast along x and y.
        * ``(n_x, n_y, n_z)`` — full 3-D grid.

    flag : object, optional
        Update flag passed to :class:`.SceneParameter`. If ``None`` (default),
        a kernel dict is returned.

    search : .SearchSceneParameter, optional
        Parameter lookup strategy forwarded to :class:`.SceneParameter`.
        Only used when ``flag`` is set.

    filter_type : str, optional
        Mitsuba filter type string. Defaults to ``str(geometry.filter_type)``.

    wrap_mode : str, optional
        Mitsuba wrap mode string. Defaults to ``str(geometry.wrap_mode)``.

    spectral_index : bool, optional, default: False
        If ``True``, pass ``ctx.si`` instead of ``ctx`` to ``eval_grid``.

    unit : pint.Unit, optional
        If set, convert grid values to this unit before processing.

    dtype : numpy.dtype, optional, default: numpy.float32
        NumPy dtype of the resulting grid buffer.

    extra_kwargs : dict, optional
        Additional keyword arguments forwarded to ``eval_grid``.

    include_to_world : bool, optional, default: True
        Whether to include the ``to_world`` transform in the kernel dict.
        Only used when ``flag`` is ``None``.

    Returns
    -------
    dict
        Mitsuba kernel dict, when ``flag`` is ``None``.

    .SceneParameter
        Scene parameter update object, when ``flag`` is set.

    Examples
    --------
    Generate a kernel dict for a plane-parallel scene:

    .. code-block:: python

        generate_gridvolume(geometry, my_component.eval_sigma_t)

    Generate a scene parameter for spectral updates:

    .. code-block:: python

        generate_gridvolume(
            geometry,
            my_component.eval_sigma_t,
            flag=UpdateFlags.SPECTRAL,
            search=SearchSceneParameter(...),
        )
    """
    pf = _partial(
        make_volume_grid,
        geometry,
        eval_grid=eval_grid,
        dtype=dtype,
        unit=unit,
        spectral_index=spectral_index,
        extra_dim=flag is not None,
        to_mi=flag is None,
        extra_kwargs=extra_kwargs or {},
    )
    if flag is not None:
        return SceneParameter(pf, flag, search=search)
    return _postprocess_template(
        geometry,
        pf,
        filter_type or str(geometry.filter_type),
        wrap_mode or str(geometry.wrap_mode),
        include_to_world,
    )
