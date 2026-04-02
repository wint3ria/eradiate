# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext eradiate

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import eradiate

eradiate.set_mode("mono_polarized")
integrator = "eovolpath"

# %%
from eradiate.units import unit_registry as ureg

# %% [markdown]
# ## I. Coordinates definition

# %% [markdown]
#  - The ZGrid class is removed in favor of two 3D coordinate grid classes handling plane parallel and spherical shell geometries
#  - The `grid` module defines these class at the top level of the eradiate package (i.e. `eradiate.grid`). The ZGrid class used to be defined within the radprops module
#  - Grids provide a mappring from geometric coordinates to the internal grid data storage facilities in Eradiate, always indexed on unsigned 3D array positions X, Y and Z. 
#  - `eradiate.grid` centralizes the logic for coordinate mapping, a useful property for future developments.

# %%
from eradiate.grid import GridCoords, PlaneParallelGridCoords, SphericalShellGridCoords

# %%
z_grid = GridCoords.make_default()
z_grid

# %%

# %%
grid_test = z_grid.resampled_to_cell_size(0.5 * ureg.hm, 0.5 * ureg.hm)
grid_test = grid_test.cropped(2 * ureg.hm, 3 * ureg.hm)
grid_test

# %%
grid_3d = PlaneParallelGridCoords(
    levels=np.linspace(0, 100, 101) * ureg.m,
    edges_x=grid_test.edges_x,
    edges_y=grid_test.edges_y,
)

# %%
grid_3d

# %% [markdown]
# The PlaneParallelGridCoords defines a mapping from cartesian coordinates space of the scene to 3D memory array cells according to its discretization.

# %%
SphericalShellGridCoords(
    levels=np.linspace(0, 100, 101) * ureg.meter,
    azimuths=np.linspace(175, 185, 11) * ureg.degree,
    colatitudes=np.linspace(85, 95, 11) * ureg.degree,
)

# %%
grid_1d = grid_3d.single_column()

# %% [markdown]
# ## II. Geometry

# %%
from eradiate.scenes.geometry import PlaneParallelGeometry

geometry_3d = PlaneParallelGeometry(grid=grid_3d, toa_altitude=grid_3d.levels[-1])
geometry_3d

# %%
grid_1d = PlaneParallelGridCoords(
    levels=np.linspace(0, 100, 101) * ureg.m,
    edges_x=[-5e5, 5e5] * ureg.km,
    edges_y=[-5e5, 5e5] * ureg.km,
)
grid_1d

# %%
geometry_1d = PlaneParallelGeometry(grid=grid_1d, toa_altitude=grid_1d.levels[-1])
geometry_1d

# %% [markdown]
# ## II. Defining a 3D Atmosphere

# %%
from eradiate.scenes.atmosphere import MolecularAtmosphere, ParticleLayer

atmosphere_1d = MolecularAtmosphere(geometry=geometry_1d)
atmosphere_3d = MolecularAtmosphere(geometry=geometry_3d)

# %%
from eradiate.experiments import AtmosphereExperiment

exp_1d = AtmosphereExperiment(atmosphere=atmosphere_1d)
exp_3d = AtmosphereExperiment(atmosphere=atmosphere_3d)

res_1d = eradiate.run(exp_1d, spp=100000)
res_3d = eradiate.run(exp_3d, spp=100000)

assert np.allclose(res_1d.radiance.item(), res_3d.radiance.item(), rtol=1e-2)

# %%
# TODO : complex composition of heterogeneous atmosphere

# %% [markdown]
# ## III. 3D Particle Layers

# %% [markdown]
# Spatial variations are defined mostly using a column major process. Five spatial variation implementations are provided:
#  - Optical thickness $\tau_{ref}$ variation for each column
#  - Spatially varying Z extents along the X or Y axes
#  - Horizontal distributions evaluated along the X or Y axes
#  - Spatially varying Z-columns distribution parameters along the X or Y axes
#  - Distribution compositions

# %% [markdown]
# ### 1. $\tau_{ref}$ variations w.r.t horizontal coordinates

# %%

# %% [markdown]
# Particle layers can now vary along the horizontal extents. 

# %%
ParticleLayer(geometry=geometry_3d).x_extent

# %%
ParticleLayer(geometry=geometry_3d).top

# %%
tau_max = 5

x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 6))
d = np.sqrt(x * x + y * y)
sigma, mu = 1.0, 0.0
g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
tau_ref = np.ones((4, 6))
tau_ref *= g.T
tau_ref = tau_ref * tau_max

i = np.repeat(np.arange(tau_ref.shape[0]).reshape(1, -1), tau_ref.shape[1], axis=0)
j = np.repeat(np.arange(tau_ref.shape[1]).reshape(-1, 1), tau_ref.shape[0], axis=1)
mask = i % 2 == j % 2
tau_ref = np.where(mask.T, tau_ref, 0)

cbar = plt.colorbar(plt.imshow(tau_ref, cmap="Greys"), label="$\\tau_{ref}$")
plt.title("Particles total column optical thickness $\\tau_{ref}$")
plt.show()
plt.close()

# %%
roi_size = 7.0 * ureg.km

# %%
from eradiate.scenes.surface import BasicSurface

checker = {"type": "checkerboard"}

surface = BasicSurface(
    **{
        "shape": {
            "type": "rectangle",
        },
        "bsdf": checker,
    }
)
target = [0, 0, 0] * ureg.m
surface

# %%
from eradiate.scenes.measure import PerspectiveCameraMeasure

origin = [0, 0.01, 250] * ureg.km
resolution = 1024
spp = 512

camera = PerspectiveCameraMeasure(
    id="camera",
    origin=origin,
    target=target,
    up=[0, 0, 1],
    film_resolution=(resolution, resolution),
    spp=spp,
    far_clip=1e7 * ureg.km,
    fov=0.2,
)


def plot_camera(res, vmax=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    res.squeeze(drop=True).plot.imshow(
        ax=ax,
        origin="upper",
        cmap="Greys_r",
        robust=True,
        add_colorbar=True,
        vmin=0,
        vmax=vmax,
    )
    ax.set_aspect(1)


# %%
extra_objs = {
    "mysurface": {
        "factory": "shape",
        "type": "rectangle",
        "center": [0, 0, 0],
        "edges": [grid_3d.total_length.m_as("m"), grid_3d.total_length.m_as("m")],
        "bsdf": {
            "type": "checkerboard",
            "scale_pattern": grid_3d.total_length / grid_3d.cell_length,
        },
    }
}

# %%
geometry_1d

# %%
particles_1d = ParticleLayer(
    tau_ref=tau_max, bottom=0, top=20 * ureg.m, geometry=geometry_1d
)
particles_1d

# %%
from eradiate.experiments import CanopyAtmosphereExperiment

exp_1d = CanopyAtmosphereExperiment(
    atmosphere=particles_1d,
    surface=None,
    measures=camera,
    illumination={"type": "directional", "zenith": 30, "azimuth": 30},
    integrator={"type": integrator, "moment": True},
    extra_objects=extra_objs,
    geometry=geometry_1d,
)

# %%
# %%time

res_onedim = eradiate.run(exp_1d,spp=1)

# %%
plot_camera(res_onedim.radiance, vmax=0.7)
plt.show()
plt.close()
plot_camera(np.sqrt(res_onedim.radiance_var))
plt.show()
plt.close()
std = np.sqrt(res_onedim.radiance_var.values.ravel())
std2 = 2 * np.where(std < 0.1, std, np.nan)
plt.hist(std2, bins=50)
plt.show()
plt.close()

# %%
particles_tau2d = ParticleLayer(
    geometry=geometry_3d, tau_ref=tau_ref, bottom=0, top=20 * ureg.m
)
particles_tau2d

# %%
exp2 = AtmosphereExperiment(
    atmosphere=particles_tau2d,
    surface=None,
    measures=camera,
    integrator={"type": integrator, "moment": True},
    geometry=geometry_3d,
    illumination={"type": "directional", "zenith": 30, "azimuth": 30},
    extra_objects=extra_objs,
)

# %%
# %%time

res2 = eradiate.run(exp2)

# %%
plot_camera(res2.radiance, vmax=0.7)

# %%
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

plot_camera(res2.radiance, vmax=0.7, ax=ax1)
plot_camera((res2.radiance - res_onedim.radiance) / res_onedim.radiance * 100, ax=ax2)

# %%
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

plot_camera(np.sqrt(res2.radiance_var), ax=ax1)
plot_camera(np.sqrt(res2.radiance_var) / res2.radiance * 100, ax=ax2)

# %% [markdown]
# ### 2. Variation of layer geometry w.r.t horizontal coords
#
# TBD demo

# %% [markdown]
# ### 3. Variation of distribution properties w.r.t horizontal coords
#
# TBD demo

# %% [markdown]
# ### 4. Horizontal distributions
#
# TBD demo

# %% [markdown]
# ### 5. Distribution compositions
#
# TBD demo
#
# Clear formula for distribution fractions

# %% [markdown]
# ### 6. Spherical shell support
#
# TBD demo

# %% [markdown]
# ## IV. IPRT case C1: with particles

# %%
from eradiate.experiments import CanopyAtmosphereExperiment
from eradiate.scenes.geometry import WrapMode
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.measure import MultiPixelDistantMeasure, TargetRectangle
from eradiate.spectral import DeltaSRF
from test_3d_clouds import load_aerosol_data

# %%
w_ref = 800 * ureg.nm

# %%
c1_spp = 10000

# %%
c1_grid = PlaneParallelGridCoords.from_extent_and_resolution(
    levels=[0, 0.25] * ureg.km,
    extent_x=0.5 * ureg.km,
    extent_y=1e6 * ureg.km,  # large enough
    n_cells_x=2,
    n_cells_y=1,
)

# %%
c1_geom = PlaneParallelGeometry(
    grid=c1_grid,
    toa_altitude=c1_grid.levels[-1],
    wrap_mode=WrapMode.REPEAT,  # periodic boundary conditions
)

# %%
ds = load_aerosol_data("test_3d_clouds/watercloud.mie.cdf", "spherical", w_ref)
ds

# %%
c1_particle_layer = ParticleLayer(
    geometry=c1_geom,
    tau_ref=np.asarray([2, 18]).reshape(2, 1),
    dataset=ds,
    w_ref=w_ref,
)

# %%
c1_geometries = pd.DataFrame(
    columns=["sza", "z", "vza", "vaa"],
    index=np.arange(1, 11, 1),
    data=[
        [0, 0, 60, 0],
        [60, 0, 0, 0],
        [60, 0, 30, 0],
        [60, 0, 30, 180],
        [0, 0.25, 180, 0],
        [0, 0.25, 140, 0],
        [0, 0.25, 120, 0],
        [60, 0.25, 180, 0],
        [60, 0.25, 120, 0],
        [20, 0.25, 120, 135],
    ],
).T

# %%
c1_targets = c1_geometries.apply(
    lambda x: TargetRectangle(
        xmin=-2.5 * ureg.km,
        xmax=2.5 * ureg.km,
        ymin=-2.5 * ureg.km,
        ymax=2.5 * ureg.km,
        z=x.z * ureg.km,
    )
).rename("target")

# %%
c1_illuminations = c1_geometries.apply(
    lambda x: DirectionalIllumination(
        zenith=x.sza * ureg.degree,
        azimuth=180 * ureg.degree,
    )
).rename("illumination")

# %%
c1_measures = (
    pd.concat([c1_geometries.T, c1_targets], axis=1)
    .T.apply(
        lambda x: MultiPixelDistantMeasure.from_angles(
            angles=(np.deg2rad(x.vza), np.deg2rad(x.vaa)),
            target=x.target,
            film_resolution=[32, 16],
            srf=DeltaSRF(w_ref),
            id=str(x.name),
        )
    )
    .rename("measure")
)

# %%
c1_experiments = c1_illuminations.apply(
    lambda x: CanopyAtmosphereExperiment(
        atmosphere=c1_particle_layer,
        surface={"type": "lambertian", "reflectance": 0.0},
        measures=c1_measures.tolist(),
        illumination=x,
        geometry=c1_geom,
        integrator={"type": integrator, "moment": True},
    )
).rename("experiment")

# %%
# %%time

c1_results = {}

for ill_key in c1_experiments.index:
    res = eradiate.run(c1_experiments[ill_key], spp=c1_spp)
    for view_key in res:
        res
    break

# %%
import glob

idir = eradiate.data.asset_manager.info()["install_dir"]
glob.glob(str(idir / "aerosol/*.nc"))

# %%
ds

# %%
for k in res:
    res[k].radiance.plot(figsize=(8, 4))
    plt.show()
    plt.close()
    plt.errorbar(
        res[k].x_index,
        res[k].radiance.isel(y_index=8).values.ravel(),
        np.sqrt(res[k].radiance_var.isel(y_index=8)).values.ravel() * 2,
    )
    plt.show()
    plt.close()

    break

# %% [markdown]
# ## V. IPRT case C2: with particles

# %% [markdown]
# ### 1. Setup

# %%
c2_grid = PlaneParallelGridCoords.from_extent_and_resolution(
    levels=np.linspace(0, 5, 6) * ureg.km,
    extent_x=7 * ureg.km,
    extent_y=7 * ureg.km,
    n_cells_x=7,
    n_cells_y=7,
)
c2_geom = PlaneParallelGeometry(grid=c2_grid, toa_altitude=c2_grid.levels[-1])
c2_spp = int(1e6)
c2_w_ref = 800 * ureg.nm
c2_tau_ref = 10
c2_surface_albedo = 0.2
c2_epsilon_dist = 1e-3 * ureg.m
c2_particle_layer = ParticleLayer(
    geometry=c2_geom,
    bottom=2 * ureg.km,
    top=3 * ureg.km,
    x_extent={"extent_min": -0.5 * ureg.km, "extent_max": 0.5 * ureg.km},
    y_extent={"extent_min": -0.5 * ureg.km, "extent_max": 0.5 * ureg.km},
    w_ref=c2_w_ref,
    tau_ref=c2_tau_ref,
    dataset=ds,
)

# %%
c2_geometries = pd.DataFrame(
    columns=["sza", "z", "vza", "vaa"],
    index=np.arange(1, 10, 1),
    data=[
        [20, 0, 40, 0],
        [20, 0, 40, 60],
        [20, 0, 40, 120],
        [20, 0, 40, 180],
        [40, 5, 180, 0],
        [40, 5, 140, 0],
        [40, 5, 140, 60],
        [40, 5, 140, 120],
        [40, 5, 140, 180],
    ],
).T
c2_geometries

# %%
c2_targets = c2_geometries.apply(
    lambda x: TargetRectangle(
        xmin=-3.5 * ureg.km,
        xmax=3.5 * ureg.km,
        ymin=-3.5 * ureg.km,
        ymax=3.5 * ureg.km,
        z=x.z * ureg.km + c2_epsilon_dist,  # prevent clipping with surface at z=0
    )
).rename("target")

# %%
c2_illuminations = c2_geometries.apply(
    lambda x: DirectionalIllumination(
        zenith=x.sza * ureg.degree,
        azimuth=180 * ureg.degree,
    )
).rename("illumination")

# %%
c2_measures = (
    pd.concat([c2_geometries.T, c2_targets], axis=1)
    .T.apply(
        lambda x: MultiPixelDistantMeasure.from_angles(
            angles=(x.vza - 180, x.vaa),
            target=x.target,
            film_resolution=[70, 70],
            srf=DeltaSRF(w_ref),
            id=str(x.name),
            ray_offset=c2_epsilon_dist,
        )
    )
    .rename("measure")
)

# %%
c2_experiments = (
    pd.concat(
        [
            c2_illuminations,
            c2_measures,
        ],
        axis=1,
    )
    .T.apply(
        lambda x: CanopyAtmosphereExperiment(
            atmosphere=c2_particle_layer,
            surface={"type": "lambertian", "reflectance": c2_surface_albedo},
            measures=x.measure,
            illumination=x.illumination,
            geometry=c2_geom,
            integrator={"type": integrator, "moment": True, "stokes": True},
        )
    )
    .rename("experiment")
)

# %% [markdown]
# ### 2. Compute

# %%
# %%time

from tqdm.notebook import tqdm

with tqdm(c2_experiments, total=len(c2_experiments)) as pbar:

    def run_case(c):
        res = eradiate.run(c, spp=c2_spp)
        pbar.update()
        return res

    c2_results = c2_experiments.apply(run_case)

# %% [markdown]
# ### 3. Visualization

# %%
fig, axs = plt.subplots(9, 5, figsize=(20, 27))
cmaps = ["Blues_r", "coolwarm", "coolwarm", "coolwarm", "viridis"]

for i, key in enumerate(tqdm(c2_results.index)):
    case = c2_results[key]

    for j, s in enumerate(["I", "Q", "U", "V"]):
        ax = axs[i][j]
        cmap = cmaps[j]
        stokes_comp_em3 = case.sel(stokes=s).radiance.squeeze() * 1e3
        stokes_normed = stokes_comp_em3 / case.irradiance.squeeze()
        vms = dict()
        if s in ["Q", "U", "V"]:
            vmin = abs(stokes_normed.min().item())
            vmax = abs(stokes_normed.max().item())
            vmin, vmax = min(-vmin, -vmax), max(vmin, vmax)
            vms = dict(vmin=vmin, vmax=vmax)
        stokes_normed.plot(
            ax=ax,
            cmap=cmap,
            **vms,
        )
        ax.set_title(None)
        if i == 0:
            ax.set_title(s)
        if j == 0:
            ax.set_ylabel(f"case {key}")

    dlp_percent = case.dlp.squeeze().fillna(0) * 100
    dlp_percent.plot(ax=axs[i][4])
    axs[i][4].set_title(None)
    if i == 0:
        axs[i][4].set_title("DoP")

print("Output images...")

plt.tight_layout()
plt.savefig("c2_iprt_panel.pdf", format="pdf")
plt.show()
plt.close()

print("Done.")

# %% [markdown]
# ### 4. Comparison with Mystic

# %%
data = np.loadtxt("./test_3d_clouds/iprt_case_c2_mystic.dat")

# %%
mask = (
    (data[:, 0] == 1)  # case 1 no atmosphere, 2 atmosphere
    & (data[:, 1] == 40)  # theta0
    & (data[:, 2] == 5)  # zout
    & (data[:, 3] == 180)  # theta
    & (data[:, 4] == 0)  # phi
)

result_mystic = data[mask]

# %%
from test_3d_clouds import plot_stokes_from_ascii

# %%
plot_stokes_from_ascii(result_mystic * 1e3)

# %%
idx = ["i_x", "i_y"]
values = ["I", "Q", "U", "V", "Istd", "Qstd", "Ustd", "Vstd"]
cols = ["case", "theta_0", "z", "theta", "phi"]

mystic_df = pd.DataFrame(data=data, columns=cols + idx + values)
mystic_df = mystic_df[mystic_df.case == 1].pivot(columns=cols, index=idx, values=values)


# %%
def from_erd_ds_to_iprt_df(ds):
    df = ds.item().to_dataframe()
    df = df.reset_index()
    df["radiance"] /= df.irradiance
    df["radiance_std"] = np.sqrt(df.radiance_var) / df.irradiance
    df["theta"] = 180 - df.vza
    df["theta_0"] = df.sza
    df["i_x"] = df["x_index"] + 1
    df["i_y"] = df["y_index"] + 1
    df["phi"] = np.abs(df.vaa - 180)
    df = df.drop(
        [
            "vza",
            "vaa",
            "y",
            "x",
            "dlp",
            "irradiance",
            "x_index",
            "y_index",
            "w",
            "saa",
            "sza",
            "radiance_var",
        ],
        axis=1,
    )
    df = [
        x.rename(dict(radiance=c, radiance_std=f"{c}std"), axis=1).drop(
            ["stokes"], axis=1
        )
        for c, x in df.groupby("stokes")
    ]
    df = [x.set_index(["phi", "theta", "theta_0", "i_x", "i_y"]) for x in df]
    df = pd.concat(df, axis=1).reset_index()
    df["case"] = 1.0
    df["z"] = 5 if ds.index[0] > "4" else 0
    return df


# %%
erd_df = pd.concat(c2_results.to_frame().apply(from_erd_ds_to_iprt_df, axis=1).tolist())
erd_df = erd_df.pivot(columns=cols, index=idx, values=values)

# %%
assert mystic_df.shape == erd_df.shape


# %%
def rmse(v1, v2):
    return np.sqrt(np.mean((v1 - v2) ** 2))


# %%
def corr_metric(df1, df2, scmp, method="pearson"):
    return (
        df1[scmp]
        .T.reset_index()
        .T.corrwith(df2[scmp].T.reset_index().T, method=method)
        .to_frame()
        .set_index(df2[scmp].T.index)[0]
        .rename(scmp)
    )


# %%
cor = pd.concat(
    [corr_metric(erd_df, mystic_df, scmp) for scmp in ["I", "Q", "U", "V", "Istd"]],
    axis=1,
).reset_index(drop=True)
sns.heatmap(cor, cmap="coolwarm", vmin=-1, vmax=1, annot=True)

# %%
drmse = pd.concat(
    [
        corr_metric(erd_df, mystic_df, scmp, method=rmse)
        for scmp in ["I", "Q", "U", "V", "Istd"]
    ],
    axis=1,
).reset_index(drop=True)
sns.heatmap(drmse, cmap="Reds", vmin=0, annot=True)

# %% [markdown]
# ## VI. A proper cloud interface
#
# Using particle layers to fill Eradiate extinction coefficient grids is feasible, but lacks proper support for intrinsic cloud properties. It lets users handle important pre-processing steps at the core of the scope of Eradiate: it is not sufficient.
#
# Source cloud properties can be defined as follows:
#  - A description of clouds repartition in the 3D space
#  - An associated set of optical properties for the cloud species
#
# From the IPRT case C3 for instance:

# %%
import xarray as xr
watercloud = xr.load_dataset("test_3d_clouds/watercloud_670.mie.cdf")
cumulus = np.loadtxt("test_3d_clouds/cumulus.dat").reshape((-1, 5))
cumulus = pd.DataFrame(cumulus, columns=["ix", "iy", "iz", "ext", "Reff"])
cumulus[["ix", "iy", "iz"]] = cumulus[["ix", "iy", "iz"]].map(int)
cumulus = cumulus.set_index(["ix", "iy", "iz"])

# %%
grid_index = pd.MultiIndex.from_product([
    np.arange(1, 101, 1),
    np.arange(1, 101, 1),
    np.arange(1, 54, 1),
])
grid = pd.DataFrame(index=grid_index, columns=cumulus.columns, data=np.zeros((len(grid_index), 2)))
grid[grid.index.isin(cumulus.index)] = cumulus


# %%
def chunk_aggregate(arr, funcs, size=10):
    chunk_shape = tuple(max(1, s // size) for s in arr.shape)
    
    d0, d1, d2 = arr.shape
    c0, c1, c2 = chunk_shape

    trimmed = arr[
        :d0 - d0 % c0 if d0 % c0 else d0,
        :d1 - d1 % c1 if d1 % c1 else d1,
        :d2 - d2 % c2 if d2 % c2 else d2,
    ]

    t0, t1, t2 = trimmed.shape
    reshaped = trimmed.reshape(t0//c0, c0, t1//c1, c1, t2//c2, c2)

    out = funcs[0](reshaped, axis=1)
    out = funcs[1](out, axis=2)
    out = funcs[2](out, axis=3)

    return out


# %%
grid

# %%
from voxel_render import render_voxels
img = render_voxels(
    grid.ext.values.reshape(100, 100, 53),
    path         = "out2.png",
    width        = 1200,
    height       = 900,
    elev         = 30,
    azim         = -60,
    fov          = 45,
    dist_factor  = 2.5,
    voxel_color  = (0.1, 0.4, 0.6),
    bg_color     = (10, 10, 15),
    alpha_thresh = 0.05,
    title        = "extinction coefficient",
    verbose      = False,
)
img

# %% [markdown]
# The Z dimension of this grid is not regular. It must be evaluated against a plane parallel coord grid.

# %%
with open("test_3d_clouds/cumulus.dat") as f:
    zgrid = f.readlines()[1]
    zgrid = zgrid.split()[1:]
    zgrid = [float(z) for z in zgrid]
    zgrid = np.asarray(zgrid)
plt.plot(zgrid)

# %% [markdown]
# Clouds optical properties coords:
#
#  - wavelength
#  - reff
#  - veff (not present)
#  - ext
#
# Indexed properties:
#
#  - single scattering albedo
#  - refractive index
#  - density?
#  - extinction coef

# %%
watercloud.squeeze()

# %% [markdown]
# C3 also features aerosols

# %%
waso = xr.load_dataset("test_3d_clouds/waso_670.mie.cdf")
waso.squeeze()


# %% [markdown]
# ## VII. Refactoring
#
# ### 1. Geometry and gridcoords configuration
#
# This is WIP proposal, not implemented yet.
#
# Currently, Eradiate's API forces users to pass either a geometry object or a grid object to scene element constructors or member methods. This definition can result in a rather verbose API, and the need to add a certain level of boilerplate for setups using anything but the simplest default plane parallel geometry. Writing this boilerplate and ensuring it is always consistent is also error prone for the user.
#
# Additionally, the way the geometry to scene element relationship is handled in Eradiate is inconsistent. Many scene elements require a geometry at instanciation in order to initialize properly. However, the grid coordinates, inherent part of the geometry, can be passed to the object when performing eval-types of member methods. In some cases, only the grid of the member geometry is accepted. This behavior is unclear and the result of several modifications of the API without a clear and definitive modelisation of this relationship. The geometry being a repeated member of many different scene elements, pollutes logs. The eval-types of methods are ultimately brittle and may depend on the grid structure to terminate successfully.

# %% [raw]
# # Currently:
# geometry_3d = PlaneParallelGeometry(grid=grid_3d)
#
# exp = AtmosphereExperiment(
#     atmosphere=ParticleLayer(geometry=geometry, dataset=...), # <- repeated geom
#     measures=camera,
#     integrator={"type": integrator, "moment": True},
#     geometry=geometry                                         # <- repeated geom
# )
# # This repeated boilerplate issue is aggravated by the number of atmospheric components,
# # typically greater in a heterogeneous atmosphere.
#
# exp.atmosphere.eval_sigma_t() # ok
# exp.atmosphere.eval_sigma_t(geometry_3d.grid) # ok
# exp.atmosphere.eval_sigma_t(grid_3d_other) # may raise?

# %% [markdown]
# To address the issue, the following setup *can* be implemented:
#  - *Similarly* to how Eradiate handles a global mode as a *configuration* for all the generated scene elements, an additional geometry configuration *could* be implemented.
#  - This configuration would define the geometry used at runtime
#
# TBD develop proposal: geometry configures the scene elements. decoupling of ownership and membership. When possible, remove geom at init time. notion of dynamic default. support serialization through dask, necessary for MSP support typically

# %% [raw]
# # Future additional API:
# from eradiate.scenes.geometry import make_geometry
#
# with geometry_3d:
#     exp = AtmosphereExperiment(
#         atmosphere=ParticleLayer(dataset=...),
#         measures=camera,
#         integrator={"type": integrator, "moment": True},
#     )
#
# exp.atmosphere.eval_sigma_t() # ok
# exp.atmosphere.eval_sigma_t(geometry_3d.grid) # ok
# exp.atmosphere.eval_sigma_t(grid_3d) # ok
# exp.atmosphere.eval_sigma_t(grid_3d_other) # requires additional code review to ensure feasibility
#
# # ... or:
#
# from eradiate.scenes.geometry import set_default_geometry, get_default_geometry
#
# set_default_geometry(geometry_3d)
# exp = AtmosphereExperiment(
#     atmosphere=ParticleLayer(dataset=...),
#     measures=camera,
#     integrator={"type": integrator, "moment": True},
# )
#

# %% [markdown]
# The new contextmanager is a part of the solution. Under the hood, static methods allow to get the geometry configuring a scene element. set_default_geometry can be called multiple times in the lifetime of a process, and it is necessary to keep track of which geometry configures a given scene element. get_default_geometry is called by scene elements when they need to access their geometry object.

# %% [raw]
# set_default_geometry(geometry_1d)
# onedim_part = ParticleLayer(dataset=...)
#
# get_default_geometry(exp.atmosphere) # returns geometry_3d
# get_default_geometry(onedim_part)    # returns geometry_1d

# %% [markdown]
# It ensures the following setup still works:

# %% [raw]
# with geometry_3d:
#     exp = AtmosphereExperiment(
#         atmosphere=ParticleLayer(dataset=...),
#         measures=camera,
#         integrator={"type": integrator, "moment": True},
#     )
#
# ... # <- Any number of subsequent calls to set_default_geometry
#
# with geometry_3d:
#     eradiate.run(exp)
#
# ... # <- Any number of subsequent calls to set_default_geometry
#
# del geometry_3d # lost the geometry_3d ref in this framed scope
#
# eradiate.run(exp) # still works and uses geometry_3d, even if exp and its children do not keep any geometry as member

# %% [markdown]
# The other solution is to make the different scene elements functionaly pure wrt the geometry parameter. In other word, each scene elements would recalculate any property depending on the geometry everytime it is requested. This could lead to higher Eradiate setup times or spectral updates. To be investigated...

# %% [markdown]
# ### 2. Generating Mitsuba gridvolume objects
#
# This is already in use in 3D parts only
#
# An noticeable issue in the codebase is the repeated logic for gridvolume creation in the different phase functions and the atmospheric part. A dedicated function is proposed to factorize this logic, limiting the risks of coordinates mishandling by developers. Furthermore, this entrypoint is easier to extend, should we decide to use another storage and indexing backend for the different volume properties.
#
# TBD examples

# %% [markdown]
# ## VII. Thermoprops
#
#  - Thermophysical properties are not yet adapted to handle 3D coordinates. the current solution  for this is assembled with duck-tape

# %% [markdown]
# ## VIII. Nice to haves
#
# ### 1. Grid visualization

# %% [markdown]
# A simple voxel representation could be use to display the 3D grid features to users in Jupyter notebooks prior to launching computations. We should be careful however to ensure that this is performant enough by either disabling this feature if the grid is too large, or by applying a resampling operation in preprocessing. WIP

# %%
def plot_voxels(data, rgb=None, standardize=True, dim_agg=None, max_dim=15):

    rgb = rgb or [1.0, 0.0, 0.0]
    dim_agg = dim_agg or [np.mean] * 3
    data = chunk_aggregate(data, dim_agg, size=15)

    if standardize:
        data = (data - data.min()) / (data.max() - data.min())

    
    def explode(data):
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e
    
    facecolors = np.asarray(
        [mcolors.to_hex([*rgb, x], keep_alpha=True) for x in data.ravel()]
    )
    facecolors = facecolors.reshape(data.shape)
    
    filled = np.ones(data.shape)
    
    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    #
    ## Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.01
    y[:, 0::2, :] += 0.01
    z[:, :, 0::2] += 0.01
    x[1::2, :, :] += 0.99
    y[:, 1::2, :] += 0.99
    z[:, :, 1::2] += 0.99
    
    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2)
    # ax.set_aspect('equal') = np.loadtxt("./test_3d_clouds/iprt_case_c2_mystic.dat")

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def render_volume(vol, dist_scale=1.0, color=(1, 0, 0), img_size=256,
                  azimuth=45, elevation=30, alpha_threshold=1.0, filter=None):
    """
    azimuth:   rotation around Z axis, in degrees (0 = looking along X)
    elevation: tilt up/down, in degrees (0 = horizontal, 90 = top-down)
    """
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    filter = filter or (0.0, 1.0)
    vol = np.where(vol <= filter[0], 0, vol)
    vol = np.where(vol >= filter[1], 0, vol)

    vol = vol[:, :, ::-1]

    az = np.radians(azimuth)
    el = np.radians(elevation)

    depth = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])

    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(depth, right)
    up /= np.linalg.norm(up)

    px = np.linspace(-1, 1, img_size)
    py = np.linspace(-1, 1, img_size)
    gx, gy = np.meshgrid(px, py)
    origins = gx[..., None] * right + gy[..., None] * up
    origins =  origins * dist_scale

    D, H, W = vol.shape
    n_steps = D + H + W
    center = np.array([D/2, H/2, W/2])
    scale  = max(D, H, W) / 2

    accumulated_color = np.zeros((img_size, img_size, 3))
    alpha             = np.zeros((img_size, img_size))

    for i in range(n_steps):
        t = (i / n_steps - 0.5) * 2
        pos   = origins + t * depth
        world = pos * scale + center
        xi, yi, zi = (world[..., k].astype(int) for k in range(3))
        mask = (
            (xi >= 0) & (xi < D) &
            (yi >= 0) & (yi < H) &
            (zi >= 0) & (zi < W)
        )

        if not (alpha < alpha_threshold).any():
            break

        density = np.zeros((img_size, img_size))
        density[mask] = vol[xi[mask], yi[mask], zi[mask]]
        contrib = density * (1 - alpha)
        accumulated_color += contrib[..., None] * np.array(color)
        alpha             += contrib

    accumulated_color = np.clip(accumulated_color / (accumulated_color.max() + 1e-8), 0, 1)
    rgba = np.dstack([accumulated_color, np.clip(alpha, 0, 1)])

    def world_to_screen(world_pos):
        """Project a 3D world position to 2D screen pixel coordinates."""
        p = (world_pos - center) / scale  # back to normalized [-1,1] space
        sx =  np.dot(p, right)
        sy = -np.dot(p, up)
        return img_size/2 + sx * img_size/2 * dist_scale, \
               img_size/2 + sy * img_size/2 * dist_scale

    # Axes start at volume origin (0,0,0) and end at (D,0,0), (0,H,0), (0,0,W)
    origin = np.array([0.0, 0.0, 0.0])
    axes = [
        (np.array([D, 0, 0]), "red",   "X"),
        (np.array([0, H, 0]), "green", "Y"),
        (np.array([0, 0, W]), "blue",  "Z"),
    ]

    ox, oy = world_to_screen(origin)

    fig, ax = plt.subplots()
    ax.axis("off")

    for end_world, col, label in axes:
        ex, ey = world_to_screen(end_world)
        ox_c = float(np.clip(ox, 0, img_size))
        oy_c = float(np.clip(oy, 0, img_size))
        ex_c = float(np.clip(ex, 0, img_size))
        ey_c = float(np.clip(ey, 0, img_size))

        dot = np.dot((end_world - origin) / np.linalg.norm(end_world - origin), depth)
        a = 1.0 if dot >= 0 else 0.35
        ls = "solid" if dot >= 0 else "dashed"

        dx, dy = ex_c - ox_c, ey_c - oy_c
        length = np.sqrt(dx**2 + dy**2)
        if length < 1:
            continue  # axis points directly at camera, skip

        head = min(8, length * 0.2)
        ax.arrow(ox_c, oy_c, dx, dy,
                 head_width=head, head_length=head,
                 fc=col, ec=col, alpha=a,
                 linestyle=ls, length_includes_head=True)
        ax.text(ex_c + dx/length * 10, ey_c + dy/length * 10, label,
                color=col, fontsize=10, ha="center", va="center",
                fontweight="bold", alpha=a)
    
    ax.imshow(rgba, interpolation="nearest")
    plt.tight_layout(pad=0)
    plt.show()


# %%
from eradiate.contexts import KernelContext

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ctx =  KernelContext()
data = particles_tau2d.eval_sigma_t(ctx.si).m
plot_voxels(data)

plt.show()
plt.close()

# %%
plot_voxels(chunk_aggregate(grid.ext.values.reshape(100, 100, -1), [np.mean]* 3, 3))

# %%
render_volume(chunk_aggregate(grid.ext.values.reshape(100, 100, -1), [np.mean]* 3, 3), azimuth=30, elevation=15, dist_scale=1.0)

# %%
filled.shape, filled_2.shape

# %% [markdown]
# ## Scratch

# %%
from eradiate import traverse

tmpl, pmap = traverse(experiment.atmosphere)

# %%
ds

# %%
