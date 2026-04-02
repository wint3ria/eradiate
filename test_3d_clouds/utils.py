import matplotlib.pyplot as plt
import numpy as np
import pint
import xarray as xr


def load_aerosol_data(data_path: str, particle_shape: str, wavelength: pint.Quantity):
    # Load aerosol component dataset
    file = xr.open_dataset(data_path)
    data = file.copy()

    # phase raw shape is [wavelenth, phase, theta]
    # reshape to target [wavelenth, theta, phase]
    phase_da = data.phase.isel(nreff=0).dropna(dim="nthetamax")
    phase_raw = phase_da.transpose("nlam", "nthetamax", "nphamat").values

    n_wavelength = phase_da.nlam.shape[0]
    n_theta = phase_da.nthetamax.shape[0]

    # target shape is [wavelength, theta, i, j]
    phase_np = np.zeros((n_wavelength, n_theta, 4, 4))

    if particle_shape == "spherical":
        phase_np[:, :, 0, 0] = phase_raw[:, :, 0]
        phase_np[:, :, 1, 1] = phase_raw[:, :, 0]
        phase_np[:, :, 0, 1] = phase_raw[:, :, 1]
        phase_np[:, :, 1, 0] = phase_raw[:, :, 1]
        phase_np[:, :, 2, 2] = phase_raw[:, :, 2]
        phase_np[:, :, 3, 3] = phase_raw[:, :, 2]
        phase_np[:, :, 2, 3] = phase_raw[:, :, 3]
        phase_np[:, :, 3, 2] = phase_raw[:, :, 3]
    elif particle_shape == "spheroidal":
        phase_np[:, :, 0, 0] = phase_raw[:, :, 0]
        phase_np[:, :, 0, 1] = phase_raw[:, :, 1]
        phase_np[:, :, 1, 0] = phase_raw[:, :, 1]
        phase_np[:, :, 1, 1] = phase_raw[:, :, 4]
        phase_np[:, :, 2, 2] = phase_raw[:, :, 2]
        phase_np[:, :, 2, 3] = phase_raw[:, :, 3]
        phase_np[:, :, 3, 2] = phase_raw[:, :, 3]
        phase_np[:, :, 3, 3] = phase_raw[:, :, 5]
    else:
        NotImplementedError

    # populate the eradiate dataset that has the correct format

    def make_phase_eradiate(lbda):
        return xr.Dataset(
            data_vars={
                "sigma_t": (["w"], data.ext.isel(nreff=0).values, {"units": "1/km"}),
                "albedo": (["w"], data.ssa.isel(nreff=0).values, {"units": ""}),
                "phase": (["w", "mu", "i", "j"], phase_np),
            },
            coords={
                "w": ("w", [lbda], {"units": "um"}),
                "mu": (
                    "mu",
                    np.cos(
                        np.deg2rad(
                            data.theta.isel(
                                nlam=0, nreff=0, nphamat=0, nthetamax=phase_da.nthetamax
                            ).values
                        )
                    ),
                ),
                "i": ("i", range(4)),
                "j": ("j", range(4)),
            },
        )

    # This is a hack, do it a second time with a different wavelength
    # and concatenate so that eradiate can still do some interpolation
    phase_eradiate = xr.concat(
        [
            make_phase_eradiate(wavelength.m_as("micron") - 0.1),
            make_phase_eradiate(wavelength.m_as("micron")),
            make_phase_eradiate(wavelength.m_as("micron") + 0.1),
        ],
        dim="w",
    )
    return phase_eradiate


def plot_stokes_dolp_polar(axes, Theta, Phi, isza, rad_stokes, cbar_location):
    stokes_labels = ["I", "Q", "U", "V"]

    lp = 0

    for i in range(1, 3):
        lp += rad_stokes[:, :, i] ** 2
    dolp = (np.sqrt(lp) / rad_stokes[:, :, 0]).T

    for i in range(4):
        rad = rad_stokes[:, :, i].T
        if i == 0:
            plot_polar(
                axes[isza, i],
                Theta,
                Phi,
                rad,
                stokes_labels[i],
                cmap="viridis",
                cbar_location=cbar_location,
                vmin=0,
                vmax=np.max(rad),
                log_scale=True,
            )
        else:
            vmax_QU = np.max(np.abs(rad))
            plot_polar(
                axes[isza, i],
                Theta,
                Phi,
                rad,
                stokes_labels[i],
                cmap="coolwarm",
                cbar_location=cbar_location,
                vmin=-vmax_QU,
                vmax=vmax_QU,
                log_scale=False,
            )

    plot_polar(
        axes[isza, 4],
        Theta,
        Phi,
        dolp,
        "DoLP",
        cmap="viridis",
        cbar_location=cbar_location,
        vmin=np.min(dolp),
        vmax=np.max(dolp),
        log_scale=False,
    )


def plot_stokes_and_dop(result_eradiate):
    """
    Plot Stokes parameters (I, Q, U, V) and Degree of Polarization (DoP)
    from an Eradiate simulation result.

    Parameters
    ----------
    result_eradiate : dict or xarray-like
        Must contain `result_eradiate['radiance'].data` with shape
        [stokes, ...], e.g. [1, 4, Ny, Nx, 1, 1].
    """

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    # --- Stokes I ---
    img_I = result_eradiate["radiance"].data[0, 0, :, :, 0, 0]
    im_I = axs[0].imshow(img_I, origin="lower", aspect="equal", cmap="Blues_r")
    axs[0].set_title("I")
    fig.colorbar(im_I, ax=axs[0], orientation="vertical", shrink=0.7)

    # --- Stokes Q, U, V ---
    for idx, label in enumerate(["Q", "U", "V"], start=1):
        img = result_eradiate["radiance"].data[0, idx, :, :, 0, 0]
        vmax = np.max(np.abs(img))
        im = axs[idx].imshow(
            img, origin="lower", aspect="equal", cmap="seismic", vmin=-vmax, vmax=vmax
        )
        axs[idx].set_title(label)
        fig.colorbar(im, ax=axs[idx], orientation="vertical", shrink=0.7)

    # --- Degree of Polarization ---
    I = result_eradiate["radiance"].data[0, 0, :, :, 0, 0]
    Q = result_eradiate["radiance"].data[0, 1, :, :, 0, 0]
    U = result_eradiate["radiance"].data[0, 2, :, :, 0, 0]
    V = result_eradiate["radiance"].data[0, 3, :, :, 0, 0]

    dop = np.sqrt(Q**2 + U**2 + V**2) / np.abs(I)
    im_dop = axs[4].imshow(dop, origin="lower", aspect="equal", cmap="viridis")
    axs[4].set_title("DoP")
    fig.colorbar(im_dop, ax=axs[4], orientation="vertical", shrink=0.7)

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_stokes_from_ascii(data):
    """
    Plot Stokes I, Q, U, V and Degree of Polarization (DoP).

    Assumes:
    - 7th, 8th, 9th, and 10th columns = I, Q, U, V
    - Data reshapes to 70 x 70 arrays.

    Parameters
    ----------
    data : data array from IPRT
    """

    # --- Extract Stokes components ---
    I = data[:, 7].reshape(70, 70).T
    Q = data[:, 8].reshape(70, 70).T
    U = data[:, 9].reshape(70, 70).T
    V = data[:, 10].reshape(70, 70).T

    # --- Compute Degree of Polarization ---
    dop = np.sqrt(Q**2 + U**2 + V**2) / np.abs(I)
    dop = np.clip(dop, 0, 1)  # clamp DoP between 0–1

    # --- Prepare figure ---
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    # Stokes I
    im_I = axs[0].imshow(I, origin="lower", aspect="equal", cmap="Blues_r")
    axs[0].set_title("I")
    fig.colorbar(im_I, ax=axs[0], shrink=0.7)

    # Stokes Q
    vmax_Q = np.max(np.abs(Q))
    im_Q = axs[1].imshow(
        Q, origin="lower", aspect="equal", cmap="RdBu_r", vmin=-vmax_Q, vmax=vmax_Q
    )
    axs[1].set_title("Q")
    fig.colorbar(im_Q, ax=axs[1], shrink=0.7)

    # Stokes U
    vmax_U = np.max(np.abs(U))
    im_U = axs[2].imshow(
        U, origin="lower", aspect="equal", cmap="RdBu_r", vmin=-vmax_U, vmax=vmax_U
    )
    axs[2].set_title("U")
    fig.colorbar(im_U, ax=axs[2], shrink=0.7)

    # Stokes V
    vmax_V = np.max(np.abs(V))
    im_V = axs[3].imshow(
        V, origin="lower", aspect="equal", cmap="RdBu_r", vmin=-vmax_V, vmax=vmax_V
    )
    axs[3].set_title("V")
    fig.colorbar(im_V, ax=axs[3], shrink=0.7)

    # Degree of Polarization
    im_dop = axs[4].imshow(
        dop, origin="lower", aspect="equal", cmap="viridis"
    )  # , vmin=0, vmax=1)
    axs[4].set_title("DoP")
    fig.colorbar(im_dop, ax=axs[4], shrink=0.7)

    plt.tight_layout()
    plt.show()

    return fig, axs
