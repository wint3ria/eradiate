import os

import matplotlib.pyplot as plt
import numpy as np

import eradiate

eradiate_dir = eradiate.config.dir
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def test_albedo(mode_mono):
    """
    Albedo
    ======

    This system test verifies the behaviour of the apps capable of albedo
    computation.

    Rationale
    ---------

    We use a scene consisting of a single surface with a diffuse, spectrally
    non-uniform BRDF.

    * Geometry: A Lambertian surface with linearly varying reflectance for
      0 to 1 between 500 and 700 nm.
    * Illumination: Directional illumination from the zenith (default irradiance)
      or constant illumination (default radiance).
    * Atmosphere/canopy: No atmosphere nor canopy.
    * Measure: Distant albedo measure with a film of size 64 x 64. This
      guarantees reasonable stratification of the film sampling and ensures
      quick converge to the expected value, thus allowing for a low sample
      count.

    The test is run for the ``OneDimSolverApp`` and ``RamiSolverApp`` classes.

    Expected behaviour
    ------------------

    We expect the albedo to be equal to the reflectance of the surface.

    Results
    -------

    .. image:: generated/plots/albedo_onedim_directional.png
       :width: 100%

    .. image:: generated/plots/albedo_rami_directional.png
       :width: 100%

    .. image:: generated/plots/albedo_onedim_constant.png
       :width: 100%

    .. image:: generated/plots/albedo_rami_constant.png
       :width: 100%

    """
    apps = {
        "onedim_directional": eradiate.solvers.onedim.OneDimSolverApp(
            scene={
                "measures": [
                    {
                        "type": "distant_albedo",
                        "spectral_cfg": {
                            "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0]
                        },
                        "film_resolution": (64, 64),
                        "spp": 100,
                    }
                ],
                "atmosphere": None,
                "surface": {
                    "type": "lambertian",
                    "reflectance": {
                        "type": "interpolated",
                        "wavelengths": [500.0, 700.0],
                        "values": [0.0, 1.0],
                    },
                },
                "illumination": {"type": "directional", "zenith": 0.0},
            }
        ),
        "onedim_constant": eradiate.solvers.onedim.OneDimSolverApp(
            scene={
                "measures": [
                    {
                        "type": "distant_albedo",
                        "spectral_cfg": {
                            "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0]
                        },
                        "film_resolution": (64, 64),
                        "spp": 100,
                    }
                ],
                "atmosphere": None,
                "surface": {
                    "type": "lambertian",
                    "reflectance": {
                        "type": "interpolated",
                        "wavelengths": [500.0, 700.0],
                        "values": [0.0, 1.0],
                    },
                },
                "illumination": {"type": "constant"},
            }
        ),
        "rami_directional": eradiate.solvers.rami.RamiSolverApp(
            scene={
                "measures": [
                    {
                        "type": "distant_albedo",
                        "spectral_cfg": {
                            "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0]
                        },
                        "film_resolution": (64, 64),
                        "spp": 100,
                    }
                ],
                "canopy": None,
                "surface": {
                    "type": "lambertian",
                    "reflectance": {
                        "type": "interpolated",
                        "wavelengths": [500.0, 700.0],
                        "values": [0.0, 1.0],
                    },
                },
                "illumination": {"type": "directional", "zenith": 0.0},
            }
        ),
        "rami_constant": eradiate.solvers.rami.RamiSolverApp(
            scene={
                "measures": [
                    {
                        "type": "distant_albedo",
                        "spectral_cfg": {
                            "wavelengths": [500.0, 550.0, 600.0, 650.0, 700.0]
                        },
                        "film_resolution": (64, 64),
                        "spp": 100,
                    }
                ],
                "canopy": None,
                "surface": {
                    "type": "lambertian",
                    "reflectance": {
                        "type": "interpolated",
                        "wavelengths": [500.0, 700.0],
                        "values": [0.0, 1.0],
                    },
                },
                "illumination": {"type": "directional", "zenith": 0.0},
            }
        ),
    }

    for app_name, app in apps.items():
        # Run simulation
        app.run()
        results = app.results["measure"]

        # Plot results
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
        wavelengths = results["albedo"].w.values
        albedos = results["albedo"].values.squeeze()
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        ax1.plot(wavelengths, albedos, linestyle="--", marker="o")
        ax1.set_title("Albedo")
        ax1.set_xlabel("Wavelength [nm]")

        rdiffs = (albedos - expected) / expected
        ax2.plot(wavelengths, rdiffs, linestyle="--", marker="o")
        ax2.set_title("Relative difference")
        ax2.set_xlabel("Wavelength [nm]")
        rdiffs_max = np.max(np.abs(rdiffs[~np.isnan(rdiffs)]))
        exp = np.ceil(np.log10(rdiffs_max))
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim([-(10 ** exp), 10 ** exp])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.ticklabel_format(axis="y", style="sci", scilimits=[-3, 3])
        # Hide offset label and add it as axis label
        if abs(exp) >= 3:
            ax2.yaxis.offsetText.set_visible(False)
            ax2.yaxis.set_label_text(f"×$10^{{{int(exp)}}}$")

        plt.suptitle(f"Case: {app_name}")
        plt.tight_layout()

        filename = f"albedo_{app_name}.png"
        ensure_output_dir(os.path.join(output_dir, "plots"))
        fname_plot = os.path.join(output_dir, "plots", filename)

        fig.savefig(fname_plot, dpi=200)
        plt.close()

        # Check results
        assert np.allclose(results["albedo"].values, expected, atol=1e-3)
