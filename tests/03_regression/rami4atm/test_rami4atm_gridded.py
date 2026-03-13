import functools

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from robot.api import logger

import eradiate
from eradiate.scenes.atmosphere import (
    GriddedHeterogeneousAtmosphere,
    GriddedMolecularAtmosphere,
)
from eradiate.scenes.geometry import GriddedParallelGeometry
from eradiate.test_tools.regression import SidakTTest, figure_to_html
from eradiate.test_tools.test_cases import rami4atm

cases = [
    "hom00_rpv_e00s_m04_z30a000_brfpp",
    "hom00_whi_s00s_m04_z30a000_brfpp",
    "hom00_rpv_0d6s_m04_z30a000_brfpp",
    "hom00_whi_a00s_m04_z30a000_brfpp",
    "hom00_rpv_sd2s_m04_z30a000_brfpp",
    "hom00_rpv_ac2s_m04_z30a000_brfpp",
    "hom00_rpv_ec6s_m04_z30a000_brfpp",
]


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", cases)
@pytest.mark.filterwarnings(
    "ignore:User-specified a background spectral grid is overridden by atmosphere spectral grid"
)
def test_rami4atm_gridded(mode_ckd_double, case, artefact_dir):
    ctor = functools.partial(rami4atm.create_rami4atm_toa, case=case)
    variables = ["radiance"]
    threshold = 0.005

    _, exps = ctor(spp=1000)
    exps_3d = []

    resolution = (2, 3)
    resx, resy = resolution

    for exp in exps:
        mol_atm_1d = exp.atmosphere.molecular_atmosphere

        geometry = GriddedParallelGeometry(
            zgrid=exp.geometry.zgrid,
            xy_resolution=resolution,
        )

        mol_atm_3d = None
        if mol_atm_1d:
            thermoprops_grid = [
                mol_atm_1d.thermoprops for _ in range(resx) for __ in range(resy)
            ]
            mol_atm_3d = GriddedMolecularAtmosphere(
                geometry=geometry,
                absorption_data=mol_atm_1d.absorption_data,
                thermoprops_grid=thermoprops_grid,
                grid_resolution=resolution,
                has_absorption=mol_atm_1d.has_absorption,
                has_scattering=mol_atm_1d.has_scattering,
            )

        if len(exp.atmosphere.particle_layers):
            tau_ref = exp.atmosphere.particle_layers[0].tau_ref
            tau_ref = tau_ref * np.ones(resolution)

        atm = GriddedHeterogeneousAtmosphere(
            geometry=geometry,
            molecular_atmosphere=mol_atm_3d,
            particle_layers=[
                attrs.evolve(layer, tau_ref=tau_ref)
                for layer in exp.atmosphere.particle_layers
            ],
        )

        exps_3d.append(
            attrs.evolve(
                exp,
                atmosphere=atm,
                geometry=geometry,
                integrator={"type": "volpath", "moment": True},
            )
        )

    raw_results_3d = [eradiate.run(exp) for exp in exps_3d]

    result = raw_results_3d[0]
    logger.info(result._repr_html_(), html=True)

    raw_results = [eradiate.run(exp) for exp in exps]
    reference = raw_results[0]

    logger.info(result._repr_html_(), html=True)

    logger.info(reference._repr_html_(), html=True)

    for variable in variables:
        figure, ax = plt.subplots(1)

        reference[variable].mean(dim="w").plot(ax=ax)
        result[variable].mean(dim="w").plot(ax=ax)

        svg_figure = figure_to_html(figure)
        logger.info(svg_figure, html=True)

        test = SidakTTest(
            name=case,
            value=result,
            reference=reference,
            threshold=threshold,
            archive_dir=artefact_dir,
            variable=variable,
            plot=False,
        )

        passed = test.run(diagnostic=True)
        assert passed
