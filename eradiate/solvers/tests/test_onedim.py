import numpy as np

from eradiate.solvers.onedim import OneDimSolver


def test_onedimsolver(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Construct
    solver = OneDimSolver()
    assert solver.dict_scene == OneDimSolver.DEFAULT_DICT_SCENE

    # Check if default scene is valid
    assert load_dict(solver.dict_scene) is not None

    # Run simulation with default parameters (and check if result array is cast to scalar)
    assert solver.run() == 0.1591796875

    # Run simulation with array of vzas (and check if result array is squeezed)
    result = solver.run(vza=np.linspace(0, 90, 91), spp=32)
    assert result.shape == (91,)
    assert np.all(result == 0.1591796875)

    # Run simulation with array of vzas and vaas
    result = solver.run(vza=np.linspace(0, 90, 11),
                        vaa=np.linspace(0, 180, 11),
                        spp=32)
    assert result.shape == (11, 11)
    assert np.all(result == 0.1591796875)


# def test_onedimsolverdict(variant_scalar_mono):
#     # Construct
#     solver = OneDimSolverDict()
#     assert solver.mode == DEFAULT_ONEDIMDICT_MODE
#     assert solver.illumination == DEFAULT_ONEDIMDICT_ILLUMINATION
#     assert solver.measure == DEFAULT_ONEDIMDICT_MEASURE
#     assert solver.atmosphere == None
#     assert solver.surface == DEFAULT_ONEDIMDICT_SURFACE
#
#     # Run simulation with default parameters
#     assert solver.run() == 0.1591796875
#
#     # Run simulation with custom emitter
#     solver = OneDimSolverDict(illumination={'type': 'constant'})
#     assert np.isclose(solver.run(), 0.5, rtol=1e-2)

