.. _sec-reference-scenes:

Scene generation [eradiate.scenes]
==================================

.. _sec-reference-scenes-core:

Core [eradiate.scenes.core]
---------------------------
.. currentmodule:: eradiate.scenes.core

.. autosummary::
   :toctree: generated/

   SceneElement
   KernelDict

.. _sec-reference-scenes-atmosphere:

Atmosphere [eradiate.scenes.atmosphere]
---------------------------------------
.. currentmodule:: eradiate.scenes.atmosphere

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   AtmosphereFactory
   Atmosphere

**Scene elements**

.. autosummary::
   :toctree: generated/

   HomogeneousAtmosphere
   HeterogeneousAtmosphere

.. _sec-reference-scenes-biosphere:

Biosphere [eradiate.scenes.biosphere]
-------------------------------------
.. currentmodule:: eradiate.scenes.biosphere

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   BiosphereFactory
   Canopy

**Scene elements**

.. autosummary::
   :toctree: generated/

   CanopyElement
   LeafCloud
   AbstractTree
   InstancedCanopyElement
   DiscreteCanopy

**Parameters for LeafCloud generators**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/

      _leaf_cloud.CuboidLeafCloudParams
      _leaf_cloud.SphereLeafCloudParams
      _leaf_cloud.EllipsoidLeafCloudParams
      _leaf_cloud.CylinderLeafCloudParams
      _leaf_cloud.ConeLeafCloudParams

.. _sec-reference-scenes-surface:

Surfaces [eradiate.scenes.surface]
----------------------------------
.. currentmodule:: eradiate.scenes.surface

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   SurfaceFactory
   Surface

**Scene elements**

.. autosummary::
   :toctree: generated/

   BlackSurface
   LambertianSurface
   RPVSurface

.. _sec-reference-scenes-illumination:

Illumination [eradiate.scenes.illumination]
-------------------------------------------
.. currentmodule:: eradiate.scenes.illumination

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   IlluminationFactory
   Illumination

**Scene elements**

.. autosummary::
   :toctree: generated/

   DirectionalIllumination
   ConstantIllumination

.. _sec-reference-scenes-measure:

Measures [eradiate.scenes.measure]
----------------------------------
.. currentmodule:: eradiate.scenes.measure

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   MeasureFactory
   Measure
   DistantMeasure

**Sensor information data structure**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/

      _core.SensorInfo

**Measure spectral configuration**

   .. autosummary::
      :toctree: generated/

      MeasureSpectralConfig

**Result storage and processing**

.. autosummary::
   :toctree: generated/

   MeasureResults

**Scene elements**

.. autosummary::
   :toctree: generated/

   DistantRadianceMeasure
   DistantReflectanceMeasure
   DistantFluxMeasure
   DistantAlbedoMeasure
   PerspectiveCameraMeasure
   RadiancemeterMeasure
   RadiancemeterArrayMeasure

**Target and origin specification for DistantMeasure**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/

      _distant.TargetOrigin
      _distant.TargetOriginPoint
      _distant.TargetOriginRectangle
      _distant.TargetOriginSphere

.. _sec-reference-scenes-phase_functions:

Phase functions [eradiate.scenes.phase]
---------------------------------------
.. currentmodule:: eradiate.scenes.phase

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   PhaseFunctionFactory
   PhaseFunction

**Scene elements**

.. autosummary::
   :toctree: generated/

   RayleighPhaseFunction
   HenyeyGreensteinPhaseFunction

.. _sec-reference-scenes-integrators:

Integrators [eradiate.scenes.integrators]
-----------------------------------------
.. currentmodule:: eradiate.scenes.integrators

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   IntegratorFactory
   Integrator

**Scene elements**

.. autosummary::
   :toctree: generated/

   PathIntegrator
   VolPathIntegrator
   VolPathMISIntegrator

.. _sec-reference-scenes-spectra:

Spectra [eradiate.scenes.spectra]
---------------------------------
.. currentmodule:: eradiate.scenes.spectra

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   SpectrumFactory
   Spectrum

**Scene elements**

.. autosummary::
   :toctree: generated/

   UniformSpectrum
   InterpolatedSpectrum
   SolarIrradianceSpectrum
   AirScatteringCoefficientSpectrum
