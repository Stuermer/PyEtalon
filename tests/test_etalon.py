import pytest

from PyEtalon.etalon import Etalon
import numpy as np
from PyEtalon.custom_material_data import Vacuum, Suprasil, Silver
from PyEtalon import plotting

@pytest.fixture(scope="module")
def silver_etalon():
    return Etalon(
        [Vacuum, Silver, Suprasil],  # specify the layer materials
        ["Vacuum", "Silver", "Substrate"],  # specify the layer names
        [-np.inf, 30.0, np.inf],  # specify the layer thicknesses
        [0, 1, 2],  # specify the layer order/indices
        wavelength=np.linspace(500, 1100, 5000),  # specify the wavelength range
        d_spacer=5e-3,  # specify the spacer thickness in meters
        aoi=0.0,  # specify the angle of incidence in degrees
        identifier="Silver 30nm",  # specify the identifier of the etalon
    )

def test_etalon():
    pass


def test_reflectivity_transmissivity():
    e = Etalon(
        [Vacuum, Suprasil],
        ["vacuum", "Suprasil"],
        [-np.inf, np.inf],
        [0, 1],
        wavelength=np.array([587.56]),
        d_spacer=0.01,
        aoi=20.0,
        identifier="test",
    )
    ref = e.mirror_reflectivity
    trans = e.mirror_transmission
    # data checked on https://www.rp-photonics.com/fresnel_equations.html
    assert np.isclose(ref, 0.0411, atol=1e-4)
    assert np.isclose(trans, 0.959, atol=1e-3)


def test_etalon_fsr(silver_etalon):
    assert silver_etalon.ideal_fsr == 29.9792458


def test_plotting(silver_etalon):
    plotting.plot_gd(silver_etalon)
    plotting.plot_fsr(silver_etalon)
    plotting.plot_gdd(silver_etalon)
    plotting.plot_phase(silver_etalon)
    plotting.plot_absorption(silver_etalon)
    plotting.plot_layer_thickness_influence(silver_etalon)
    plotting.plot_layer_thickness_influence_gd(silver_etalon)
    plotting.plot_layer_thickness_influence_reflectivity(silver_etalon)
    plotting.plot_normalized_fsr(silver_etalon)
    plotting.plot_stack([silver_etalon])
    plotting.plot_absorption(silver_etalon)
    plotting.plot_normalized_fsr(silver_etalon)
    plotting.plot_mirror_transmission(silver_etalon)
    plotting.plot_reflectivity(silver_etalon)
    plotting.plot_reflectivity_finesse(silver_etalon)
    plotting.plot_refractive_index_materials(silver_etalon, silver_etalon.wavelength / 1000.0)
    plotting.plot_refractive_index_materials(
        silver_etalon, silver_etalon.wavelength / 1000.0, False
    )
