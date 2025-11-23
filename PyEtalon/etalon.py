from __future__ import annotations

import json
import logging
import pathlib
from functools import cached_property
from typing import Callable

import numpy as np
from pyindexrepo import Material
from scipy.interpolate import UnivariateSpline

from PyEtalon.tmm import rt

# from tmmnlay import calculate_rt as calculate_rt_gpu
logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
# logger.addHandler(c_handler)
logger.setLevel("DEBUG")

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

# constants
c = 299792458  # speed of light [m/s]


def guess_m(
    wavelength: float | np.ndarray,
    d: float = 9.99e-3,
    aoi: float = 0.0,
    n: float | Callable = 1.0,
    phase: Callable | None = None,
) -> tuple[int | np.ndarray, np.ndarray]:
    """
    Guesses the peak number m for a given peak wavelength.

    Args:
        wavelength: peak wavelength in [nm]
        d: spacer thickness in [m]
        aoi: angle of incidence in [rad]
        n: refractive index of cavity material
        phase: phase shift upon reflection in [rad]

    Returns:
        guessed peak numbers, difference between integer and float peak numbers
    """
    if phase is None:
        if isinstance(n, float):
            float_m = 2.0 * n * d * np.cos(aoi) / (wavelength * 1e-9)
        elif callable(n):
            float_m = 2.0 * n(wavelength) * d * np.cos(aoi) / (wavelength * 1e-9)
        else:
            raise ValueError("n needs to be a float or a callable function")
        int_m = np.array(np.rint(float_m), dtype=int)
    else:
        if isinstance(n, float):
            float_m = (
                2.0 * d * n * np.cos(aoi)
                + phase(wavelength) * (wavelength * 1e-9) / np.pi
            ) / (wavelength * 1e-9)
        elif callable(n):
            float_m = (
                2.0 * d * n(wavelength.copy()) * np.cos(aoi)
                + phase(wavelength.copy()) * (wavelength * 1e-9) / np.pi
            ) / (wavelength * 1e-9)
        elif isinstance(n, np.ndarray):
            float_m = (
                2.0 * d * n * np.cos(aoi)
                + phase(wavelength.copy()) * (wavelength * 1e-9) / np.pi
            ) / (wavelength * 1e-9)
        else:
            raise ValueError(
                "n needs to be a float, an array of floats or a callable function."
            )
        int_m = np.array(np.rint(float_m), dtype=int)
    return int_m, float_m - int_m


def lambda_peaks(
    m: int | np.ndarray,
    d: float = 9.99e-3,
    n: float | Callable = 1.0,
    aoi: float = 0.0,
    phase: None | Callable = None,
) -> float | np.ndarray:
    """
    Returns peak wavelength for given etalon parameters

    It uses the dispersion formula as given in https://doi.org/10.1364/AO.30.004126
    Args:
        m: Diffraction order (int) or 1D array with 'int' type.
        d: physical mirror distance [m]
        n: Refractive index of medium in between mirrors
        aoi: Angle of incidence [rad]
        phase: Interpolation function f = phase_shift(wavelength[nm])

    Returns:
        peak wavelength in [nm]

    """
    repeat = 3
    # approximate wavelength without phase shift:
    wl = 2.0 * d * 1.0 * np.cos(aoi) / m * 1e9
    if isinstance(n, float):
        wl = 2.0 * d * n * np.cos(aoi) / m * 1e9
    if callable(n):
        for _ in range(repeat):
            wl = 2.0 * d * n(wl) * np.cos(aoi) / m * 1e9
    if phase is not None:
        if isinstance(n, float):
            for _ in range(repeat):
                wl = (2.0 * d * 1e9 * np.cos(aoi) * n + phase(wl) / np.pi * wl) / m
        if callable(n):
            for _ in range(repeat):
                wl = (2.0 * d * 1e9 * np.cos(aoi) * n(wl) + phase(wl) / np.pi * wl) / m
    return wl


class Etalon:
    """
    Class for modeling an etalon with a multilayer thin film stack.

    Attributes:
        materials: List of materials used in the stack
        names: List of names of the materials
        d_stack_design: List of layer thicknesses in [nm]
        d_corrections: List of layer thickness corrections in [nm] - default is an array of zeros (same length as d_stack)
        num_layers: Number of layers in the stack
        identifier: Identifier for the etalon
        wavelength_min: Minimum wavelength in [nm] (if not given, it is calculated from the wavelength vector)
        wavelength_max: Maximum wavelength in [nm] (if not given, it is calculated from the wavelength vector)
        wavelength: Wavelength vector in [nm]
        normalized_wavelength: Normalized wavelength vector
        aoi: Angle of incidence in [rad]
        _d_spacer: Spacer thickness in [m] (incl. corrections if given)
        d_spacer_correction: Spacer thickness correction in [nm] - default is 0.0
        idx_stack: List of indices of the materials in the stack
        _m: Peak order number
    """

    def __init__(
        self,
        materials: list[Material],
        material_names: list[str] | None,
        d_stack: list[float] | np.ndarray,
        idx_stack: list[int],
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
        wavelength: list[float] | np.ndarray = None,
        d_spacer: float = 10e-3,
        aoi: float = 0.0,
        identifier: str = "Etalon",
    ):
        """Constructor for the Etalon class

        Args:
            materials: List of materials used in the stack
            material_names: List of names of the materials
            d_stack: List of layer thicknesses in [nm]
            idx_stack: List of indices of the materials in the stack
            wavelength_min: Minimum wavelength in [nm] (if not given, it is calculated from the wavelength vector)
            wavelength_max: Maximum wavelength in [nm] (if not given, it is calculated from the wavelength vector)
            wavelength: Wavelength vector in [nm]
            d_spacer: Spacer thickness in [m] (incl. corrections if given)
            aoi: Angle of incidence in [deg]
            identifier: Identifier for the etalon
        """
        if material_names is None:
            material_names = [m.name for m in materials]
        assert len(materials) == len(material_names), (
            "Materials and Material_names need to be the same length."
        )
        assert len(d_stack) == len(idx_stack), (
            f"Length of d_stack ({len(d_stack)}) and idx_stack ({len(idx_stack)}) is not equal."
        )

        assert wavelength is not None or (
            wavelength_min is not None and wavelength_max is not None
        ), (
            "Either specify the wavelength vector for calculations explicitly or specify minimum and maximum wavelength"
        )

        self.materials = materials
        self.names = material_names
        self.d_stack_design = np.array(d_stack)
        self.d_corrections = np.zeros_like(
            d_stack[1:-1]
        )  # exclude the spacer and ambient material
        self.num_layers = len(d_stack) - 2  # exclude the spacer and ambient material
        self.identifier = identifier
        if wavelength is not None:
            self.wavelength = wavelength
            self.wavelength_min = np.min(wavelength)
            self.wavelength_max = np.max(wavelength)
            self.normalized_wavelength = np.linspace(0, 1, len(wavelength))

        else:
            self.wavelength_min = wavelength_min
            self.wavelength_max = wavelength_max
            self.wavelength = np.linspace(wavelength_min, wavelength_max, 5000)
            self.normalized_wavelength = np.linspace(0, 1, len(self.wavelength))

        self.aoi = aoi
        self._d_spacer = d_spacer
        self.d_spacer_correction = 0.0

        self.idx_stack = np.array(idx_stack, dtype=int)

        self.material_coefficients = []
        for i, m in enumerate(materials[1:-1]):
            # check if material has attribute 'coefficients' - could be TabulatedIndexData
            if hasattr(m.n, "coefficients"):
                self.material_coefficients.append(m.n.coefficients)
            if m.k is not None:
                if hasattr(m.k, "coefficients"):
                    self.material_coefficients.append(m.k.coefficients)

        # Convert given wavelength to peak order number
        # self._m = guess_m(self.wavelength, self.d_spacer, self.aoi, self.n(0), self.phase_spline)
        self._m = np.arange(
            int(
                (
                    2.0
                    * self.d_spacer
                    * self.get_refractive_index(0, self.wavelength_max)
                )
                / (self.wavelength_max * 1e-9)
            ),
            int(
                (
                    2.0
                    * self.d_spacer
                    * self.get_refractive_index(0, self.wavelength_min)
                )
                / (self.wavelength_min * 1e-9)
            ),
        )[::-1]

    def __str__(self):
        return f"{self.identifier} Etalon: \n Number of layers: {self.num_layers - 2}. \n Used materials are: {self.materials}"

    def json_dict(self):
        json_dict = {}
        for k in [
            # "materials",
            # "names",
            # "d_stack",
            # "idx_stack",
            # "wavelength_min",
            # "wavelength_max",
            # "d_spacer",
            # "aoi",
            # "identifier",
            "all_parameters",
        ]:
            val = getattr(self, k)
            # TODO: make it work for materials
            if isinstance(val, np.ndarray):
                json_dict[k] = val.tolist()
            else:
                json_dict[k] = getattr(self, k)
        return json_dict

    def save_to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.json_dict(), f, indent=4)

    def get_refractive_index(self, idx_mat, wl=None):
        if wl is None:
            return self.materials[idx_mat].get_n(
                np.atleast_1d(self.wavelength / 1000.0)
            )
        else:
            return self.materials[idx_mat].get_n(np.atleast_1d(wl / 1000.0))

    def get_refractive_index_cavity(self, wl):
        """Returns the refractive index of the cavity material"""
        return self.get_refractive_index(0, wl)

    @property
    def rt(self):
        """Returns the reflection and transmission coefficients of the mirror coating stack"""
        return rt(self.nk_stack, self.d_stack, self.wavelength, self.aoi)

    @property
    def d_spacer(self) -> float:
        """Distance of spacer in [m] (incl. corrections if given)"""
        return self._d_spacer + self.d_spacer_correction / 1e9

    @property
    def nk_stack(self):
        # Calculate refractive indices
        res = np.empty(
            (len(self.wavelength), len(self.d_stack_design)), dtype=np.complex128
        )

        for i in np.sort(np.unique(self.idx_stack)):
            nn = self.get_refractive_index(i, self.wavelength)
            kk = self.materials[i].get_k(self.wavelength / 1000.0)
            nk_data = np.vectorize(complex)(nn, kk)
            for j, idx_mat in enumerate(self.idx_stack):
                if idx_mat == i:
                    res[:, j] = nk_data

        return res

    @cached_property
    def num_material_coefficients(self):
        """Returns the total number of coefficients for all coating materials

        The substrate and ambient material are not included.
        """
        n = 0
        for m in self.materials[1:-1]:
            if hasattr(m.n, "coefficients"):
                n += m.n.coefficients.size
            if m.k is not None:
                if hasattr(m.k, "coefficients"):
                    n += m.k.coefficients.size
        return n

    @cached_property
    def n_parameter(self):
        """Total number of parameters

        Number of layers + 1 for the spacer + number of coefficients for each coating material

        """
        return self.num_layers + 1 + self.num_material_coefficients

    @cached_property
    def all_parameter_names(self):
        names = ["d_spacer"]
        names.extend([f"d_{i}" for i in range(self.d_stack.size - 2)])
        for i, m in enumerate(self.materials[1:-1]):
            for j in range(m.n.coefficients.size):
                names.append(f"{self.materials[i + 1].name}_n_{j}")
            if m.k is not None:
                for j in range(m.k.coefficients.size):
                    names.append(f"{self.materials[i + 1].name}_k_{j}")
        return names

    @property
    def all_parameters(self):
        pars = np.zeros(self.n_parameter)
        pars[0] = self.d_spacer_correction
        pars[1 : self.num_layers + 1] = self.d_corrections
        j = 0
        for i, m in enumerate(self.materials[1:-1]):
            if hasattr(m.n, "coefficients"):
                n = m.n.coefficients.size
                pars[self.num_layers + 1 + j : self.num_layers + 1 + j + n] = (
                    m.n.coefficients
                )
                j += n
            if m.k is not None:
                if hasattr(m.k, "coefficients"):
                    n = m.k.coefficients.size
                    pars[self.num_layers + 1 + j : self.num_layers + 1 + j + n] = (
                        m.k.coefficients
                    )
                    j += n
        return pars

    def print_parameters(self):
        """print parameters names and parameters and index in a table"""
        print(f"{'Index':<5} {'Parameter':<40} {'Value':<20}")
        print("-" * 65)
        for i, (name, par) in enumerate(
            zip(self.all_parameter_names, self.all_parameters)
        ):
            print(f"{i:<5} {name:<40} {par:<20}")

    def print_parameters_horizontally(self, values_only=False):
        """print index, parameter names and values in a table.
        First row contains index, second row contains name, third row contains all values.
        Fix the column width to 20 characters"""
        if not values_only:
            print(f"{'Index':<10}", end="")
            for i in range(self.n_parameter):
                print(f"{i:<6}", end="")
            print()
            print("-" * 65)
            print(f"{'Parameter':<10}", end="")
            for name in self.all_parameter_names:
                print(f"{name:<7}", end="")
            print()
            print("-" * 65)
        print(f"{'Value':<10}", end="")
        for par in self.all_parameters:
            print(f"{par:7.3f}", end="")
        print()

    @all_parameters.setter
    def all_parameters(self, pars):
        assert pars.size == self.n_parameter, (
            f"Number of parameters does not match. Expected {self.n_parameter}, got {pars.size}"
        )

        self.d_spacer_correction = pars[0]
        self.d_corrections = pars[1 : self.num_layers + 1]

        j = 0
        for i, m in enumerate(self.materials[1:-1]):
            if hasattr(m.n, "coefficients"):
                n = m.n.coefficients.size

                m.n.coefficients = pars[
                    self.num_layers + 1 + j : self.num_layers + 1 + j + n
                ]
                j += n
            if m.k is not None:
                if hasattr(m.k, "coefficients"):
                    n = m.k.coefficients.size

                    m.k.coefficients = pars[
                        self.num_layers + 1 + j : self.num_layers + 1 + j + n
                    ]
                    j += n
        # self.update()

    @property
    def d_stack(self):
        """Returns the stack thickness including any corrections given (if any)"""
        dstack = self.d_stack_design.copy()
        dstack[1:-1] += self.d_corrections
        return dstack

    @property
    def d_stack_normalized(self):
        """Returns the stack thickness normalized to the maximum thickness"""
        stack_max_thickness = np.max(self.d_stack[1:-1])
        return self.d_stack / stack_max_thickness

    @property
    def phase(self):
        """Returns the unwrapped phase shift upon reflection in [rad] as a function of wavelength"""
        return np.unwrap(np.angle(self.rt[0]))

    @property
    def phase_spline(self):
        """Returns the phase as a spline"""
        return UnivariateSpline(self.wavelength, self.phase, k=3, s=0)

    @property
    def gd(self):
        """Returns the group delay upon reflection in [fs]"""
        return (
            self.phase_spline.derivative(1)(self.wavelength)
            * 1e6
            * self.wavelength**2
            / (2.0 * np.pi * c)
        )

    @property
    def gd_spline(self):
        """Returns the group delay upon reflection in [fs] as a spline"""
        return UnivariateSpline(self.wavelength, self.gd, k=5, s=0.0)

    @property
    def gdd(self):
        """Returns the group delay dispersion upon reflection in [fs^2]"""
        return (
            self.phase_spline.derivative(2)(self.wavelength)
            * self.wavelength**3
            / (2.0 * np.pi)
            / 1000.0
        )

    @property
    def gdd_spline(self):
        """Returns the group delay dispersion upon reflection in [fs^2] as a spline"""
        return UnivariateSpline(self.wavelength, self.gdd, k=5, s=0)

    def calculate_reflectivity_transmissivity(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate reflectivity and transmissivity from reflection and transmission coefficients.

        Returns:
            tuple[np.ndarray, np.ndarray]:ay of float64 values.
                - Transmissivity (T): 1D array of float64 values.
        """
        # Convert angle of incidence to radians
        aoi_rad = np.radians(self.aoi)

        # Extract refractive indices at the boundaries
        n_i = self.nk_stack[:, 0]  # Incident medium
        n_t = self.nk_stack[:, -1]  # Transmitted medium

        # Calculate transmitted angle using Snell's law: n_i * sin(theta_i) = n_t * sin(theta_t)
        sin_theta_t = np.real(n_i) * np.sin(aoi_rad) / np.real(n_t)
        cos_theta_t = np.sqrt(
            1 - sin_theta_t**2
        )  # cos(theta_t), assuming real refractive indices
        cos_theta_i = np.cos(aoi_rad)  # cos(theta_i)
        r, t = self.rt
        # Reflectivity: R = |r|^2
        R = np.abs(r) ** 2

        # Transmissivity: T = |t|^2 * Re(n_t cos(theta_t)) / Re(n_i cos(theta_i))
        T = (
            (np.abs(t) ** 2)
            * (np.real(n_t) * cos_theta_t)
            / (np.real(n_i) * cos_theta_i)
        )

        return R, T

    @property
    def mirror_reflectivity(self):
        """Returns the reflectivity of the mirror coating stack"""
        return self.calculate_reflectivity_transmissivity()[0]

    @property
    def mirror_transmission(self):
        """Returns the transmission of the mirror coating stack"""
        return self.calculate_reflectivity_transmissivity()[1]

    @property
    def fsr(self):
        """Return the real free spectral range in [GHz]

        The free spectral range is given by the formula:

        FSR [GHz] = c / (2 * n * d * cos(theta) - lambda^2 * 10^-9 / pi * dphi/dlambda)

        see https://doi.org/10.1103/PhysRevA.37.1802

        where:
        c: speed of light [m/s]
        n: refractive index of cavity material
        d: spacer thickness [m]
        theta: angle of incidence [rad]
        lambda: peak wavelength [nm]
        dphi/dlambda: derivative of phase with respect to wavelength [rad/nm]

        The factor 10^-9 is used to convert the wavelength from nm to m. There is a factor of 10^-18 from lambda^2, but
        also a factor of 10^9 from the derivative of the phase. The factor of 10^-9 cancels out.

        """
        n = self.get_refractive_index_cavity(self.wavelength)
        return (
            c
            / (
                2.0 * n * self.d_spacer * np.cos(np.deg2rad(self.aoi))
                - (self.wavelength**2 / np.pi)
                * 1e-9
                * self.phase_spline.derivative(1)(self.wavelength)
            )
            / 1e9
        )

    @property
    def ideal_fsr(self) -> float:
        """Return the ideal free spectral range in [GHz]

        The ideal free spectral range is given by the formula:

        FSR [GHz] = c / (2 * n * d * cos(theta))

            where:
            c: speed of light [m/s]
            n: refractive index of cavity material
            d: spacer thickness [m]
            theta: angle of incidence [rad]

        It neglects the phase shift upon reflection.
        """
        return (
            c
            / (
                2.0
                * self.get_refractive_index_cavity(self.wavelength).mean()
                * self.d_spacer
                * np.cos(np.deg2rad(self.aoi))
            )
            / 1e9
        )

    @property
    def fsr_wavelength(self):
        """Return the free spectral range in [nm]

        ChatGPT gives me this as approximation (phase shift is a small correction to optical path)

        FSR_lambda = lambda^2 / (2nL cos(theta)) (1+ (lambda^2/(2pi n L cos(theta)) dPhi/dlambda))
        """
        n = self.get_refractive_index_cavity(self.wavelength)
        # First, compute the base term: lambda^2 in meters
        lambda_sq_m2 = (self.wavelength * 1e-9) ** 2
        cos_theta = np.cos(np.deg2rad(self.aoi))

        # Base FSR in meters
        base_fsr_m = lambda_sq_m2 / (2 * n * self.d_spacer * cos_theta)

        # Correction term due to mirror phase dispersion
        correction_factor = (
            lambda_sq_m2 / (2 * np.pi * n * self.d_spacer * cos_theta)
        ) * self.phase_spline.derivative(1)(self.wavelength)

        # FSR in meters including phase correction
        fsr_m = base_fsr_m * (1 + correction_factor)

        # Convert result to nanometers
        return fsr_m * 1e9

    @property
    def coefficient_of_finesse(self):
        """Returns the coefficient of finesse of the mirror coating stack

        The coefficient of finesse is given by the formula:

        F = 4 * R / (1 - R)^2

        where:
        R: mirror reflectivity

        """
        return 4.0 * self.mirror_reflectivity / (1.0 - self.mirror_reflectivity) ** 2

    @property
    def reflectivity_finesse(self):
        """Returns the reflectivity finesse of the mirror coating stack

        The reflectivity finesse is given by the formula:

        F = pi / (2 * arcsin(1 / sqrt(F)))

        where:
        F: coefficient of finesse

        """
        return np.pi / (2.0 * np.arcsin(1.0 / np.sqrt(self.coefficient_of_finesse)))

    def peak_wavelength_ideal(self):
        return lambda_peaks(
            self._m,
            d=self.d_spacer,
            n=self.get_refractive_index_cavity,
            aoi=np.deg2rad(self.aoi),
        )

    def peak_wavelength_real(self, m=None):
        if m is None:
            m = self._m
        return lambda_peaks(
            m,
            d=self.d_spacer,
            n=self.get_refractive_index_cavity,
            aoi=np.deg2rad(self.aoi),
            phase=self.phase_spline,
        )

    def residuals_ideal_vs_real(self):
        return self.peak_wavelength_ideal() - self.peak_wavelength_real()

    def set_data(self, wavelength):
        self.wavelength = wavelength
        self.normalized_wavelength = np.linspace(0, 1, len(wavelength))
        self._m = guess_m(
            wavelength,
            self.d_spacer,
            self.aoi,
            n=self.get_refractive_index_cavity,
            phase=self.phase_spline,
        )[0]

    def update(self):
        self._m = guess_m(
            self.wavelength,
            self.d_spacer,
            self.aoi,
            n=self.get_refractive_index_cavity,
            phase=self.phase_spline,
        )[0]

    def measured_residuals(self):
        return (
            (self.wavelength - self.peak_wavelength_real())
            / self.peak_wavelength_real()
            * c
        )

    def wavelength_vs_residuals_m_per_s(self):
        return (
            self.peak_wavelength_ideal(),
            self.residuals_ideal_vs_real() / self.peak_wavelength_ideal() * c,
        )

    def inc_m(self):
        self._m += 1

    def dec_m(self):
        self._m -= 1

    def guess_m(self, measured_wl, phase=None):
        if phase is None:
            phase = self.phase_spline
        return guess_m(
            measured_wl,
            self.d_spacer,
            np.deg2rad(self.aoi),
            self.get_refractive_index_cavity,
            phase=phase,
        )

    def transmission_spectrum(self, wavelength):
        """Returns the transmission spectrum of the entire etalon for the given wavelength.

        The transmission spectrum is calculated as:

        T = 1 / (1 + F * sin^2(pi * d / lambda + phi))

        where:
        T: transmission spectrum
        F: coefficient of finesse
        d: spacer thickness
        lambda: wavelength
        phi: phase shift upon reflection

        Args:
            wavelength: Wavelength in [nm]

        """
        F_spline = UnivariateSpline(
            self.wavelength, self.coefficient_of_finesse, k=3, s=0
        )

        return 1.0 / (
            1.0
            + F_spline(wavelength)
            * np.sin(np.pi * self.d_spacer * 1e9 / wavelength) ** 2
        )

    def load_parameters(self, filename: str | pathlib.Path):
        with open(filename, "r") as f:
            pars = json.load(f)
        self.all_parameters = np.array(pars["all_parameters"])


# def load_from_file(filename):
#     with open(filename, "r") as json_file:
#         return Etalon(**json.load(json_file))
