import numpy as np
from numba import njit


@njit(cache=True)
def rt(
    n: np.ndarray, d: np.ndarray, wvl: np.ndarray, aoi: float = 0.0, pol: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the reflection and transmission coefficients for a multilayer.

    Parameters:
        n: 2D array-like, shape (num_wavelengths, num_layers)
            Refractive indices for each layer and wavelength.
        d: 1D array-like, shape (num_layers)
            Thickness of each layer (units must match wavelength).
        wvl: 1D array-like, shape (num_wavelengths)
            Wavelengths of light (units must match thickness).
        aoi: float, optional
            Angle of incidence in degrees. Default is 0.0.
        pol: int, optional
            Polarization state, 0 for TE (s) or 1 for TM (p). Default is 0.

    Returns:
        r: 1D array, shape (num_wavelengths)
            Reflection coefficients for each wavelength.
        t: 1D array, shape (num_wavelengths)
            Transmission coefficients for each wavelength.
    """

    # Calculate squared sine of angle of incidence
    sin_squared_aoi = np.sin(np.radians(aoi)) ** 2
    n_squared = n**2

    # Precompute interface matrices
    n_current_layer, n_next_layer = n[:, :-1], n[:, 1:]
    s2_per_layer = (n_squared[:, 0:1] * sin_squared_aoi) / n_squared
    cos_current_layer, cos_next_layer = (
        np.sqrt(1 - s2_per_layer[:, :-1]),
        np.sqrt(1 - s2_per_layer[:, 1:]),
    )

    num1 = n_current_layer * cos_current_layer + n_next_layer * cos_next_layer
    num2 = n_current_layer * cos_next_layer + n_next_layer * cos_current_layer

    rjk_TE = (
        n_current_layer * cos_current_layer - n_next_layer * cos_next_layer
    ) / num1
    tjk_TE = 2.0 * n_current_layer * cos_current_layer / num1
    rjk_TM = (
        n_current_layer * cos_next_layer - n_next_layer * cos_current_layer
    ) / num2
    tjk_TM = 2.0 * n_current_layer * cos_next_layer / num2

    rjk = rjk_TE if pol == 0 else rjk_TM
    tjk = tjk_TE if pol == 0 else tjk_TM

    # Precompute layer matrices
    wave_vector_z = (
        np.sqrt(n_squared[:, 1:-1] - n_squared[:, 0:1] * sin_squared_aoi) * d[1:-1]
    )
    Bj = 2j * np.pi * wave_vector_z / wvl[:, None]
    exp_neg_Bj, exp_pos_Bj = np.exp(-Bj), np.exp(Bj)

    # Total transfer matrix initialization
    S11, S12 = 1.0 / tjk[:, 0], rjk[:, 0] / tjk[:, 0]
    S21, S22 = S12, S11

    for j in range(d.shape[0] - 2):  # Loop over middle layers
        B11, B12 = (
            exp_neg_Bj[:, j] / tjk[:, j + 1],
            exp_neg_Bj[:, j] * rjk[:, j + 1] / tjk[:, j + 1],
        )
        B21, B22 = (
            exp_pos_Bj[:, j] * rjk[:, j + 1] / tjk[:, j + 1],
            exp_pos_Bj[:, j] / tjk[:, j + 1],
        )

        # Update total matrix S = S * B
        C11 = S11 * B11 + S12 * B21
        C12 = S11 * B12 + S12 * B22
        C21 = S21 * B11 + S22 * B21
        C22 = S21 * B12 + S22 * B22

        S11, S12, S21, S22 = C11, C12, C21, C22

    # Final calculation for reflection and transmission
    r = S21 / S11
    t = 1.0 / S11

    return r, t


#
#
# @cuda.jit
# def rt_cuda(n, d, wvl, aoi, pol, r_real, r_imag, t_real, t_imag):
#     """
#     CUDA kernel to calculate the reflection and transmission coefficients for a multilayer.
#
#     Parameters:
#         n: 2D device array, shape (num_wavelengths, num_layers)
#             Refractive indices for each layer and wavelength.
#         d: 1D device array, shape (num_layers)
#             Thickness of each layer (units must match wavelength).
#         wvl: 1D device array, shape (num_wavelengths)
#             Wavelengths of light (units must match thickness).
#         aoi: float
#             Angle of incidence in degrees.
#         pol: int
#             Polarization state, 0 for TE (s) or 1 for TM (p).
#         r_real, r_imag: 1D device arrays
#             Output arrays for real and imaginary parts of reflection coefficients.
#         t_real, t_imag: 1D device arrays
#             Output arrays for real and imaginary parts of transmission coefficients.
#     """
#     idx = cuda.grid(1)
#     if idx < wvl.shape[0]:
#         # Constants
#         s2 = (np.sin(aoi * np.pi / 180))**2  # Squared sine of the angle of incidence
#         num_layers = d.shape[0]
#
#         # Precompute n^2 and angle terms
#         n2 = cuda.local.array(32, dtype=np.complex128)  # Max layers = 32
#         for j in range(num_layers):
#             n2[j] = n[idx, j] * n[idx, j]
#         s2_per_layer = (n2[0] * s2) / n2
#
#         # Compute cosines for layers
#         cj = cuda.local.array(32, dtype=np.float32)
#         for j in range(num_layers):
#             cj[j] = np.sqrt(1.0 - s2_per_layer[j])
#
#         # Initialize S-matrix
#         S11_real, S11_imag = 1.0, 0.0
#         S21_real, S21_imag = 0.0, 0.0
#
#         for j in range(num_layers - 1):
#             nj, nk = n[idx, j], n[idx, j + 1]
#             cj_j, cj_k = cj[j], cj[j + 1]
#
#             # Fresnel coefficients
#             num1 = nj * cj_j + nk * cj_k
#             num2 = nj * cj_k + nk * cj_j
#
#             if pol == 0:  # TE polarization
#                 rjk = (nj * cj_j - nk * cj_k) / num1
#                 tjk = 2.0 * nj * cj_j / num1
#             else:  # TM polarization
#                 rjk = (nj * cj_k - nk * cj_j) / num2
#                 tjk = 2.0 * nj * cj_k / num2
#
#             # Phase term
#             if j > 0:
#                 kj = np.sqrt(n2[j] - n2[0] * s2) * d[j]
#                 phase = 2.0 * np.pi * kj / wvl[idx]
#                 cos_phase = np.cos(phase)
#                 sin_phase = np.sin(phase)
#             else:
#                 cos_phase = 1.0
#                 sin_phase = 0.0
#
#             # Update S-matrix
#             T11_real = cos_phase / tjk
#             T11_imag = -sin_phase / tjk
#             T21_real = T11_real * rjk
#             T21_imag = T11_imag * rjk
#
#             S11_new_real = S11_real * T11_real - S11_imag * T11_imag
#             S11_new_imag = S11_real * T11_imag + S11_imag * T11_real
#             S21_new_real = S21_real * T11_real - S21_imag * T11_imag
#             S21_new_imag = S21_real * T11_imag + S21_imag * T11_real
#
#             S11_real, S11_imag = S11_new_real, S11_new_imag
#             S21_real, S21_imag = S21_new_real, S21_new_imag
#
#         # Final reflection and transmission coefficients
#         r_real[idx] = S21_real / (S11_real**2 + S11_imag**2)
#         r_imag[idx] = S21_imag / (S11_real**2 + S11_imag**2)
#         t_real[idx] = S11_real / (S11_real**2 + S11_imag**2)
#         t_imag[idx] = S11_imag / (S11_real**2 + S11_imag**2)
#
# def rt_host(n, d, wvl, aoi=0.0, pol=0):
#     num_wavelengths = wvl.shape[0]
#
#     # Allocate device memory
#     n_device = cuda.to_device(n.astype(np.complex128))
#     d_device = cuda.to_device(d.astype(np.float32))
#     wvl_device = cuda.to_device(wvl.astype(np.float32))
#     r_real_device = cuda.device_array(num_wavelengths, dtype=np.float32)
#     r_imag_device = cuda.device_array(num_wavelengths, dtype=np.float32)
#     t_real_device = cuda.device_array(num_wavelengths, dtype=np.float32)
#     t_imag_device = cuda.device_array(num_wavelengths, dtype=np.float32)
#
#     # Configure CUDA kernel
#     threads_per_block = 256
#     blocks_per_grid = (num_wavelengths + threads_per_block - 1) // threads_per_block
#
#     # Launch kernel
#     rt_cuda[blocks_per_grid, threads_per_block](
#         n_device, d_device, wvl_device, aoi, pol,
#         r_real_device, r_imag_device, t_real_device, t_imag_device
#     )
#
#     # Copy results back to host
#     r_real = r_real_device.copy_to_host()
#     r_imag = r_imag_device.copy_to_host()
#     t_real = t_real_device.copy_to_host()
#     t_imag = t_imag_device.copy_to_host()
#
#     # Combine real and imaginary parts into complex arrays
#     r = r_real + 1j * r_imag
#     t = t_real + 1j * t_imag
#
#     return r, t
#
