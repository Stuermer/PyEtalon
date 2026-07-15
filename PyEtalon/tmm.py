import PyEtalon._jax_config  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.typing import ArrayLike


@jax.jit
def rt_jax(
    n: ArrayLike,
    d: ArrayLike,
    wvl: ArrayLike,
    aoi: float = 0.0,
    pol: int = 0,
):
    nw = wvl.shape[0]

    # Broadcast wavelength-independent refractive indices
    if n.shape[0] == 1:
        n = jnp.broadcast_to(n, (nw, n.shape[1]))

    sin2 = jnp.sin(jnp.deg2rad(aoi)) ** 2

    n2 = n * n
    n0 = n2[:, :1]

    s2 = n0 * sin2 / n2
    cosj = jnp.sqrt(1.0 - s2)

    nj = n[:, :-1]
    nk = n[:, 1:]

    cj = cosj[:, :-1]
    ck = cosj[:, 1:]

    ############################
    # Fresnel coefficients
    ############################

    den_te = nj * cj + nk * ck
    den_tm = nj * ck + nk * cj

    r_te = (nj * cj - nk * ck) / den_te
    t_te = 2 * nj * cj / den_te

    r_tm = (nj * ck - nk * cj) / den_tm
    t_tm = 2 * nj * ck / den_tm

    rjk = lax.select(pol == 0, r_te, r_tm)
    tjk = lax.select(pol == 0, t_te, t_tm)

    inv_t = 1.0 / tjk

    ############################
    # Interface matrices
    ############################

    A = jnp.stack(
        (
            jnp.stack((inv_t, rjk * inv_t), axis=-1),
            jnp.stack((rjk * inv_t, inv_t), axis=-1),
        ),
        axis=-2,
    )
    # (Nwvl,Ninterfaces,2,2)

    ############################
    # No internal layers
    ############################

    if d.shape[0] == 2:
        M = A[:, 0]

    else:
        kz = jnp.sqrt(n2[:, 1:-1] - n0 * sin2) * d[1:-1]

        beta = 2j * jnp.pi * kz / wvl[:, None]

        em = jnp.exp(-beta)
        ep = jnp.exp(beta)

        A1 = A[:, 1:]

        # Multiply propagation matrix into interface matrix analytically
        B = jnp.stack(
            (
                em[..., None] * A1[:, :, 0],
                ep[..., None] * A1[:, :, 1],
            ),
            axis=-2,
        )

        # scan over layers
        B = jnp.swapaxes(B, 0, 1)

        def body(M, Bj):
            return jnp.matmul(M, Bj), None

        M, _ = lax.scan(body, A[:, 0], B)

    r = M[:, 1, 0] / M[:, 0, 0]
    t = 1.0 / M[:, 0, 0]

    return r, t


def rt(
    n,
    d,
    wvl,
    aoi=0.0,
    pol=0,
):
    """Computes the transmission and reflection coefficients for a multilayer stack.

    Args:
        n: Complex refractive index of the stack (N_wvl x N_layers) or (1 x N_layers)
        d: Physical thickness of the layers (N_layers)
        wvl: Wavelength vector (N_wvl)
        aoi: Angle of incidence [deg]
        pol: Polarization (0: TE, 1: TM)

    Returns:
        Transmission and reflection coefficients
    """
    if not isinstance(n, jnp.ndarray):
        n = jnp.array(n)
    if not isinstance(d, jnp.ndarray):
        d = jnp.array(d)
    if not isinstance(wvl, jnp.ndarray):
        wvl = jnp.array(wvl)

    r, t = rt_jax(
        n,
        d,
        wvl,
        aoi,
        pol,
    )
    return np.asarray(r), np.asarray(t)
