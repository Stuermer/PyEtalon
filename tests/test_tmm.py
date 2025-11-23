import numpy as np

from PyEtalon.tmm import rt


def test_tmm():
    n = np.array([[1.0, 1.5]], dtype=np.complex128)
    d = np.array([-np.inf, np.inf])
    wl = np.array([0.5])
    aoi = 20.0
    r, t = rt(n, d, wl, aoi, 0)
    R = np.abs(r) ** 2

    assert np.isclose(R, 0.0471, atol=1e-4)

    # test other polarization
    r, t = rt(n, d, wl, aoi, 1)
    R = np.abs(r) ** 2
    assert np.isclose(R, 0.0335, atol=1e-4)


if __name__ == "__main__":
    test_tmm()
