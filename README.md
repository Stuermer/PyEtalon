![Logo](logo.png)
# pyetalon 
PyEtalon is a Python library to analyse the multilayer interference of thin films in the context of etalons.

It provides an easy interface to calculate the phase shift of a known multilayer stack. Also, it allows fitting of the 
multilayer stack parameters to measured peak data of an etalon.

PyEtalon makes heavy use of PyIndexRepo, a python library to access the refractive index data from [RefractiveIndex.info](https://refractiveindex.info/).
See the documentation of PyIndexRepo for more information on how to access the refractive index data.

## User guide

### Background
The free spectral range (FSR) of an etalon is the distance between two adjacent transmission peaks.
For an ideal etalon, it is given by the following equation:

$FSR_{\nu}= \Delta \nu = \frac{c}{2nL\cdot \cos(\theta)}$

where $c$ is the speed of light in vacuum, $n$ is the refractive index of the medium between the plates, $\theta$ is the angle of incidence, and $L$ is the distance between the plates.

In wavelength, the FSR is given by:

$FSR_{\lambda} = \Delta \lambda = \frac{\lambda^2}{2nL \cos(\theta)}$

where $\lambda$ is the wavelength of light in vacuum.

Similarly, the absolute peak wavelength of the etalon is given by:

$\lambda = \frac{2nL \cos(\theta)}{m}$

where $m$ is the order of the interference peak.

All equations above are only valid for an ideal etalon. In reality, upon reflection on the etalon mirrors,
the light undergoes a phase shift. This phase shift is given by the Fresnel equations and depends on the refractive index of the materials involved.
As a consequence, both, the FSR and the absolute peak wavelength are modified by the phase shift.
It is most intuitive to think of the phase shift as a change in the effective optical path length or mirror separation $L$.

The peak wavelength is then given by:

$\lambda = \frac{2nL \cos(\theta)+ \lambda \frac{\Phi_{\lambda}}{\pi}}{m}$

### Example
Let's walk through the simulation of a simple etalon with a single layer of 30nm of silver on a glass substrate.
The etalon is illuminated with a plane wave at normal incidence. 

The wavelength dependent refractive indices of the three materials are represented by a Material object, 
which can be created from a tabulated data set or a formula or by querying the refractive index database.

To calculate optical properties of an etalon, create an `Etalon` object and use the provided helper functions.

```python
import numpy as np

from pyetalon.etalon import Etalon
from pyetalon.plotting import plot_fsr
from pyindexrepo import RefractiveIndexLibrary, Material, TabulatedIndexData, FormulaIndexData
from pyindexrepo.dispersion_formulas import formula_2


# Create vacuum 'material' by using a tabulated data set
Vacuum = Material(TabulatedIndexData([.1, 10.000], [1.0, 1.0]))
# Create the substrate material by using the Sellmeier formula and vendor data for Suprasil
Suprasil = Material(FormulaIndexData(formula_2, [0., 6.72472034E-01, 4.50684530E-03, 4.31646851E-01, 1.33090179E-02, 8.85914296E-01, 9.67375952E+01], .185, 2.326), None)
# Create the silver material by querying the refractive index database, selecting the Ciesielski data set for thin films
db = RefractiveIndexLibrary()
Silver = db.get_material('main', 'Ag', 'Ciesielski')


# Create an etalon object
silver_etalon = Etalon(
    [Vacuum, Silver, Suprasil],                     # specify the layer materials
    ["Vacuum", "Silver", "Substrate"],              # specify the layer names
    [-np.inf, 30., np.inf],                         # specify the layer thicknesses
    [0, 1, 2],                                      # specify the layer order/indices
    wavelength=np.linspace(500, 1100, 5000),        # specify the wavelength range
    d_spacer=5E-3,                                  # specify the spacer thickness in meters
    aoi=.0,                                         # specify the angle of incidence in degrees
    identifier='Silver 30nm'                        # specify the identifier of the etalon
)
```

