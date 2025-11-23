from pyindexrepo.dispersion_formulas import formula_2
from pyindexrepo.main import (
    FormulaIndexData,
    Material,
    RefractiveIndexLibrary,
    TabulatedIndexData,
)

db = RefractiveIndexLibrary()
Silver = db.get_material("main", "Ag", "Ciesielski")
Gold = db.get_material("main", "Au", "Yakubovsky-25nm")

Vacuum = Material(
    TabulatedIndexData(
        [0.1, 0.2, 0.3, 0.5, 10.000], [1.0, 1.0, 1.0, 1.0, 1.0], "interp1d", True
    ),
    None,
    name="Vacuum",
)

Suprasil = Material(
    FormulaIndexData(
        formula_2,
        [
            0.0,
            6.72472034e-01,
            4.50684530e-03,
            4.31646851e-01,
            1.33090179e-02,
            8.85914296e-01,
            9.67375952e01,
        ],
        0.185,
        2.326,
    ),
    None,
    name="Substrate",
)

SiO2 = Material(
    TabulatedIndexData(
        [
            0.4,
            0.43157894737,
            0.46315789474,
            0.49473684211,
            0.52631578947,
            0.55789473684,
            0.58947368421,
            0.62105263158,
            0.65263157895,
            0.68421052632,
            0.71578947368,
            0.74736842105,
            0.77894736842,
            0.81052631579,
            0.84210526316,
            0.87368421053,
            0.90526315789,
            0.93684210526,
            0.96842105263,
            1.0,
        ],
        [
            1.47013366,
            1.46705551,
            1.4646206,
            1.4626252,
            1.46097832,
            1.45958769,
            1.45839856,
            1.45736981,
            1.45646412,
            1.45566345,
            1.45494223,
            1.45428953,
            1.4536911,
            1.45313792,
            1.4526227,
            1.45213777,
            1.45167956,
            1.45124205,
            1.45082263,
            1.45041761,
        ],
        "interp1d",
        True,
    ),
    None,
    name="SiO2",
)
