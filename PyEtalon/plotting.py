from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, cm

from PyEtalon.etalon import Etalon

c = 3e8


def plot_phase(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Unwrapped Phase Shift of coating")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("phase shift [rad]")
    for ee in e:
        plt.plot(ee.wavelength, ee.phase, label=ee.identifier)
    plt.legend()


def plot_gd(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("GD of coating")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("GD")
    for ee in e:
        plt.plot(ee.wavelength, ee.gd, label=ee.identifier)
    plt.legend()


def plot_gdd(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("GDD of coating")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("GDD")
    for ee in e:
        plt.plot(ee.wavelength, ee.gdd, label=ee.identifier)
    plt.legend()


def plot_layer_thickness_influence(e: Etalon, d_change=1.0):
    n_layer = e.num_layers

    custommap = plt.get_cmap("RdBu")
    plt.figure()
    ax = plt.axes()
    cmm = cm.rainbow
    ax.set_prop_cycle("color", [cmm(i) for i in np.linspace(0, 1, n_layer)])

    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_layer), custommap.N)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmm)
    e.update()
    wl, design = e.wavelength_vs_residuals_m_per_s()
    for i in range(n_layer):
        corrections = np.zeros_like(e.d_corrections)
        corrections[i] = d_change
        e.d_corrections = corrections
        # e.update()
        plt.plot(wl, design - e.wavelength_vs_residuals_m_per_s()[1])
    plt.title(f"Layer {d_change} nm thickness change - {e.identifier}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Peak shift vs. design values [m/s]")
    cbar = plt.colorbar(mapper, ticks=np.linspace(0, n_layer - 1, n_layer), ax=ax)
    cbar.ax.set_yticklabels([f"{i + 1}" for i in range(n_layer)])
    cbar.set_label("Layer number")


def plot_layer_thickness_influence_gd(e: Etalon, d_change=1.0):
    n_layer = e.num_layers

    custommap = plt.get_cmap("RdBu")
    plt.figure()
    ax = plt.axes()
    cmm = cm.rainbow
    ax.set_prop_cycle("color", [cmm(i) for i in np.linspace(0, 1, n_layer)])

    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_layer), custommap.N)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmm)
    wl = e.wavelength
    design = e.gd
    for i, ddd in enumerate(range(1)):
        pars = np.zeros_like(e.all_parameters)
        pars[i] += d_change
        e.all_parameters = pars

        plt.plot(wl, e.gd - design)
        plt.title(f"Layer {d_change} nm thickness change")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("GD chang")
    cbar = plt.colorbar(mapper, ticks=np.linspace(0, n_layer - 1, n_layer), ax=ax)
    cbar.ax.set_yticklabels([f"{i + 1}" for i in range(n_layer)])
    cbar.set_label("Layer number")

    plt.tight_layout()


def plot_layer_thickness_influence_reflectivity(e: Etalon, d_change=1.0):
    n_layer = e.num_layers

    custommap = plt.get_cmap("RdBu")
    plt.figure()
    ax = plt.axes()
    cmm = cm.rainbow
    ax.set_prop_cycle("color", [cmm(i) for i in np.linspace(0, 1, n_layer)])

    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_layer), custommap.N)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmm)
    wl = e.wavelength
    design = e.mirror_reflectivity
    plt.plot(wl, design)
    design_params = e.all_parameters.copy()
    for i, ddd in enumerate(range(n_layer)):
        pars = design_params.copy()
        pars[i] += d_change
        e.all_parameters = pars

        plt.plot(wl, e.mirror_reflectivity)
        plt.title(f"Layer {d_change} nm thickness change")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectivity")
    cbar = plt.colorbar(mapper, ticks=np.linspace(0, n_layer - 1, n_layer), ax=ax)
    cbar.ax.set_yticklabels([f"{i + 1}" for i in range(n_layer)])
    cbar.set_label("Layer number")

    plt.tight_layout()


def plot_reflectivity(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Mirror Reflectivity")
    for ee in e:
        plt.plot(ee.wavelength, ee.mirror_reflectivity, label=ee.identifier)
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectivity ")


def plot_mirror_transmission(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Mirror Transmission")
    for ee in e:
        plt.plot(ee.wavelength, ee.mirror_transmission, label=ee.identifier)
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Mirror Transmission ")


def plot_absorption(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Mirror Absorption")
    for ee in e:
        plt.plot(
            ee.wavelength,
            1.0 - ee.mirror_transmission - ee.mirror_reflectivity,
            label=ee.identifier,
        )
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Mirror Absorption ")


def plot_normalized_fsr(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Noramlized FSR")
    for ee in e:
        # plot the FSR normalized to the nominal FSR
        plt.plot(ee.wavelength, ee.fsr / ee.ideal_fsr, label=ee.identifier)
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized FSR (FSR / ideal FSR)")


def plot_fsr(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("FSR")
    for ee in e:
        plt.plot(ee.wavelength, ee.fsr, label=ee.identifier)
    plt.legend()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("FSR [GHz]")


def plot_refractive_index_materials(
    etalon, wavelength, exclude_outer_two_materials=True, plot_k=True
):
    wl = wavelength.copy()
    if exclude_outer_two_materials:
        mats = etalon.materials[1:-1].copy()
        mats_names = etalon.names[1:-1].copy()
    else:
        mats = etalon.materials.copy()
        mats_names = etalon.names.copy()

    fig, axes = plt.subplots(max(len(mats), 2), sharex=True)
    fig.suptitle("Refractive indices")
    for i, mat in enumerate(mats):
        (p1,) = axes[i].plot(wl, mat.get_n(wavelength), label="n")
        axes[i].text(0.05, 0.15, mats_names[i], transform=axes[i].transAxes)
        axes[i].set_ylabel("n")

        if plot_k:
            ax_k = axes[i].twinx()
            ax_k.set_ylabel("k")
            (p2,) = ax_k.plot(wl, mat.get_k(wavelength), "g", label="k")
            axes[i].legend([p1, p2], [plot.get_label() for plot in [p1, p2]])

    axes[len(mats) - 1].set_xlabel("Wavelength [nm]")
    # plt.show()


def plot_stack(ee: list[Etalon]):
    minx = 0
    maxx = 0
    fig, ax = plt.subplots()
    plt.title("Coating Stack")
    for ii, e in enumerate(ee):
        all_names = np.empty(shape=(len(e.d_stack_design)), dtype="<U10")
        for i in np.unique(e.idx_stack):
            all_names[i == e.idx_stack] = e.names[i]

        # ax.set_title(f"Coating Stack {e.identifier}")

        # replace inf length in d_stack with 2x maximum length in d_stack for plotting
        dlist_inf_removed = e.d_stack_design[1:-1].copy()
        dlist_inf_removed = np.append(
            dlist_inf_removed, 1.3 * np.max(dlist_inf_removed)
        )
        dlist_inf_removed = np.insert(dlist_inf_removed, 0, np.max(dlist_inf_removed))
        dlist_cummulative = np.cumsum(dlist_inf_removed)
        dlist_cummulative -= dlist_cummulative[0]

        left = np.array(dlist_cummulative[:-1])
        right = np.array(dlist_cummulative[1:])
        left = np.insert(left, 0, -dlist_inf_removed[0])
        right = np.append(right, dlist_inf_removed[-1] + dlist_cummulative[-1])
        bottom = np.zeros(len(left)) + ii
        top = bottom + 1

        colors = ["gray", "lightblue", "lightgreen", "darkgray"]
        unique_names = np.array(e.names)

        for i, (l, r, b, t, name) in enumerate(
            zip(left, right, bottom, top, all_names)
        ):
            idx_name = np.where(unique_names == name)[0][0]
            ax.add_patch(
                patches.Rectangle(
                    (l, b),
                    r - l,
                    t - b,
                    edgecolor="k",
                    facecolor=colors[idx_name],
                )
            )
            ax.text(l + 10, 0.05 + ii, name, rotation=90)

        # ax.yaxis.set_visible(False)
        ax.set_xlabel("Accum. thickness [nm]")

        # style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        # kw = dict(arrowstyle=style, color="k")

        # ax.add_patch(patches.FancyArrowPatch((500, 0.25), (500, 0.75), connectionstyle="arc3,rad=2", **kw))
        if left[0] < minx:
            minx = left[0]
        if right[-1] > maxx:
            maxx = right[-1]

    # add name of etalon on y-axis instead of numbers, center the text vertically
    ax.set_yticks(np.array(list(range(len(ee)))) + 0.5)
    ax.set_yticklabels([e.identifier for e in ee])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(0.0, len(ee))

    # plt.show()


def plot_reflectivity_finesse(e: Union[Etalon, List[Etalon]]):
    e = [e] if isinstance(e, Etalon) else e
    plt.figure()
    plt.title("Finesse")
    for ee in e:
        plt.plot(ee.wavelength, ee.reflectivity_finesse, label=ee.identifier)
    plt.legend()
