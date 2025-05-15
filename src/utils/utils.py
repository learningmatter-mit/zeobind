import os
import sys
from datetime import datetime
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import yaml
from collections import defaultdict
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List
from rdkit import Chem
from rdkit.Chem import RemoveAllHs, Draw
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from torch.utils.data import Dataset
from zeobind.src.utils.logger import log_msg
from zeobind.src.utils.default_utils import (
    LOAD_NORM_BINS_FILE,
    MPL_FONTS_DIR,
)
from zeobind.src.utils.mol_features import compute_osda_features
from zeobind.src.utils.default_utils import DEFAULT_ZEOLITE_PRIOR_FILE

PAIR_COLS = ["SMILES", "Zeolite"]

SCALERS = dict(
    standard=StandardScaler,
    minmax=MinMaxScaler,
)
INPUT_SCALER_FILE = "input_scaling.json"
OUTPUT_SCALER_FILE = "truth_scaling.json"


def sort_by(x, idx=2):
    """Only works if the basename has the form 'x_y_z' where y is the index to sort by."""
    return int(os.path.basename(x).split("_")[idx])


def filter_smis_by_charge(smiles, lowc=0.0, highc=1.0):
    charges = [x.count("+]") - x.count("-]") for x in smiles]
    return [smiles[i] for i, c in enumerate(charges) if c >= lowc and c <= highc]


def get_competition(mat):
    competition_energy = mat.apply(lambda row: row.nsmallest(2).max(), axis=1)
    competition = mat - competition_energy.values.reshape(-1, 1)
    return competition


def create_inputs(kwargs, indices, save=False):
    """
    Args:
    kwargs (dict): Dictionary of arguments.

    Returns:
    X (pd.DataFrame): DataFrame of input features.
    """
    osda_indices = indices.get_level_values(PAIR_COLS[0]).to_list()
    zeolite_indices = indices.get_level_values(PAIR_COLS[1]).to_list()

    # get prior file names and check if they already exist. If they do, load them. If not, process them.

    osda_filename = f"{kwargs['output']}/{os.path.basename(kwargs['osda_prior_file'])}"
    if not ".pkl" in osda_filename:
        osda_filename += ".pkl"
    if os.path.exists(osda_filename):
        kwargs["osda_prior_file"] = osda_filename
        save = False 
    
    zeolite_filename = f"{kwargs['output']}/{os.path.basename(kwargs['zeolite_prior_file'])}"
    if not ".pkl" in zeolite_filename:
        zeolite_filename += ".pkl"
    if os.path.exists(zeolite_filename):
        kwargs["zeolite_prior_file"] = zeolite_filename
        save = False

    osda_inputs = process_prior_file(kwargs["osda_prior_file"], is_zeolite=False)
    zeolite_inputs = process_prior_file(kwargs["zeolite_prior_file"])

    if kwargs.get("osda_prior_map"):
        with open(kwargs["osda_prior_map"], "r") as f:
            cols = json.load(f)
        osda_inputs = osda_inputs[cols.keys()]
    if kwargs.get("zeolite_prior_map"):
        with open(kwargs["zeolite_prior_map"], "r") as f:
            cols = json.load(f)
        zeolite_inputs = zeolite_inputs[cols.keys()]

    if save:
        osda_inputs.to_pickle(osda_filename)
        zeolite_inputs.to_pickle(zeolite_filename)

    osda_inputs = osda_inputs.loc[osda_indices].reset_index(drop=True)
    zeolite_inputs = zeolite_inputs.loc[zeolite_indices].reset_index(drop=True)
    inputs = pd.concat([osda_inputs, zeolite_inputs], axis=1).set_index(indices)

    return inputs


def process_prior_file(file, is_zeolite=True):
    """
    Read the input file. The file can either already contain the framework/ molecule features, or be a list of framework names/ molecule smiles. If it is the latter, the features will be computed.
    """
    _, file_ext = os.path.splitext(file)

    if file_ext == ".csv":
        return pd.read_csv(file, index_col=0)

    elif file_ext == ".pkl":
        return pd.read_pickle(file)
    
    else:
        # not a file of features but just a list of SMILES or Framework names
        with open(file, "r") as f:
            lines = f.readlines()
        identities = [x.strip() for x in lines]
        if is_zeolite:
            priors = pd.read_pickle(DEFAULT_ZEOLITE_PRIOR_FILE)
            return priors.loc[identities]
        else: 
            priors = compute_osda_features(identities)
            return priors


def get_load_norm_bins(file=LOAD_NORM_BINS_FILE):  
    """
    Get loading related bins for loading multilassification task.

    Args:
    file (str): Path to yaml file containing loading bins.

    Returns:
    np.array: Array of loading bins.
    dict: Dictionary mapping between loading class to actual loading.
    np.array: Array of histogram bins for plotting.
    """
    with open(file, "r") as f:
        load_norm_bins = yaml.load(f, Loader=yaml.FullLoader)
    bins = np.array(load_norm_bins)

    # Map between loading class to actual loading
    bins_dict = dict(zip(np.array(range(len(bins))), bins))

    # Histogram bins for plotting
    histogram_bins = np.append(np.insert(bins, 0, -1), 22)
    histogram_bins += 0.05

    return bins, bins_dict, histogram_bins


def cluster_isomers(smiles: Iterable) -> Dict[str, List[str]]:
    """
    Cluster isomers based on SMILES strings.

    Args:
    smiles (Iterable): List of SMILES strings.

    Returns:
    dict: Dictionary mapping between non-isomeric InchiKey to list of stereochemistry-containing InchiKeys.
    """

    nonisomeric_smiles_lookup = defaultdict(set)
    for smile in smiles:
        m = Chem.MolFromSmiles(smile)
        m = RemoveAllHs(m)
        # Remove isomeric information
        relaxed_smiles = Chem.rdmolfiles.MolToSmiles(m, isomericSmiles=False)
        nonisomeric_smiles_lookup[relaxed_smiles].add(smile)
    log_msg(
        "cluster_isomers",
        "Number of nonisomeric SMILES:",
        len(nonisomeric_smiles_lookup),
    )
    return nonisomeric_smiles_lookup


class MultiTaskTensorDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


def save_scaler(scaler, scaler_type, op_folder, filename=OUTPUT_SCALER_FILE):
    if scaler_type == "standard":
        scaler_dict = {
            "scaler_type": "standard",
            "mean": scaler.mean_.tolist(),
            "var": scaler.var_.tolist(),
        }
    elif scaler_type == "minmax":
        scaler_dict = {
            "scaler_type": "minmax",
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
        }
    else:
        raise ValueError(f"Invalid scaler type {scaler_type}")

    filename = os.path.join(op_folder, filename)
    with open(filename, "w") as f:
        json.dump(scaler_dict, f)


def scale_data(data, scaler):
    """
    Scale data.
    - scaler: dict
    """
    if scaler["scaler_type"] == "minmax":
        return (data - np.array(scaler["data_min"])) / (
            np.array(scaler["data_max"]) - np.array(scaler["data_min"])
        )
    elif scaler["scaler_type"] == "standard":
        data_scaled = (data - np.array(scaler["mean"])) / np.sqrt(
            np.array(scaler["var"])
        )
        # NaN if var is NaN. Fill with zero
        data_scaled = data_scaled.fillna(0.0)
        return data_scaled


def rescale_data(data, scaler):
    """
    Apply inverse scaling to data.
    - scaler: dict
    """
    if scaler["scaler_type"] == "minmax":
        return data * (
            np.array(scaler["data_max"]) - np.array(scaler["data_min"])
        ) + np.array(scaler["data_min"])
    elif scaler["scaler_type"] == "standard":
        return data * np.sqrt(np.array(scaler["var"])) + np.array(scaler["mean"])


def get_prior_files(file=None, folder=None):
    """Get prior files from either a single file or a folder of files."""
    if file is not None:
        if type(file) is str:
            file = [file]
        elif type(file) is list:
            pass
        else:
            raise ValueError("Invalid file type")
        return file
    elif folder is not None:
        files = glob.glob(os.path.join(folder, "*"))
        # expects prior files to be in pickle format
        files = [f for f in files if ".pkl" in f]
        return files
    else:
        raise ValueError("No files to predict on")


cm = 1 / 2.54  # centimeters in inches


def setup_mpl():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    if os.path.exists(MPL_FONTS_DIR):
        # Register the fonts with Matplotlib
        for font_dir in os.listdir(MPL_FONTS_DIR):
            font_path = os.path.join(MPL_FONTS_DIR, font_dir)
            if os.path.isdir(font_path):
                for font_file in os.listdir(font_path):
                    if font_file.endswith(".ttf"):
                        fm.fontManager.addfont(os.path.join(font_path, font_file))

        # Set the default font for the entire notebook
        plt.rcParams["font.family"] = (
            "Avenir"  # Replace 'YourFontName' with the name of your font
        )


def show_mols(smis, legend=None, molsperrow=10, maxmols=50, subimgsize=(200, 200)):
    """Show molecules in a grid."""
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    if not legend:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=molsperrow,
            subImgSize=subimgsize,
            maxMols=maxmols,
        )
    else:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=molsperrow,
            subImgSize=subimgsize,
            maxMols=maxmols,
            legends=legend,
        )
    return img


def save_mpl_fig(fig, basename, ending=".pdf", dpi=300):
    fig.savefig(basename + ending, bbox_inches="tight", dpi=dpi)


def format_axs(
    axs,
    xtick_size,
    ytick_size,
    spines_width,
    xlabel,
    ylabel,
    xlabel_size,
    ylabel_size,
    limits=None,
    tick_width=2,
    tick_size=6,
    weight=None,
):
    axs.tick_params(axis="x", which="major", labelsize=xtick_size)
    axs.tick_params(axis="y", which="major", labelsize=ytick_size)
    axs.xaxis.set_tick_params(width=tick_width, size=tick_size)
    axs.yaxis.set_tick_params(width=tick_width, size=tick_size)

    # axs.tick_params(axis='both', which='minor="something", labelsize=8)
    for axis in ["top", "bottom", "left", "right"]:
        axs.spines[axis].set_linewidth(spines_width)

    axs.set_xlabel(xlabel, fontsize=xlabel_size, weight=weight)
    axs.set_ylabel(ylabel, fontsize=ylabel_size, weight=weight)

    if not limits:
        return axs
    if "x" in limits.keys():
        axs.set_xlim(limits["x"])
    if "y" in limits.keys():
        axs.set_ylim(limits["y"])

    return axs


def get_cb(
    fig, sc, axs, label, ticks=None, linewidth=2, tickwidth=2, labelsize=18, labelpad=20, cmap='viridis',
):
    """Get colorbar"""
    sc.set_cmap(cmap)
    cb = fig.colorbar(sc, ax=axs)
    cb.set_label(label, fontsize=labelsize)
    cb.outline.set_linewidth(linewidth)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(width=tickwidth)
    cb.ax.tick_params(labelsize=labelsize)
    # pad
    cb.ax.xaxis.labelpad = labelpad
    return cb


def get_legend(
    fig,
    bbox_to_anchor=(0.5, 1.1),
    fontsize=15,
    loc="upper center",
    ncol=2,
    axs=None,
    format_handles=False,
    legend_colors=None,
    linewidth=2,
    edgecolor="k",
    put_where="fig",
):
    """Get legend."""
    positions = {"fig": fig, "axs": axs}
    legend = positions[put_where].legend(
        fontsize=fontsize,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
    )

    if format_handles:
        for idx, handle in enumerate(legend.legendHandles):
            handle.set_color(legend_colors[idx])
            handle.set_linewidth(linewidth)
            handle.set_edgecolor(edgecolor)

    return legend


def get_color_values(num_colors, c="viridis"):
    """Get a list of color values from a colormap"""
    cmap = plt.cm.get_cmap(c)
    color_values = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return color_values


def draw_parity(axs, xlimits, ylimits, lw=1):
    """Draw a parity line on a plot"""
    limits = [min(xlimits[0], ylimits[0]), max(xlimits[1], ylimits[1])]
    axs.plot(limits, limits, "k--", lw=lw)
    return axs

def create_cmap_from_colors(cmap_name, colors):
    """
    Create a colormap from a list of colors.
    Args:
        cmap_name (str): Name of the colormap.
        colors (list): List of colors.
    Returns:
        LinearSegmentedColormap: Colormap object.
    Example:
        colors = np.array([
            (247,251,255),
            (222,235,247),
            (198,219,239),
            (158,202,225),
            (107,174,214),
            (66,146,198),
            (33,113,181),
            (8,81,156),
            (8,48,107),
        ]) / 256
        cmap_name = "blue_white"
        cmap = create_cmap_from_colors(cmap_name, colors)
    """
    return LinearSegmentedColormap.from_list(cmap_name, colors)

def restrict_cmap(cmap, min_val, max_val):
    """
    Restrict a colormap to a certain range.
    Args:
        cmap: Colormap to restrict.
        min_val: Minimum value of the colormap.
        max_val: Maximum value of the colormap.
    Returns:
        Restricted colormap.
    
    Example:
        cmap = plt.get_cmap("viridis")
        restricted_cmap = restrict_cmap(cmap, 0.2, 0.8)
    """
    new_colors = cmap(np.linspace(min_val, max_val, 256))
    return LinearSegmentedColormap.from_list("new_viridis", new_colors)