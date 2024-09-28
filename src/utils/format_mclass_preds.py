from scipy import stats 
import glob
import os
import numpy as np
import pandas as pd
import time
import argparse
from zeobind.src.utils.utils import sort_by, get_load_norm_bins

# ## Workflow
# 1. single E * single B, the files should match up
# 2. one matrix of (single E * single B)
#     1. All models + mean + std
# 3. one matrix of binary means and stds
# 4. one matrix of energy means and stds


def check_same_file(file1, file2):
    file1 = os.path.basename(file1)
    file2 = os.path.basename(file2)
    file1 = file1.split("_preds")[0]
    file2 = file2.split("_preds")[0]
    return file1 == file2


def make_mat(df, values, index="SMILES", columns="Zeolite"):
    return df.pivot(index=index, columns=columns, values=values)


def format_from_df(
    bemat, mdf, op_dir_loading, op_dir_beosda, zprior_file, to_save, basename, combine=True
):
    """Format ensemble predictions from binary and energy models.

    Arguments:
        bemat -- DataFrame containing formatted BE outputs, with SMILES in index and Zeolite in columns
        mdf -- DataFrame containing loading multiclassification outputs
        op_dir_loading -- directory to save loading matrices
        op_dir_beosda -- directory to save BE matrices
        to_save -- if True, saves matrices
        basename -- basename of file to save matrices as

    Keyword Arguments:
        combine -- if True, returns 6 DataFrames. Else, returns 6 dictionaries containing (smile type, DataFrame) key-value pairs. (default: {True})

    Returns:
        See combine argument
    """
    single_start = time.time()
    print("[format_from_df] Start")

    # get mode of ensemble predictions 

    load_norm_cols = [f"load_norm_class ({i})" for i in range(5)] 
    loadings = mdf[load_norm_cols].values 
    lmode = stats.mode(loadings, axis=1)[0] 
    mdf['load_norm_class'] = lmode

    mmat = make_mat(mdf, "load_norm_class")
    mmat = mmat.loc[bemat.index][bemat.columns]

    # get mapping from class to actual loading per unit cell 

    _, bins_dict, _ = get_load_norm_bins() 
    lmat = np.vectorize(bins_dict.get)(mmat)
    lmat = pd.DataFrame(lmat, index=mmat.index, columns=mmat.columns)
    print("[format_from_df] lmat shape:", lmat.shape)

    # get number of Si atoms per unit cell for each framework 

    zpriors = pd.read_pickle(zprior_file) 
    num_si = zpriors.loc[mmat.columns].num_atoms / 3 

    # get binding energy in kJ/ mol OSDA

    beosda_mat = bemat / lmat * num_si
        
    # save files 

    os.makedirs(op_dir_loading, exist_ok=True)
    os.makedirs(op_dir_beosda, exist_ok=True)

    lmat.to_csv(os.path.join(op_dir_loading, f"{basename}_loading_norm.csv"))
    beosda_mat.to_csv(os.path.join(op_dir_beosda, f"{basename}_beosda.csv"))

    # NOTE: no need to separate smiles since they are already separated

    print("[format_from_df] Time after save:", (time.time() - single_start) / 60, "mins")
    return mmat, beosda_mat


def format_from_file_pair(
    bef, mf, op_dir_loading, op_dir_beosda, zprior_file, to_save=True, combine=True
):
    """Format ensemble predictions from binary and energy models. See format_from_df method for more details."""

    print("[format_from_file_pair] Formatting single file pair")

    if not check_same_file(bef, mf):
        print("check same file - BE file:", bef)
        print("check same file - loading file:", mf)
        raise Exception("files do not match")

    print("\n[format_from_file_pair] Formatting")
    print(bef)
    print(mf)

    # read in data

    basename = os.path.basename(mf).split(".csv")[0]
    bemat = pd.read_csv(bef, index_col=0)
    mdf = pd.read_csv(mf, index_col=0)

    mmat, beosda_mat = format_from_df(
        bemat, mdf, op_dir_loading, op_dir_beosda, zprior_file, to_save, basename, combine
    )

    return mmat, beosda_mat


def format_predictions(
    befiles, mfiles, op_dir_loading, op_dir_beosda, zprior_file, to_save=True, combine=True
):
    """
    Format ensemble predictions from loading multiclassification models and formatted binding energies (BE) files.
    See format_from_file_pair method for more details.
    """

    # check if files match
    assert len(befiles) == len(
        mfiles
    ), "[format_predictions] number of BE and loading files do not match"

    # sort files by name
    if len(befiles) > 1:
        befiles = sorted(befiles, key=sort_by)
        mfiles = sorted(mfiles, key=sort_by)

    for bef, mf in zip(befiles, mfiles):
        # try:
        format_from_file_pair(
            bef, mf, op_dir_loading, op_dir_beosda, zprior_file, to_save, combine
        )
        # except Exception as e:
        #     breakpoint()


if __name__ == "__main__":
    """
    This script is for formatting energy and binary predictions into 3 matrices:
        1. S-Z matrix with binary means and stds (Z_mean, Z_std)
        2. S-Z matrix with energy means and stds (Z_mean, Z_std)
        3. S-Z matrix with binding energies from multiplying binary class with energy (Z_1, Z_2, .., Z_5, Z_mean, Z_std)
    """

    main_start = time.time()

    parser = argparse.ArgumentParser(description="Format predictions")
    parser.add_argument(
        "--preds_dir_be", type=str, help="Directory of binding energy (BE) predictions"
    )
    parser.add_argument(
        "--preds_dir_m", type=str, help="Directory of loading predictions"
    )
    parser.add_argument(
        "--op_dir_loading", type=str, help="Output directory for loading_norm matrices"
    )
    parser.add_argument(
        "--op_dir_beosda",
        type=str,
        help="Output directory for kJ/ mol OSDA binding energy matrices",
    )
    parser.add_argument(
        "--zprior_file",
        type=str, 
        help="Filepath to the zeolite prior file. Used to get number of Si atoms",
    )
    args = parser.parse_args()

    # get list of files

    befiles = glob.glob(args.preds_dir_be + "/*_mean_*.csv")
    print(len(befiles), "be files")
    mfiles = glob.glob(args.preds_dir_m + "/*_preds.csv")
    print(len(mfiles), "preds_dir_m files")
    assert len(befiles) == len(mfiles), "number of binary and energy files do not match"

    # format

    format_predictions(befiles, mfiles, args.op_dir_loading, args.op_dir_beosda, args.zprior_file)

    print("Time taken:", (time.time() - main_start) / 60, "mins")
    print("Output directories are:")
    print(args.op_dir_loading)
    print(args.op_dir_beosda)
