import glob
import os
import numpy as np
import pandas as pd 

from zeobind.src.utils.utils import sort_by, filter_smis_by_charge
import time
import argparse

# ## Workflow
# 1. single E * single B, the files should match up
# 2. one matrix of (single E * single B)
#     1. All models + mean + std
# 3. one matrix of binary means and stds
# 4. one matrix of energy means and stds


def check_same_file(file1, file2):
    file1 = os.path.basename(file1)
    file2 = os.path.basename(file2)
    return file1 == file2


def make_mat(df, values, index="SMILES", columns="Zeolite"):
    return df.pivot(index=index, columns=columns, values=values)

def format_from_df(bdf, edf, op_dir_b, op_dir_e, op_dir_be, to_save, basename, combine=True):
    """Format ensemble predictions from binary and energy models.

    Arguments:
        bdf -- DataFrame containing binary classifier outputs
        edf -- DataFrame containing energy regressor outputs
        op_dir_b -- where to save binary matrices
        op_dir_e -- where to save energy matrices
        op_dir_be -- where to save binding energy matrices
        to_save -- if True, saves matrices 
        basename -- basename of file to save matrices as

    Keyword Arguments:
        combine -- if True, returns 6 DataFrames. Else, returns 6 dictionaries containing (smile type, DataFrame) key-value pairs. (default: {True})

    Returns:
        See combine argument
    """
    single_start = time.time()

    new_index = list(zip(edf["SMILES"], edf["Zeolite"]))
    bdf = bdf.set_index(["SMILES", "Zeolite"]).reindex(new_index).reset_index()

    # transform binary classifier outputs

    for i in range(5):
        bdf[f"exp(b) ({i})"] = np.exp(bdf[f"b ({i})"])
        bdf[f"exp(nb) ({i})"] = np.exp(bdf[f"nb ({i})"])
        bdf[f"denom ({i})"] = bdf[f"exp(b) ({i})"] + bdf[f"exp(nb) ({i})"]
        bdf[f"softmax b ({i})"] = bdf[f"exp(b) ({i})"] / bdf[f"denom ({i})"]
        bdf[f"binding ({i})"] = bdf[f"softmax b ({i})"].apply(
            lambda x: 1 if x > 0.5 else 0
        )

    binding_cols = [f"binding ({i})" for i in range(5)]
    softmax_cols = [f"softmax b ({i})" for i in range(5)]
    bdf["softmax b (mean)"] = bdf[softmax_cols].mean(axis=1)
    bdf["softmax b (std)"] = bdf[softmax_cols].std(axis=1)

    # transform energy regressor outputs

    energy_cols = [f"Binding (SiO2) ({i})" for i in range(5)]
    edf["Binding (SiO2) (mean)"] = edf[energy_cols].mean(axis=1)
    edf["Binding (SiO2) (std)"] = edf[energy_cols].std(axis=1)

    # 1. S-Z matrix with binary means and stds (Z_mean, Z_std)

    bmat_mean = make_mat(bdf, "softmax b (mean)")

    bmat_std = make_mat(bdf, "softmax b (std)")

    # 2. S-Z matrix with energy means and stds (Z_mean, Z_std)

    emat_mean = make_mat(edf, "Binding (SiO2) (mean)")

    emat_std = make_mat(edf, "Binding (SiO2) (std)")

    # 3. S-Z matrix with binding energies from multiplying binary class with energy (Z_1, Z_2, .., Z_5, Z_mean, Z_std)

    be_cols = [f"BE ({i})" for i in range(5)]
    bedf = edf[["SMILES", "Zeolite"]]
    bedf.loc[:, be_cols] = np.multiply(
        bdf[binding_cols].values, edf[energy_cols].values
    )
    bedf["BE (mean)"] = bedf[be_cols].mean(axis=1)
    bedf["BE (std)"] = bedf[be_cols].std(axis=1)

    bemat_mean = make_mat(bedf, "BE (mean)")

    bemat_std = make_mat(bedf, "BE (std)")

    # separate smis

    smis_neu = filter_smis_by_charge(edf["SMILES"], 0, 0)
    smis_monoq = filter_smis_by_charge(edf["SMILES"], 1, 1)
    smis_diq = filter_smis_by_charge(edf["SMILES"], 2, 2)
    smis_others = filter_smis_by_charge(edf["SMILES"], 3, 100) + filter_smis_by_charge(
        edf["SMILES"], -100, -1
    )
    smis_labels = ["neu", "monoq", "diq", "others"]

    # format smis

    smis_neu = sorted(set(smis_neu))
    smis_monoq = sorted(set(smis_monoq))
    smis_diq = sorted(set(smis_diq))

    # create op dirs

    if to_save:

        os.makedirs(op_dir_b, exist_ok=True)
        os.makedirs(op_dir_e, exist_ok=True)
        os.makedirs(op_dir_be, exist_ok=True)

    bmat_mean_s_dict = {}
    bmat_std_s_dict = {}
    emat_mean_s_dict = {}
    emat_std_s_dict = {}
    bemat_mean_s_dict = {}
    bemat_std_s_dict = {}

    for smis, label in zip([smis_neu, smis_monoq, smis_diq, smis_others], smis_labels):
        
        print(f"[format] Locating data for {label} molecules")

        # filter by smis
        bmat_mean_s = bmat_mean.loc[smis]
        bmat_std_s = bmat_std.loc[smis]
        emat_mean_s = emat_mean.loc[smis]
        emat_std_s = emat_std.loc[smis]
        bemat_mean_s = bemat_mean.loc[smis]
        bemat_std_s = bemat_std.loc[smis]

        # print shapes
        print("bmat:", bmat_mean_s.shape, bmat_std_s.shape)
        print("emat:", emat_mean_s.shape, emat_std_s.shape)
        print("bemat:", bemat_mean_s.shape, bemat_std_s.shape)

        print("[format] Time before save:", (time.time() - single_start) / 60, "mins")

        # save
        if to_save:
            bmat_mean_s.to_csv(
                os.path.join(op_dir_b, f"{basename}_bmat_mean_{label}.csv")
            )
            bmat_std_s.to_csv(
                os.path.join(op_dir_b, f"{basename}_bmat_std_{label}.csv")
            )
            emat_mean_s.to_csv(
                os.path.join(op_dir_e, f"{basename}_emat_mean_{label}.csv")
            )
            emat_std_s.to_csv(
                os.path.join(op_dir_e, f"{basename}_emat_std_{label}.csv")
            )
            bemat_mean_s.to_csv(
                os.path.join(op_dir_be, f"{basename}_bemat_mean_{label}.csv")
            )
            bemat_std_s.to_csv(
                os.path.join(op_dir_be, f"{basename}_bemat_std_{label}.csv")
            )

        bmat_mean_s_dict[label] = bmat_mean_s
        bmat_std_s_dict[label] = bmat_std_s
        emat_mean_s_dict[label] = emat_mean_s
        emat_std_s_dict[label] = emat_std_s
        bemat_mean_s_dict[label] = bemat_mean_s
        bemat_std_s_dict[label] = bemat_std_s

    if combine:
        bmat_mean_s_dict = pd.concat(bmat_mean_s_dict.values(), axis=0)
        bmat_std_s_dict = pd.concat(bmat_std_s_dict.values(), axis=0)
        emat_mean_s_dict = pd.concat(emat_mean_s_dict.values(), axis=0)
        emat_std_s_dict = pd.concat(emat_std_s_dict.values(), axis=0)
        bemat_mean_s_dict = pd.concat(bemat_mean_s_dict.values(), axis=0)
        bemat_std_s_dict = pd.concat(bemat_std_s_dict.values(), axis=0)


    print("[format] Time after save:", (time.time() - single_start) / 60, "mins")
    return bmat_mean_s_dict, bmat_std_s_dict, emat_mean_s_dict, emat_std_s_dict, bemat_mean_s_dict, bemat_std_s_dict


def format_from_file_pair(bf, ef, op_dir_b, op_dir_e, op_dir_be, to_save=True, combine=True):
    """Format ensemble predictions from binary and energy models. See format_from_df method for more details."""

    print("[format] Formatting single file pair")

    if not check_same_file(bf, ef):
        print("check same file - binary file:", bf)
        print("check same file - energy file:", ef)
        raise Exception("files do not match")

    print("\n[format] Formatting")
    print(bf)
    print(ef)

    # read in data

    bdf = pd.read_csv(bf, index_col=0)
    edf = pd.read_csv(ef, index_col=0)
    basename = os.path.basename(bf).split(".csv")[0]

    bmat_mean, bmat_std, emat_mean, emat_std, bemat_mean, bemat_std = format_from_df(bdf, edf, op_dir_b, op_dir_e, op_dir_be, to_save, basename, combine)

    return bmat_mean, bmat_std, emat_mean, emat_std, bemat_mean, bemat_std



def format_predictions(bfiles, efiles, op_dir_b, op_dir_e, op_dir_be, to_save=True, combine=True):
    """Format ensemble predictions from binary and energy models, for a list of prediction files. See format_from_file_pair method for more details."""

    # check if files match
    assert len(bfiles) == len(efiles), "[format] number of binary and energy files do not match"
    
    # sort files by name
    if len(bfiles) > 1:
        bfiles = sorted(bfiles, key=sort_by)
        efiles = sorted(efiles, key=sort_by)

    for bf, ef in zip(bfiles, efiles):
        try:
            format_from_file_pair(bf, ef, op_dir_b, op_dir_e, op_dir_be, to_save, combine)
        except:
            breakpoint()

        


if __name__ == "__main__":
    """
    This script is for formatting energy and binary predictions into 3 matrices:
        1. S-Z matrix with binary means and stds (Z_mean, Z_std)
        2. S-Z matrix with energy means and stds (Z_mean, Z_std)
        3. S-Z matrix with binding energies from multiplying binary class with energy (Z_1, Z_2, .., Z_5, Z_mean, Z_std)
    """

    main_start = time.time()

    parser = argparse.ArgumentParser(description="Format predictions")
    parser.add_argument("--preds_dir_b", type=str, help="Directory of binary predictions")
    parser.add_argument("--preds_dir_e", type=str, help="Directory of energy predictions")
    parser.add_argument("--op_dir_b", type=str, help="Output directory for binary matrices")
    parser.add_argument("--op_dir_e", type=str, help="Output directory for energy matrices")
    parser.add_argument("--op_dir_be", type=str, help="Output directory for binding energy matrices")
    args = parser.parse_args()

    # get list of files

    bfiles = glob.glob(args.preds_dir_b + "/*.csv")
    print(len(bfiles), "binary files")
    efiles = glob.glob(args.preds_dir_e + "/*.csv")
    print(len(efiles), "energy files")
    assert len(bfiles) == len(efiles), "number of binary and energy files do not match"

    # format

    format_predictions(bfiles, efiles, args.op_dir_b, args.op_dir_e, args.op_dir_be)

    print("Time taken:", (time.time() - main_start) / 60, "mins")
    print("Output directories are:")
    print(args.op_dir_b)
    print(args.op_dir_e)
    print(args.op_dir_be)