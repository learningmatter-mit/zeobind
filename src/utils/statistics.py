import argparse
import os
from tqdm import tqdm
import pandas as pd
import glob

from zeobind.src.utils.utils import sort_by


def get_statistics(preds, op_dir):
    """
    - preds: list of tuples of (be_preds, load_preds, beosda_preds)
        - be_preds: file containing BE predictions
        - load_preds: file containing loading predictions
        - beosda_preds: file containing BE (kJ/ mol OSDA) predictions
    """
    for f in tqdm(preds):
        be_preds, load_preds, beosda_preds = f

        # analyze BE to get best and 2nd framework for each molecule
        bemat = pd.read_csv(be_preds, index_col=0)
        ce = bemat.apply(lambda row: row.nsmallest(2).max(), axis=1)
        second_fws = bemat.apply(lambda row: row.nsmallest(2).idxmax(), axis=1)

        best_fws = bemat.idxmin(1)
        pairs = list(zip(bemat.index, best_fws.values))
        ee = [bemat.loc[p[0], p[1]] for p in pairs]
        ee = pd.DataFrame(ee, index=best_fws.index, columns=["be"])

        df = pd.concat([best_fws, ee, second_fws], axis=1)

        df["ce"] = df["be"].values - ce.values

        df.columns = ["best_fw", "best_be", "second_fw", "ce"]

        df = df.reset_index()
        best_pairs = df[["best_fw", "SMILES"]].values.tolist()
        second_pairs = df[["second_fw", "SMILES"]].values.tolist()

        # get loading of selected pairs

        load_mat = pd.read_csv(load_preds, index_col=0)
        load_df = load_mat.unstack()
        df["best_load"] = load_df.loc[best_pairs].values
        df["second_load"] = load_df.loc[second_pairs].values

        # get beosda of selected pairs

        beosda_mat = pd.read_csv(beosda_preds, index_col=0)
        beosda_df = beosda_mat.unstack()
        df["best_beosda"] = beosda_df.loc[best_pairs].values
        df["second_beosda"] = beosda_df.loc[second_pairs].values

        df.to_csv(f"{op_dir}/{os.path.basename(be_preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of predictions")
    parser.add_argument("--pred_dir", type=str, help="Path to predictions directory")
    parser.add_argument("--op_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "--charge", help="Charge of molecules", choices=["diq", "monoq"]
    )
    args = parser.parse_args()

    be_preds = glob.glob(f"{args.pred_dir}/formatted_be/*mean_{args.charge}.csv")
    load_preds = glob.glob(f"{args.pred_dir}/formatted_mclass/*_preds_loading_norm.csv")
    beosda_preds = glob.glob(f"{args.pred_dir}/formatted_beosda/*_preds_beosda.csv")

    be_preds = sorted(be_preds, key=sort_by)
    load_preds = sorted(load_preds, key=sort_by)
    beosda_preds = sorted(beosda_preds, key=sort_by)
    preds = zip(be_preds, load_preds, beosda_preds)
    get_statistics(preds, args.op_dir)
