"""Script with functions for screening hypothetical molecules as OSDA candidates for targeted frameworks."""

import argparse
import glob
import multiprocessing
import os
import time
import pandas as pd
from zeobind.src.utils.utils import get_competition, log_msg

class OSDAScreener:
    def __init__(
        self,
        fw,
        be_mean_files,
        be_std_files,
        # load_mean_files,
        oprior_files,
        filters,
        op_dir,
        parallel=False,
        num_threads=None,
    ):
        """
        Handles input files and output directories, as well as filtering functions for screening OSDA candidates.

        Inputs:
        - be_mean_files: list of filepaths to pd.DataFrames with SMILES in index and Zeolites in columns, containing ensemble mean for binding energy.
        - be_std_files: list of filepaths to pd.DataFrames with SMILEs in index and Zeolites in columns, containing ensemble standard deviation for binding energy.
        # - load_mean_files: list of filepaths to pd.DataFrames with SMILES in index and Zeolites in columns, containing loading/ unit cell.
        - oprior_files: list of filepaths to pd.DataFrames with SMILES in index, containing OSDA features.
        - filters: dict with (filter name, value) pairs.
        - op_dir: Output directory for the filtered OSDA candidates.
        - parallelise: If True, uses multiprocessing to parallelise the screening process.

        """
        self.fw = fw
        self.be_mean_files = be_mean_files
        self.be_std_files = be_std_files
        # self.load_mean_files = load_mean_files
        self.oprior_files = oprior_files
        self.op_dir = op_dir
        self.parallel = parallel
        self.num_threads = num_threads
        self.filters = filters

        self.group_files()
        self.make_op_subdirs()

    def group_files(self):
        """
        Groups the files. Assumes each of the lists are ordered the same way.
        """
        self.files = [
            (em, es, o)
            for em, es, o in zip(
                self.be_mean_files, self.be_std_files, self.oprior_files
            )
        ]
        # self.files = [(em, es, lm, o) for em, es, lm, o in zip(self.be_mean_files, self.be_std_files, self.load_mean_files, self.oprior_files)]

    def make_op_subdirs(self):
        """Make subdirectories to store output files."""
        self.be_means_op_dir = f"{self.op_dir}/be_means"
        self.be_stds_op_dir = f"{self.op_dir}/be_stds"
        # self.load_means_op_dir = f"{self.op_dir}/load_means"
        self.opriors_op_dir = f"{self.op_dir}/opriors"
        self.ce_means_op_dir = f"{self.op_dir}/ce"

        os.makedirs(self.op_dir, exist_ok=True)
        os.makedirs(self.be_means_op_dir, exist_ok=True)
        os.makedirs(self.be_stds_op_dir, exist_ok=True)
        # os.makedirs(self.load_means_op_dir, exist_ok=True)
        os.makedirs(self.opriors_op_dir, exist_ok=True)
        os.makedirs(self.ce_means_op_dir, exist_ok=True)

    def collate_ops(self):
        """
        Collate individual filtered files into a single file, and saves in the main output directory.
        """

        def collate(filepaths, filetype="csv"):
            data = glob.glob(filepaths)
            if filetype == "csv":
                data = pd.concat([pd.read_csv(f, index_col=0) for f in data], axis=0)
            elif filetype == "pkl":
                data = pd.concat([pd.read_pickle(f) for f in data], axis=0)
            return data

        be_means = collate(f"{self.be_means_op_dir}/*.csv", filetype="csv")
        be_stds = collate(f"{self.be_stds_op_dir}/*.csv", filetype="csv")
        # load_means = collate(f"{self.load_means_op_dir}/*.csv", filetype="csv")
        opriors = collate(f"{self.opriors_op_dir}/*.pkl", filetype="pkl")
        df = pd.concat([be_means[self.fw], be_stds[self.fw], opriors], axis=1)
        df.columns = ['be', 'ce'] + list(opriors.columns)

        df.to_csv(f"{self.op_dir}/{self.fw}_all.csv")

        # be_means.to_csv(f"{self.op_dir}/be_means_all.csv")
        # be_stds.to_csv(f"{self.op_dir}/be_stds_all.csv")
        # load_means.to_csv(f"{self.op_dir}/load_means_all.csv")
        # opriors.to_pickle(f"{self.op_dir}/opriors_all.pkl")

        log_msg("collate_ops", f"{be_means.shape[0]} OSDA candidates saved in {self.op_dir}")

    def filter(self):
        if self.parallel:
            log_msg("filter", "Parallelising screening process...")
            with multiprocessing.Pool(self.num_threads) as pool:
                pool.map(self.filter_single, self.files)
        else:
            log_msg("filter", "Sequential screening...")
            for files in self.files:
                self.filter_single(files)

    def filter_single(self, files):
        """
        Filter OSDA candidates based on the specified filters.

        Input:
        - files: Tuple of (BE means, BE stds, molecule features).
        """
        log_msg("filter_single", f"Filtering {files[0]}...")
        bem_file, bes_file, opr_file = files
        bem = pd.read_csv(bem_file, index_col=0)
        bes = pd.read_csv(bes_file, index_col=0)
        opr = pd.read_pickle(opr_file)

        # Filters based on docked pose emtrics

        bem = bem[bem[self.fw] <= self.filters["be_cutoff"]]
        if bem.empty: 
            return None
        bec = get_competition(bem)
        bec = bec[bec[self.fw] <= self.filters["ce_cutoff"]]
        # loadm = loadm[loadm[self.fw].between(self.load_lower, self.load_upper)]
        
        # Filters based on molecule characteristics
        if self.filters.get("remove_aromatic_n", False) is True:
            opr = opr[opr.has_aromatic_n == 0]
        if self.filters.get("remove_3mr", False) is True:
            opr = opr[opr.has_3mr == 0]
        if self.filters.get("remove_4mr", False) is True:
            opr = opr[opr.has_4mr == 0]
        if self.filters.get("remove_double_bond", False) is True:
            opr = opr[opr.has_double_bond == 0]
        if self.filters.get("remove_neighboring_n", False) is True:
            opr = opr[opr.has_neighboring_n == 0]
        if self.filters.get("remove_stereocenters", False) is True:
            opr = opr[opr.has_stereocenter == 0]

        if self.filters["num_c_between_n_min"] > 0:
            opr = opr[opr.num_c_between_n >= self.filters["num_c_between_n_min"]]
        else:
            opr = opr[opr.num_c_between_n == -1]
        opr = opr[opr.mol_volume.between(self.filters["vol_lower"], self.filters["vol_upper"])]
        opr = opr[opr.num_rot_bonds.between(self.filters["rot_lower"], self.filters["rot_upper"])]
        opr = opr[opr.sa_score <= self.filters["sa_cutoff"]]
        opr = opr[opr.cost_mol <= self.filters["px_cutoff"]]
        opr = opr[opr.max_sim_to_lit <= self.filters["sim_cutoff"]]

        # Find intersection of all filtered DataFrames

        smiles = (
            set(bem.index) & set(bes.index) & set(opr.index) & set(bec.index)
        )  # & set(loadm.index)
        if len(smiles) > 0:
            smiles = list(smiles)
            bem = bem.loc[smiles]
            bes = bes.loc[smiles]
            opr = opr.loc[smiles]
            bec = bec.loc[smiles]
            # loadm = loadm.loc[smiles]
        else:
            log_msg("filter_single", f"No OSDA candidates saved from {bem_file}")
            return None

        # Save filtered DataFrames
        bem.to_csv(f"{self.be_means_op_dir}/{os.path.basename(bem_file)}")
        bes.to_csv(f"{self.be_stds_op_dir}/{os.path.basename(bes_file)}")
        opr.to_pickle(f"{self.opriors_op_dir}/{os.path.basename(opr_file)}")
        bec.to_csv(f"{self.ce_means_op_dir}/{os.path.basename(bem_file.replace('_bemat_', '_cemat_'))}")
        # loadm.to_csv(f"{self.load_means_op_dir}/{os.path.basename(loadm)}")
        log_msg("filter_single", f"{len(smiles)} OSDA candidates saved from {bem_file}")


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Screen OSDA candidates for a given framework.")

    # set up 
    parser.add_argument("--parallel", action="store_true", help="Parallelise screening.")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use.", default=24)
    parser.add_argument("--preds_dir", type=str, help="Directory containing formatted predictions.", required=True)
    parser.add_argument("--opriors_dir", type=str, help="Directory containing OSDA features.", required=True)
    parser.add_argument("--ofile_root", type=str, help="Root of output file names.", default="osda_priors")
    parser.add_argument("--zfile_root", type=str, help="Root of zeolite file names.", default="zeolite_priors")
    parser.add_argument("--output", type=str, help="Output directory for filtered OSDA candidates.", required=True)

    # target
    parser.add_argument("--charge", type=str, help="Desired charge of molecules.", choices=["monoq", "diq"], required=True)
    parser.add_argument("--fw", type=str, help="Framework to screen for.", required=True)

    # filters 
    parser.add_argument("--be_cutoff", type=float, help="Binding energy cutoff.", default=0.0)
    parser.add_argument("--ce_cutoff", type=float, help="Competitive energy cutoff.", default=0.0)
    parser.add_argument("--vol_lower", type=float, help="Lower volume limit.", default=50.0)
    parser.add_argument("--vol_upper", type=float, help="Upper volume limit.", default=1000.0)
    parser.add_argument("--rot_lower", type=int, help="Lower rotatable bond limit.", default=0)
    parser.add_argument("--rot_upper", type=int, help="Upper rotatable bond limit.", default=20)
    parser.add_argument("--sim_cutoff", type=float, help="Similarity cutoff.", default=0.8)
    parser.add_argument("--sa_cutoff", type=float, help="Synthetic accessibility score cutoff.", default=5.0)
    parser.add_argument("--px_cutoff", type=float, help="Price cutoff in $/mol.", default=50000.0)
    parser.add_argument("--remove_aromatic_n", action="store_true", help="Remove molecules with aromatic nitrogens.")
    parser.add_argument("--remove_3mr", action="store_true", help="Remove molecules with 3-membered rings.")
    parser.add_argument("--remove_4mr", action="store_true", help="Remove molecules with 4-membered rings.")
    parser.add_argument("--remove_double_bond", action="store_true", help="Remove molecules with double bonds.")
    parser.add_argument("--remove_neighboring_n", action="store_true", help="Remove molecules with neighboring nitrogens.")
    parser.add_argument("--remove_stereocenters", action="store_true", help="Remove molecules with stereocenters.")
    parser.add_argument("--num_c_between_n_min", type=int, help="Minimum number of carbons between nitrogens.", default=-1)

    args = parser.parse_args()

    be_mean_files = [
        f"{args.preds_dir}/formatted_be/{args.ofile_root}_{i}_{args.zfile_root}_preds_bemat_mean_{args.charge}.csv"
        for i in range(args.nfiles)
    ]
    be_std_files = [
        f"{args.preds_dir}/formatted_be/{args.ofile_root}_{i}_{args.zfile_root}_preds_bemat_std_{args.charge}.csv"
        for i in range(args.nfiles)
    ]
    oprior_files = [f"{args.opriors_dir}/osda_priors_{i}.pkl" for i in range(args.nfiles)]

    screener = OSDAScreener(
        args.fw,
        be_mean_files,
        be_std_files,
        oprior_files,
        args.__dict__, # filters
        args.output,
        parallel=args.parallel,
        num_threads=args.num_threads,
    )

    screener.filter()
    screener.collate_ops()

    end = time.time()
    log_msg("main", f"Screening completed in {end - start} seconds.")