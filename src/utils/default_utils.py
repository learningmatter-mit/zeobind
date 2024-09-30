import os 

home = os.path.expanduser("~")
MPL_FONTS_DIR = os.path.join(home, "bin/fonts")

RESULTS_FILE = "zeobind/data/results.csv"
LOAD_NORM_BINS_FILE = "zeobind/src/utils/load_norm_bins.yml"
DEFAULT_ZEOLITE_PRIOR_FILE = "zeobind/data/datasets/training_data/zeolite_priors_0.pkl"