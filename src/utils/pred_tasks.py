from utils.utils import get_load_norm_bins

MCLASS_TASK = "loading_classification"
BINARY_TASK = "binary_classification"
ENERGY_TASK = "energy_regression"
COL_DICT = {
    ENERGY_TASK: "Binding (SiO2)",
    MCLASS_TASK: "load_norm", 
    BINARY_TASK: "b",
}


class EnergyTask:
    def __init__(self):
        self.task = ENERGY_TASK
        self.label = COL_DICT[ENERGY_TASK]
        self.output_label = [COL_DICT[ENERGY_TASK] + " pred"]
        self.class_op_size = 1


class FitTask:
    def __init__(self):
        self.task = BINARY_TASK
        self.label = COL_DICT[BINARY_TASK]
        self.output_label = ["nb", "b"]
        self.class_op_size = 2


class LoadingTask:
    def __init__(self):
        self.task = MCLASS_TASK
        bins, bins_dict, _ = get_load_norm_bins()
        self.label = [f"{COL_DICT[MCLASS_TASK]}_{i}" for i in bins_dict.keys()]
        self.output_label = [f"{COL_DICT[MCLASS_TASK]}_{i} pred" for i in bins_dict.keys()]
        self.class_op_size = len(bins_dict.keys())


class MultitaskTask: 
    def __init__(self):
        self.task = "multitask"
        bins, bins_dict, _ = get_load_norm_bins()
        self.label = [COL_DICT[BINARY_TASK], COL_DICT[ENERGY_TASK]] + [f"{COL_DICT[MCLASS_TASK]}_{i}" for i in bins_dict.keys()]
        self.output_label = ["nb", "b"] + [COL_DICT[ENERGY_TASK] + " pred"] + [f"{COL_DICT[MCLASS_TASK]}_{i} pred" for i in bins_dict.keys()]
        self.binary_class_op_size = 2
        self.energy_class_op_size = 1 
        self.load_class_op_size = len(bins_dict.keys())


PREDICTION_TASK_DICT = dict(energy_regression=EnergyTask, binary_classification=FitTask, loading_classification=LoadingTask, multitask=MultitaskTask)
