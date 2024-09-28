from utils.utils import get_load_norm_bins

MCLASS_TASK = "loading_classification"
BINARY_TASK = "binary_classification"
ENERGY_TASK = "energy_regression"
COL_DICT = {
    ENERGY_TASK: "Binding (SiO2)",
    MCLASS_TASK: "load_norm", # TODO: load_norm_class?
    BINARY_TASK: "binding",
}


class EnergyTask:
    def __init__(self):
        # self.task = "energy_regression"
        self.task = ENERGY_TASK
        self.label = COL_DICT[ENERGY_TASK]
        self.class_op_size = 1


class FitTask:
    def __init__(self):
        # self.task = "binary_classification"
        self.task = BINARY_TASK
        self.label = COL_DICT[BINARY_TASK]
        self.class_op_size = 1  # TODO check


class LoadingTask:
    def __init__(self):
        # self.task = "loading_classification"
        self.task = MCLASS_TASK
        bins, bins_dict, _ = get_load_norm_bins()
        self.label = [f"{COL_DICT[MCLASS_TASK]}_{i}" for i in bins_dict.keys()]
        self.class_op_size = len(bins_dict.keys())

PREDICTION_TASK_DICT = dict(energy=EnergyTask, binding=FitTask, loading=LoadingTask)
