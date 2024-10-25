import json
import os
import random
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors3D, rdFreeSASA, GraphDescriptors, MolFromSmiles, AddHs
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem import AllChem as Chem
from sklearn.decomposition import IncrementalPCA
from rdkit.Chem.AllChem import (
    ComputeMolVolume,
    EmbedMultipleConfs,
    UFFGetMoleculeForceField,
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
    GetConformerRMS,
)
from rdkit.Chem.rdmolops import RemoveHs
PERIODICTABLE = Chem.GetPeriodicTable()

from zeobind.src.utils.logger import log_msg

def get_whim(mol):
    return rdMD.CalcWHIM(mol)


def get_morse(mol):
    return rdMD.CalcMORSE(mol)


def get_volume(mol, grid_spacing=0.2, box_margin=2.0):
    return ComputeMolVolume(mol, gridSpacing=grid_spacing, boxMargin=box_margin)


def get_axes(mol):
    atoms = mol.GetAtoms()
    xyz = np.array([atom.GetPos() for atom in atoms])
    pca = IncrementalPCA(2)
    try:
        uv = pca.fit(xyz).transform(xyz)
    except ValueError:
        print("axes fp valueerror for mol", mol.id)

    if xyz.shape[0] == 1:
        print("axes fp only one atom for mol", mol.id)
        return 0.0, 0.0

    length = uv.max(0) - uv.min(0)
    length.sort()

    return length[0], length[1]


def get_box(mol):
    from analysis.shape import box_metric

    return box_metric(mol)


def get_getaway(mol):
    return Chem.rdMolDescriptors.CalcGETAWAY(mol)


def get_num_bonds(mol):
    # Excludes bonds to hydrogen atoms
    return len(Chem.RemoveAllHs(mol).GetBonds())


def get_num_rot_bonds(mol, strict=rdMD.NumRotatableBondsOptions.Strict):
    # Excludes bonds to hydrogen atoms
    return Chem.rdMolDescriptors.CalcNumRotatableBonds(
        Chem.RemoveAllHs(mol), strict=strict
    )


def get_asphericity(mol, use_atomic_masses=True):
    return Descriptors3D.Asphericity(mol, useAtomicMasses=use_atomic_masses)


def get_eccentricity(mol, use_atomic_masses=True):
    return Descriptors3D.Eccentricity(mol, useAtomicMasses=use_atomic_masses)


def get_inertial_shape_factor(mol, use_atomic_masses=True):
    return Descriptors3D.InertialShapeFactor(mol, useAtomicMasses=use_atomic_masses)


def get_spherocity_index(mol):
    return Descriptors3D.SpherocityIndex(mol)


def get_gyration_radius(mol, use_atomic_masses=True):
    return Descriptors3D.RadiusOfGyration(mol, useAtomicMasses=use_atomic_masses)


def get_pmi1(mol, use_atomic_masses=True):
    return Descriptors3D.PMI1(mol, useAtomicMasses=use_atomic_masses)


def get_pmi2(mol, use_atomic_masses=True):
    return Descriptors3D.PMI2(mol, useAtomicMasses=use_atomic_masses)


def get_pmi3(mol, use_atomic_masses=True):
    return Descriptors3D.PMI3(mol, useAtomicMasses=use_atomic_masses)


def get_npr1(mol, use_atomic_masses=True):
    return Descriptors3D.NPR1(mol, useAtomicMasses=use_atomic_masses)


def get_npr2(mol, use_atomic_masses=True):
    return Descriptors3D.NPR2(mol, useAtomicMasses=use_atomic_masses)


def get_free_sasa(mol):
    # https://github.com/rdkit/rdkit/issues/1827
    # default probe is 1.4A, change with rdFreeSASA.SASAOpts
    # radii is a list of atomic radius. VdW radius is used here
    # using `radii = rdFreeSASA.classifyAtoms(mol)` returns a list of zeroes - DO NOT USE
    # check that hydrogen atoms are present
    ptable = Chem.GetPeriodicTable()
    radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    return rdFreeSASA.CalcSASA(mol, radii)


def get_bertz_ct(mol):
    return GraphDescriptors.BertzCT(mol)


def get_mol_weight(mol):
    weights = [PERIODICTABLE.GetAtomicWeight(atom.GetSymbol()) for atom in mol.GetAtoms()]
    return sum(weights)


def get_formal_charge(mol):
    return Chem.GetFormalCharge(mol)


def list_wrapper(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if type(result) in [int, float]:
            return [result]
        return list(result)

    return wrapper


METRIC_FUNCTIONS = {
    "box": get_box,
    "mol_volume": list_wrapper(get_volume),
    "axes": list_wrapper(get_axes),
    "whim": list_wrapper(get_whim),
    "getaway": list_wrapper(get_getaway),
    "num_bonds": list_wrapper(get_num_bonds),
    "num_rot_bonds": list_wrapper(get_num_rot_bonds),
    "asphericity": list_wrapper(get_asphericity),
    "eccentricity": list_wrapper(get_eccentricity),
    "inertial_shape_factor": list_wrapper(get_inertial_shape_factor),
    "spherocity_index": list_wrapper(get_spherocity_index),
    "gyration_radius": list_wrapper(get_gyration_radius),
    "pmi1": list_wrapper(get_pmi1),
    "pmi2": list_wrapper(get_pmi2),
    "pmi3": list_wrapper(get_pmi3),
    "npr1": list_wrapper(get_npr1),
    "npr2": list_wrapper(get_npr2),
    "free_sasa": list_wrapper(get_free_sasa),
    "bertz_ct": list_wrapper(get_bertz_ct),
    "morse": list_wrapper(get_morse),
    "mol_weight": list_wrapper(get_mol_weight),
    "formal_charge": list_wrapper(get_formal_charge),
}


class ConformerGenerator(object):
    """
    Generates conformations of molecules from 2D representation.
    """

    def __init__(self, smiles, forcefield="mmff"):
        """
        Initialises the class
        """
        self.mol = MolFromSmiles(smiles)
        self.full_clusters = []
        self.forcefield = forcefield
        self.conf_energies = []
        self.initial_confs = None
        self.smiles = smiles

    def generate(
        self,
        max_generated_conformers=50,
        prune_thresh=0.01,
        maxattempts_per_conformer=5,
        output=None,
        threads=1,
    ):
        """
        Generates conformers

        Note  the number max_generated_conformers required is related to the
        number of rotatable bonds
        """
        self.mol = AddHs(self.mol, addCoords=True)
        self.initial_confs = EmbedMultipleConfs(
            self.mol,
            numConfs=max_generated_conformers,
            pruneRmsThresh=prune_thresh,
            maxAttempts=maxattempts_per_conformer,
            useRandomCoords=False,
            # Despite what the documentation says -1 is a seed
            # It doesn't mean random generation
            numThreads=threads,
            randomSeed=random.randint(1, 10000000),
        )
        if len(self.initial_confs) == 0:
            log_msg(
                "conformgenerator",
                "Generated " + str(len(self.initial_confs)) + " initial confs\n",
            )
            log_msg(
                "conformgenerator",
                "Trying again with {} attempts and random coords\n".format(
                    max_generated_conformers * 10
                ),
            )

            self.initial_confs = EmbedMultipleConfs(
                self.mol,
                numConfs=max_generated_conformers,
                pruneRmsThresh=prune_thresh,
                useRandomCoords=True,
                maxAttempts=10 * maxattempts_per_conformer,
                # Despite what the documentation says -1 is a seed
                # It doesn't mean random
                # generatrion
                numThreads=threads,
                randomSeed=random.randint(1, 10000000),
            )

        log_msg(
            "conformgenerator",
            "Generated " + str(len(self.initial_confs)) + " initial confs\n",
        )
        return self.initial_confs

    def minimise(self, output=None, minimize=True):
        """
        Minimises conformers using a force field
        """

        if "\\" in self.smiles or "/" in self.smiles:
            log_msg(
                "conformgenerator",
                "Smiles string contains slashes, which specify cis/trans stereochemistry.\n",
            )
            log_msg(
                "conformgenerator",
                "Bypassing force-field minimization to avoid generating incorrect isomer.\n",
            )
            minimize = False

        if self.forcefield != "mmff":
            raise ValueError("Unrecognised force field")
        else:
            props = MMFFGetMoleculeProperties(self.mol)
            for i in range(0, len(self.initial_confs)):
                potential = MMFFGetMoleculeForceField(self.mol, props, confId=i)
                if potential is None:
                    log_msg("conformgenerator", "MMFF not available, using UFF\n")
                    potential = UFFGetMoleculeForceField(self.mol, confId=i)
                    assert potential is not None
                if minimize:
                    log_msg(
                        "conformgenerator", "Minimising conformer number {}\n".format(i)
                    )
                    potential.Minimize()
                mmff_energy = potential.CalcEnergy()
                self.conf_energies.append((i, mmff_energy))

        self.conf_energies = sorted(self.conf_energies, key=lambda tup: tup[1])
        return self.mol

    def cluster(
        self,
        rms_tolerance=0.1,
        max_ranked_conformers=10,
        energy_window=5,
        Report_e_tol=10,
        output=None,
    ):
        """
        Removes duplicates after minimisation
        """
        self.counter = 0
        self.factormax = 3
        self.mol_no_h = RemoveHs(self.mol)
        calcs_performed = 0
        self.full_clusters = []
        confs = self.conf_energies[:]
        ignore = []
        ignored = 0

        for i, pair_1 in enumerate(confs):
            if i == 0:
                index_0, energy_0 = pair_1
            log_msg(
                "conformgenerator",
                "clustering cluster {} of {}\n".format(i, len(self.conf_energies)),
            )
            index_1, energy_1 = pair_1
            if abs(energy_1 - energy_0) > Report_e_tol:
                msg = "Breaking because hit Report Energy Window, E was {} Kcal/mol and minimum was {} \n"
                log_msg("conformgenerator", msg.format(energy_1, energy_0))
                break
            if i in ignore:
                ignored += i
                continue
            self.counter += 1
            if self.counter == self.factormax * max_ranked_conformers:
                log_msg("conformgenerator", "Breaking because hit MaxNConfs \n")
                break
            clustered = [[self.mol.GetConformer(id=index_1), energy_1, 0.00]]
            ignore.append(i)
            for j, pair_2 in enumerate(confs):
                if j > 1:
                    index_2, energy_2 = pair_2
                    if j in ignore:
                        ignored += 1
                        continue
                    if abs(energy_1 - energy_2) > energy_window:
                        break
                    if abs(energy_1 - energy_2) <= 1e-3:
                        clustered.append(
                            [self.mol.GetConformer(id=index_2), energy_2, 0.00]
                        )
                        ignore.append(j)
                        rms = GetConformerRMS(self.mol_no_h, index_1, index_2)
                        calcs_performed += 1
                        if rms <= rms_tolerance:
                            clustered.append(
                                [self.mol.GetConformer(id=index_2), energy_2, rms]
                            )
                            ignore.append(j)
            self.full_clusters.append(clustered)
        log_msg("conformgenerator", "{} ignore passes made\n".format(ignored))
        log_msg(
            "conformgenerator",
            "{} overlays needed out of a possible {}\n".format(
                calcs_performed, len(self.conf_energies) ** 2
            ),
        )
        ranked_clusters = []
        for i, cluster in enumerate(self.full_clusters):
            if i < self.factormax * max_ranked_conformers:
                ranked_clusters.append(cluster[0])
        return ranked_clusters


def compute_osda_features(
    smiles,
    num_conformers=20,
):
    with open("zeobind/src/configs/osda_v1_phys.json", "r") as f:
        config = json.load(f)

    features = pd.DataFrame(index=smiles, columns=config.keys())

    for smi in smiles:
        generator = ConformerGenerator(smi)
        generator.generate()
        generator.minimise()
        clustered_confs = generator.cluster()
        clustered_confs = clustered_confs[:num_conformers]

        for fea in features.columns:
            conf_feas = []
            for i, conf in enumerate(clustered_confs):
                conf_feas.append(METRIC_FUNCTIONS[fea](conf[0].GetOwningMol()))
            conf_feas = np.array(conf_feas)
            conf_feas = np.mean(conf_feas, axis=0)
            features.loc[smi, fea] = conf_feas

    return features
