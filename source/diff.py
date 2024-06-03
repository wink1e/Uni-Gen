import numpy as np
from scipy.spatial import distance_matrix

PYYKKO_RADII = {'H': 0.32, 'He': 0.46, 'Li': 1.33, 'Be': 1.02, 'B': 0.85,
                'C': 0.75, 'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67,
                'Na': 1.55, 'Mg': 1.39, 'Al': 1.26, 'Si': 1.16,
                'P': 1.11,
                'S': 1.03, 'Cl': 0.99, 'Ar': 0.96, 'K': 1.96, 'Ca': 1.71,
                'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22,
                'Mn': 1.19,
                'Fe': 1.16, 'Co': 1.11, 'Ni': 1.10, 'Cu': 1.12, 'Zn': 1.18,
                'Ga': 1.24, 'Ge': 1.21, 'As': 1.21, 'Se': 1.16, 'Br': 1.14,
                'Kr': 1.17, 'Rb': 2.10, 'Sr': 1.85, 'Y': 1.63, 'Zr': 1.54,
                'Nb': 1.47, 'Mo': 1.38, 'Tc': 1.28, 'Ru': 1.25, 'Rh': 1.25,
                'Pd': 1.20, 'Ag': 1.28, 'Cd': 1.36, 'In': 1.42, 'Sn': 1.40,
                'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
                'Cs': 2.32,
                'Ba': 1.96, 'La': 1.80, 'Ce': 1.63, 'Pr': 1.76,
                'Nd': 1.74,
                'Pm': 1.73, 'Sm': 1.72, 'Eu': 1.68, 'Gd': 1.69,
                'Tb': 1.68,
                'Dy': 1.67, 'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64,
                'Yb': 1.70,
                'Lu': 1.62, 'Hf': 1.52, 'Ta': 1.46, 'W': 1.37, 'Re': 1.31,
                'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23, 'Au': 1.24,
                'Hg': 1.33,
                'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51, 'Po': 1.45,
                'At': 1.47,
                'Rn': 1.42, 'Fr': 2.23, 'Ra': 2.01, 'Ac': 1.86,
                'Th': 1.75,
                'Pa': 1.69, 'U': 1.70, 'Np': 1.71, 'Pu': 1.72, 'Am': 1.66,
                'Cm': 1.66, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65, 'Fm': 1.67,
                'Md': 1.73, 'No': 1.76, 'Lr': 1.61}


def get_sprint_vec(coord: np.ndarray, atom_type: np.ndarray, n: int = 6, m: int = 12, power: int = 1) -> np.ndarray:
    r"""
    Generate sprint descriptor from coordination.
    :param coord: The coordination of the molecule, with the shape of (n_atom, 3) or (3*n_atom,).
    :param atom_type: The atom type of the molecule, with the shape of (n_atom,)
    :param n: The exponential number.
    :param m: The exponential number.
    :param power: The power of the aij.
    :return: The sprint coordinates with the shape of (n_atom, )
    """
    if len(coord.shape) == 1:
        coord = coord.reshape((-1, 3))
    n_atom = len(atom_type)
    rij = distance_matrix(coord, coord)
    a = list(map(lambda i: PYYKKO_RADII[i], atom_type))
    a = np.tile(a, (n_atom, 1))
    dij = a + a.transpose()
    rr = rij / dij
    aij = np.divide(1.0 - rr ** n, 1.0 - rr ** m)

    v, w = np.linalg.eigh(aij)
    power = max(power, 1)
    s = 1.0 / np.power(v.max(), power - 1) * np.sqrt(n_atom)
    sprint = np.dot(aij, np.abs(w[:, -1])) * s
    return np.sort(sprint, kind='mergesort')


def get_diff(sprint_1: np.ndarray, sprint_2: np.ndarray) -> float:
    r"""
    Get diff of two sprint vector.
    :return: float, the diff of two molecule.
    """
    assert len(sprint_1) == len(sprint_2), "Two descriptor must be the same!"
    n = len(sprint_1)
    temp = 1 + np.mean(np.abs(sprint_1 - sprint_2))
    return 1 / temp


def get_sample_mean_diff(coords: np.ndarray, atom_type: np.ndarray) -> float:
    r"""
    Get the mean diff of n frame of molecules.
    :param coords: The coordination of the molecules, with the shape of (n_frame, n_atom*3).
    :param atom_type: The atom type of the molecules, with the shape of (n_atom,)
    :return: the mean diff of the n frame of molecules, which is a float.
    """
    sprints = np.empty(0)
    for k in range(len(coords)):
        sprints = np.concatenate([sprints, get_sprint_vec(coords[k], atom_type)])
    sprints = sprints.reshape(-1, len(atom_type))
    diff_list = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            diff = get_diff(sprints[i], sprints[j])
            diff_list.append(diff)
    return sum(diff_list) / len(diff_list)
