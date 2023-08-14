from typing import Optional
import numpy as np
from pymatgen.core.operations import SymmOp

ATOM_ID_MASS = {'H': (1, 1.00794), 'He': (2, 4.002602), 'Li': (3, 6.941), 'Be': (4, 9.012182), 'B': (5, 10.811),
                'C': (6, 12.0107), 'N': (7, 14.0067), 'O': (8, 15.9994), 'F': (9, 18.9984032), 'Ne': (10, 20.1797),
                'Na': (11, 22.98976928), 'Mg': (12, 24.305), 'Al': (13, 26.9815386), 'Si': (14, 28.0855),
                'P': (15, 30.973762),
                'S': (16, 32.065), 'Cl': (17, 35.453), 'Ar': (18, 39.948), 'K': (19, 39.0983), 'Ca': (20, 40.078),
                'Sc': (21, 44.955912), 'Ti': (22, 47.867), 'V': (23, 50.9415), 'Cr': (24, 51.9961),
                'Mn': (25, 54.938045),
                'Fe': (26, 55.845), 'Co': (27, 58.933195), 'Ni': (28, 58.6934), 'Cu': (29, 63.546), 'Zn': (30, 65.38),
                'Ga': (31, 69.723), 'Ge': (32, 72.64), 'As': (33, 74.9216), 'Se': (34, 78.96), 'Br': (35, 79.904),
                'Kr': (36, 83.798), 'Rb': (37, 85.4678), 'Sr': (38, 87.62), 'Y': (39, 88.90585), 'Zr': (40, 91.224),
                'Nb': (41, 92.90638), 'Mo': (42, 95.96), 'Tc': (43, 97.9072), 'Ru': (44, 101.07), 'Rh': (45, 102.9055),
                'Pd': (46, 106.42), 'Ag': (47, 107.8682), 'Cd': (48, 112.411), 'In': (49, 114.818), 'Sn': (50, 118.71),
                'Sb': (51, 121.76), 'Te': (52, 127.60), 'I': (53, 126.90447), 'Xe': (54, 131.293),
                'Cs': (55, 132.9054519),
                'Ba': (56, 137.327), 'La': (57, 138.90547), 'Ce': (58, 140.116), 'Pr': (59, 140.90765),
                'Nd': (60, 144.242),
                'Pm': (61, 144.9127), 'Sm': (62, 150.36), 'Eu': (63, 151.964), 'Gd': (64, 157.25),
                'Tb': (65, 158.92535),
                'Dy': (66, 162.50), 'Ho': (67, 164.93032), 'Er': (68, 167.259), 'Tm': (69, 168.93421),
                'Yb': (70, 173.054),
                'Lu': (71, 174.9668), 'Hf': (72, 178.49), 'Ta': (73, 180.94788), 'W': (74, 183.84), 'Re': (75, 186.207),
                'Os': (76, 190.23), 'Ir': (77, 192.217), 'Pt': (78, 195.084), 'Au': (79, 196.966569),
                'Hg': (80, 200.59),
                'Tl': (81, 204.3833), 'Pb': (82, 207.2), 'Bi': (83, 208.9804), 'Po': (84, 208.9824),
                'At': (85, 209.9871),
                'Rn': (86, 222.0176), 'Fr': (87, 223.0197), 'Ra': (88, 226.0254), 'Ac': (89, 227.0),
                'Th': (90, 232.0377),
                'Pa': (91, 231.03588), 'U': (92, 238.02891), 'Np': (93, 237), 'Pu': (94, 244), 'Am': (95, 243),
                'Cm': (96, 247), 'Bk': (97, 247), 'Cf': (98, 251), 'Es': (99, 252), 'Fm': (100, 257),
                'Md': (101, 258), 'No': (102, 259), 'Lr': (103, 266)}


class GMolecule:
    r"""
    Creating an GMolecule instance, which could be easily perturbed to a random new form. :param coord: The cartesian
    coordination of the molecules. Have the size of (n_frame, 3*n_atom). :param atom_type: The atom type of the
    molecule.  Have the size of (n_atom). Implicitly indicate that all frame should have the same atom_type :param
    groups: To indicate the group. i.e. for a group of atom index (11,12,13). [[11,12,13],...].
    """

    def __init__(self, coord: Optional[np.array] = None,
                 atom_type: Optional[np.array] = None,
                 groups: Optional[list] = None) -> None:
        self.coord = coord
        self.atom_type = atom_type
        self.atom_num = len(atom_type)
        self.groups = groups
        self.n_frame = len(coord)  # Implicitly, len(np.array) will return the first dimension.
        self.center = None  # With the shape of (n_frame, 3)
        self.coord_number = None  # With the shape of (n_frame, n_atom)
        if coord is not None and atom_type is not None:
            self.init_center()
            self.init_coord_number()

    @classmethod
    def load(cls, md_frame):
        r"""
        Load from CP2K_MD module to create an instance of GMolecule.
        :param md_frame: CP2K_MD instance, which have the coordination of a molecule and atom type.
        By default, the center of mass and coordination number will be initialized by default function.
        :return: GMolecule instance.
        """
        molecule_obj = cls(md_frame.coord, md_frame.atom_type)
        return molecule_obj

    def get_frame(self, idx: int):
        r"""
        Get certain frame of a molecule coordination and atom type.
        :param idx: Index indicating the frame to extract.
        :return: (coordination, atom_type), which have the size of (n_atom, 3) and (n_atom)
        """
        assert idx in range(self.n_frame), "Index out of range."
        out = self.coord[idx]
        out = out.reshape(-1, 3)
        return out, self.atom_type

    def init_center(self):
        r"""
        Initialize the center of mass.
        :return: None
        """
        mass = np.array(list(map(lambda x: ATOM_ID_MASS[x][1], self.atom_type)))
        self.center = np.sum(np.multiply(self.coord.reshape(self.n_frame, self.atom_num, 3),
                                         mass.reshape(self.atom_num, 1)), axis=1) / np.sum(mass)

    def update_coord(self, new_coord, frame_idx):
        r"""
        update the frame_idx frame with the new_coord.
        """
        old_coord = self.coord
        old_coord[frame_idx] = new_coord
        self.coord = old_coord

    def init_coord_number(self, r0=0.5, scaled_factor=500) -> None:
        r"""
        Generate self.coord_number: np.array, shape=(n_frame, n_atom) From self.coord: np.Ndarray, shape=(n_frame,
        n_atom*3). Coordination Number(CN) of ith atom is calculated by: (scaled_factor = s_0 here)
         .. math::
          \{\sum_j s_0 \times 1/[(1+r_{ij}/r_0)^6]\} - s_0

        Scaled_factor and r0 are hyperparameters. For any rij, (rij/r0) should BIGGER THAN 1
        AT ANY TIME, 1 / [1+(rij/r0)**6] increase severely from 0.5 to 1 when (rij/r0) decrease from 1 to 0,
        so choose a small r0=0.5 angstrom as default.
        When scaled_factor=1, CN is too small(~0.001), choose scaled_factor=500 as default
        """
        pair_matrix = np.zeros((self.n_frame, self.atom_num, self.atom_num))
        for i in range(3):
            pair_matrix += (self.coord[:, i::3][:, :, None] - self.coord[:, i::3][:, None, :]) ** 2
        assert np.isclose(pair_matrix[0, 0, 0], 0)
        pair_matrix = np.sqrt(pair_matrix)
        self.coord_number = np.sum((1 / (1 + (pair_matrix / r0) ** 6)), axis=-1) * scaled_factor - scaled_factor

    def random_shift(self, angle_range, distance=1.0, min_coord_num=3, center=None,
                     from_outer=True, const_site=None):
        if angle_range[0] > angle_range[1]:
            raise ValueError("Shift angle range: min > max!")
        if angle_range[0] < 0.0 or angle_range[1] > 90.0:
            raise ValueError("Shift angle range must between (0, 90)")
        gmap = list(range(self.atom_num))
        if self.groups is not None:
            for group in self.groups:
                lowest = min(group)
                for i in group:
                    gmap[i] = lowest
        for j in range(self.n_frame):
            if center is None:
                center = self.center[j]
            coord = self.coord[j].squeeze()
            coord = coord.reshape(-1, 3)
            coord_number = self.coord_number[j]
            shift = np.zeros((self.atom_num, 3))
            steps = np.random.uniform(0.5, 1.0, self.atom_num) * distance
            indices = map(lambda v: np.linalg.norm(v - center), coord)
            distance = []
            for index in indices:
                distance.append(index)
            indices = np.argsort(distance)

            if from_outer:
                np.flipud(indices)

            rnd = np.random.randint(self.atom_num // 4, self.atom_num // 2)
            counter = 0

            for k, i in enumerate(indices):
                site = coord[i]
                if coord_number[i] > min_coord_num:
                    continue
                if const_site is not None:
                    if i in const_site:
                        continue
                if gmap[i] < i:
                    shift[i] = shift[gmap[i]]
                    continue
                vdiff = site - center
                vlen = np.linalg.norm(vdiff)
                if vlen < 0.01:
                    continue
                v_sc = vdiff / vlen
                v_perp = random_perpendicular(v_sc)
                phase = np.random.choice((-1.0, 1.0))
                angle = float(np.random.uniform(*angle_range, size=1))
                axis = np.cross(v_sc, v_perp)
                op = SymmOp.from_axis_angle_and_translation(axis, angle * phase)
                shift[i] = op.operate(v_sc) * steps[i]

                counter += 1
                if counter >= rnd:
                    break
            new_coord = self.coord[j] + shift.reshape(-1, 1).squeeze()
            self.update_coord(new_coord, j)


def random_perpendicular(vec):
    result = np.zeros(3)
    if vec[0] != 0.0:
        result[0] = np.random.uniform(-1.0, 1.0)
        result[1] = np.random.uniform(-1.0, 1.0)
        result[2] = (vec[1] * result[1] + vec[2] * result[2]) / vec[0]
        result = result / np.linalg.norm(result)
    elif vec[1] != 0.0:
        result[0] = np.random.uniform(-1.0, 1.0)
        result[2] = np.random.uniform(-1.0, 1.0)
        result[1] = (vec[0] * result[0] + vec[2] * result[2]) / vec[1]
        result = result / np.linalg.norm(result)
    elif vec[2] != 0.0:
        result[1] = np.random.uniform(-1.0, 1.0)
        result[0] = np.random.uniform(-1.0, 1.0)
        result[2] = (vec[1] * result[1] + vec[0] * result[0]) / vec[2]
        result = result / np.linalg.norm(result)
    else:
        raise ValueError("Vector(0,0,0) do not have perpendicular vectors!")
    return result
