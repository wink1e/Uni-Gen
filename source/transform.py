# This script transform a typical carbon cluster .xyz file into U-cluster
from argparse import ArgumentParser
from typing import Optional
import numpy as np
import os
import re

parser = ArgumentParser(description='''
This program transform a carbon fullerene cluster into certain uranyl cluster.
--f: file_path of the carbon cluster .xyz file.
--c: chemical formula of the uranyl cluster i.e. K44U44O220
''')

parser.add_argument("--input-file", type=str, metavar='f', help="input xyz file")
parser.add_argument("--chemical-formula", type=str, metavar='c', help="input chemical formula")
parser.add_argument("--output-dir", type=str, metavar='o', help="output directory", default="./")
args = parser.parse_args()


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


def read_chemical_formula(formula):
    # 正则表达式来匹配化学式中的元素和计数
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")

    # 通过正则表达式找到所有匹配项
    matches = pattern.findall(formula)

    # 将匹配结果转换为字典
    elements = {}
    for element, count in matches:
        assert element in ATOM_ID_MASS.keys()
        if element not in elements:
            elements[element] = 0
        if count == '':
            count = 1
        else:
            count = int(count)
        elements[element] += count
    if elements['U']*5 != elements['O']:
        raise "Invalid chemical formula, please check"
    selected_ion = list(elements.keys())[0]
    assert elements[selected_ion] == elements['U']
    return elements


class MyMolecule:
    def __init__(self, coord: Optional[np.array] = None,
                 atom_type: Optional[np.array] = None) -> None:
        if coord is not None:
            self.coord = coord.reshape(-1, 3)
        else:
            self.coord = coord
        self.atom_type = atom_type
        self.center = None
        self.new_coord = None
        self.new_atom_type = None
        self.neighbor_list = []
        if self.coord is not None and self.atom_type is not None:
            self.init_center()
            self.centering()
            self.init_neighbor_list()

    def init_center(self):
        mass = np.array(list(map(lambda x: ATOM_ID_MASS[x][1], self.atom_type)))
        self.center = np.sum(np.multiply(self.coord, mass.reshape(len(self.atom_type), 1)), axis=0) / np.sum(mass)

    def init_neighbor_list(self):
        assert self.coord is not None, "No coordination are found, can't initialize neighbor list."
        for i in range(len(self.atom_type)):
            count = 0
            for j in range(i+1, len(self.atom_type)):
                diff = np.linalg.norm(self.coord[i]-self.coord[j])
                if diff < 1.6:
                    self.neighbor_list.append([i, j])
        assert len(self.neighbor_list) == int(1.5 * len(self.atom_type)), \
            "Initialize neighbor list failed, please check input file."

    def struct_print(self):
        print(f"Atom number:{len(self.atom_type)}, center at{self.center}")
        for i in range(len(self.atom_type)):
            print(f"{self.atom_type[i]}: {self.coord[i][:]}")

    @classmethod
    def load_from_md(cls, md_frame):
        molecule_obj = cls(md_frame.coord, md_frame.atom_type)
        return molecule_obj

    @classmethod
    def load_from_xyz(cls, file_path):
        assert os.path.exists(file_path) and file_path.endswith('.xyz'), f"Please check the path:{file_path}!"
        coord = []
        atom_type = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            atom_num = int(lines[0].strip())
            for i in range(atom_num):
                line = lines[i+2]
                items = line.split()
                atom_type.append(items[0])
                for j in range(1, 4):
                    coord.append(float(items[j]))
        coord = np.array(coord)
        atom_type = np.array(atom_type)
        return cls(coord=coord, atom_type=atom_type)

    def convert(self, chemical_form: str):
        elements = read_chemical_formula(chemical_form)
        # bond length shift
        assert self.coord is not None and self.atom_type is not None, "No carbon cluster structure."
        self.new_atom_type = np.char.replace(self.atom_type, 'C', 'U')
        self.new_coord = self.coord * (4.3/1.4)
        # Adding uranyl oxygen
        for i in range(len(self.atom_type)):
            temp_vec = self.new_coord[i]
            vec_len = np.linalg.norm(temp_vec, 2)
            vec_o_1 = temp_vec * (vec_len-1.8)/vec_len
            vec_o_2 = temp_vec * (vec_len+1.8)/vec_len
            self.new_coord = np.append(self.new_coord, vec_o_1.reshape(-1, 3), axis=0)
            self.new_atom_type = np.append(self.new_atom_type, ['O'], axis=0)
            self.new_coord = np.append(self.new_coord, vec_o_2.reshape(-1, 3), axis=0)
            self.new_atom_type = np.append(self.new_atom_type, ['O'], axis=0)
        # Adding hyper-oxygen # TODO Hydroxy group adding
        for bond in self.neighbor_list:
            temp_vec = (self.new_coord[bond[0]] + self.new_coord[bond[1]]) / 2
            vec_uu = self.new_coord[bond[0]] - self.new_coord[bond[1]]
            unit_perp = np.cross(temp_vec, vec_uu)
            unit_perp /= np.linalg.norm(unit_perp)
            vec_o_3 = temp_vec + unit_perp * 0.75
            vec_o_4 = temp_vec - unit_perp * 0.75
            self.new_coord = np.append(self.new_coord, vec_o_3.reshape(-1, 3), axis=0)
            self.new_atom_type = np.append(self.new_atom_type, ['O'], axis=0)
            self.new_coord = np.append(self.new_coord, vec_o_4.reshape(-1, 3), axis=0)
            self.new_atom_type = np.append(self.new_atom_type, ['O'], axis=0)
        # Adding ion
        selected_ion = list(elements.keys())[0]
        ion_num = elements[selected_ion]
        self.new_atom_type = np.append(self.new_atom_type, [selected_ion]*ion_num, axis=0)
        ion_count = 0
        while ion_count < ion_num:
            # TODO detect scale_factor
            scale_factor = np.amax(self.new_coord, axis=0) - np.amin(self.new_coord, axis=0)
            ion_position = np.multiply(np.random.random((1, 3)) - 0.5, scale_factor*0.8)
            success_flag = False
            for j in range(len(self.new_coord)):
                diff = np.linalg.norm(ion_position-self.new_coord[j])
                if diff < 2.0:
                    break
                if j == len(self.new_coord) - 1:
                    success_flag = True
            if success_flag:
                self.new_coord = np.append(self.new_coord, ion_position, axis=0)
                ion_count += 1

    def save_new_frame(self, save_path):
        elements_name = list(np.unique(self.new_atom_type))
        elements_list = list(self.new_atom_type)
        struc_str = ''
        for element in elements_name:
            elements_count = elements_list.count(element)
            struc_str += f"{element}_{str(elements_count)}"
        save_file = os.path.join(save_path, f"{struc_str}.xyz")
        with open(save_file, 'w') as f:
            f.write(f'{int(len(self.new_atom_type))}\n\n')
            for i in range(len(self.new_coord)):
                f.write(f"{self.new_atom_type[i]}\t{float(self.new_coord[i, 0])}\t{float(self.new_coord[i, 1])}"
                        f"\t{float(self.new_coord[i, 2])}\n")

    def centering(self):
        self.coord = self.coord - self.center
        self.center = np.array([0.0, 0.0, 0.0])


my_obj = MyMolecule.load_from_xyz(file_path=args.input_file)
my_obj.convert(chemical_form=args.chemical_formula)
my_obj.save_new_frame(save_path=args.output_dir)

#if __name__ == '__main__':
#    g_obj = MyMolecule.load_from_xyz(file_path='./C44-D2-83.xyz')
#    g_obj.struct_print()
#    g_obj.convert(chemical_form='K44U44O220')
#    g_obj.save_new_frame('./')
