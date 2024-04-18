import diff
import numpy as np


def calc_covmat(gen_coord, gen_atoms, ref_coord, ref_atoms, delta):
    r"""
    this function calculate the matching of the refenrence set and generated set.
    gen_coord: (n_frames, 3*n_atoms);
    gen_atoms: (n_frmaes, n_atoms);
    ref_coord: (n_frames, 3*n_atoms);
    ref_atoms: 
    """
    assert  0.0 < delta < 1.0, "The difference tolerance of generated set and refrence set must between 0 and 1."
    n_gen = len(gen_coord)
    n_ref = len(ref_coord)
    mat_value = 0.0
    cov_value = 0.0
    gen_sp = []
    ref_sp = []
    # Calc sprint_vec
    for i in range(n_gen):
        gen_sp.append(diff.get_sprint_vec(gen_coord[i, :], gen_atoms[i, :]))
    for j in range(n_ref):
        ref_sp.append(diff.get_sprint_vec(ref_coord[j, :], ref_atoms[j, :]))
    # Calculating matching and coverage
    for j in range(n_ref):
        diff_temp = []
        for i in range(n_gen):
            diff_temp.append(diff.get_diff(ref_sp[j], gen_sp[i]))
        mat_value += max(diff_temp)
        if max(diff_temp) > delta:
            cov_value += 1.0
    mat_value /= n_ref
    cov_value /= n_ref
    return mat_value, cov_value


if __name__ == "__main__":
    ...