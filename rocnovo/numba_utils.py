import numpy as np
import numpy.typing as npt
from numba import njit

from rocnovo.tokenizer.peptide import H2O, PROTON

@njit
def cal_ppm_error(t_mz: npt.NDArray, e_mz: float):
    ppm = (-t_mz + e_mz) / t_mz * 1e6
    return ppm

@njit
def mass2mz(mass_seq: npt.NDArray, charge: int):
    return (mass_seq + H2O) / charge + PROTON

@njit
def get_candidate_ptm_indices(mass_seq: npt.NDArray, charge: int, precursor_mz: float, ppm: int):
    mz = mass2mz(mass_seq, charge)
    ppm_error = cal_ppm_error(mz, precursor_mz)
    indices = np.where(np.abs(ppm_error) < ppm)[0]
    return indices, ppm_error[indices], mz[indices]


@njit
def cosine_similarity(A: npt.NDArray, B: npt.NDArray, eps=1e-12):
    norm_A = np.sqrt(np.sum(A ** 2, axis=1)) + eps
    norm_B = np.sqrt(np.sum(B ** 2, axis=1)) + eps
    normalize_A = A / norm_A[:, np.newaxis]
    normalize_B = B / norm_B[:, np.newaxis]
    scores = np.dot(normalize_A, normalize_B.T)
    return scores