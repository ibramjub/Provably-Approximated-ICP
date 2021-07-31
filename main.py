##################################################################################################################
# Python implementation of the paper "Provably Approximated ICP" (ICCV'21).
# By: Ibrahim Jubran, Alaa Maalouf, Ron Kimmel, Dan Feldman.
# Please cite our paper when using the code.
##################################################################################################################

import numpy as np
from numpy.linalg import norm
import random
from itertools import combinations
from multiprocessing import Process
import multiprocessing

from ICP import *

small_number = 0.0001

# Compute the nearest neighbour cost (Euclidean distance to the power of rho) of every point in Q to the points in Qtag
def Get_NN_Error(Qtag, Q, rho, p=2, threshold=np.infty, use_best_k=None):
    distance_array = cdist(Qtag.T, Q.T, 'minkowski', p=p) ** rho
    if not threshold == np.infty: distance_array[distance_array > threshold] = threshold
    if use_best_k:
        error = distance_array.min(axis=1)[:use_best_k].sum()
    else:
        error = distance_array.min(axis=1).sum()
    return error


# Generate a random rotation matrix
def rand_rotation_matrix(d):
    A = np.random.random((d, d))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:  Q[:, 0] = -Q[:, 0]
    return Q

# Compute a rotation matrix R such that Rx = y
def align_vectors(x, y):
    d = x.shape[0]

    # when x = y or x = -y #uglyPatch
    if (norm(x - y) < small_number):  return np.identity(d)
    if (norm(x + y) < small_number): return -np.identity(d)

    u = x / norm(x)
    v = (y - np.dot(u, y) * u)
    v = v.reshape(-1, 1) / norm(v)
    u = u.reshape(-1, 1)  # making v and u coumns vectors

    cos_theta = np.dot(x, y) / (norm(x) * norm(y))
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    uv = np.column_stack((u, v))
    R = np.identity(d) - u.dot(u.T) - v.dot(v.T) + np.dot(uv, np.dot(R_theta, uv.T))

    return R

# Compute a rotation matrix that aligns j<=d points (P) to some other j points (Q) in d-dimensional space.
# This is a direct implementation of the algorithm GET_ROT from the paper.
def get_rotation(P, Q):
    d, j = P.shape
    current_R_series = np.identity(d)
    while 1:
        p_unit = P[:, 0] / norm(P[:, 0])
        q_unit = Q[:, 0] / norm(Q[:, 0])
        R = align_vectors(p_unit, q_unit)

        if j == 1:
            return R.dot(current_R_series)

        W = np.linalg.svd(q_unit.reshape(1, -1), full_matrices=True)[2][1:]
        W_Rd = W.T.dot(W)

        P = W_Rd.dot(R).dot(P[:, 1:])
        Q = W_Rd.dot(Q[:, 1:])
        current_R_series = R.dot(current_R_series)
        j -= 1

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# The following function implements a modification of the algorithm APPROX_ALIGNMENT from our paper.
# It assumes the matching between P and Q is given, i.e., (p_i,q_i) are a matching pair.
# The function iterates over a subset of all the possible n^d subsets from P (the number of subsets is given in num_of_
# iters as a list, can contain multiple values for multiple runs), computes the approximated alignment (R,t) that aligns
# the current subset from P to the current subset from Q (using our novel algorithm), and returns the alignment that
# yields the smallest sum of l_p distances to the power of rho between all n pairs.
def approx_alignment_rand(P, Q, num_of_iters, return_dict=None, idx=None, rho=2, p=2):
    d, n = P.shape
    np.random.seed(random.seed())
    min_error = np.infty
    for _ in range(num_of_iters):
        idxes = np.random.choice(n, size=d, replace=False)
        i = idxes[0]

        P_in = P[:, idxes[1:]] - P[:, i].reshape(-1, 1)
        Q_in = Q[:, idxes[1:]] - Q[:, i].reshape(-1, 1)
        R = get_rotation(P_in, Q_in)
        t = np.matmul(R, P[:, i]) - Q[:, i]

        Qtag = np.matmul(R, P) - t.reshape(-1, 1)
        error_R = np.sum(norm(Q - Qtag, axis=0, ord=p) ** rho)

        if min_error > error_R:
            min_error = error_R
            opt_R = R
            opt_t = t

    if return_dict is None: return opt_R, opt_t, min_error
    return_dict[idx] = [opt_R, opt_t, min_error]


# The following function implements the algorithm ALIGN_AND_MATCH from our paper.
# It is similar to the previous function, however it also computes the matching between P and Q (which is assumed to be
# unknown). This is by iterating over a subset of all possible d point from P and over a subset of all possible d points
# from Q, and applies an algorithm similar to the above function to recover an approximated alignment (R,t), and a NN
# matching, which best register the current subset of P with the current subset of Q.
# When use_icp=True, an ICP algorithm is used as a post-process to refine the result. The ICP is run for icp_iters iterations.
# The function returns a dictionary of options. See main for an example on how to parse this dictionary.
def approx_registration_rand(P, Q, num_of_iters_list, return_dict=None,
                                        idx=None, p_options=[2],
                                        rho_options=[2], thresholds=[np.infty],
                                        use_icp=False, icp_iters=40, use_best_k=None):
    d, n = P.shape
    np.random.seed(random.seed())

    opt_erros_dict = {}
    opt_R_dict = {}
    opt_T_dict = {}

    opt_erros_dict_with_icp = {}
    opt_R_dict_with_icp = {}
    opt_T_dict_with_icp = {}

    idex_for_num_of_iterations = 0
    inter_now = num_of_iters_list[idex_for_num_of_iterations]

    for iteriter in range(num_of_iters_list[-1]):
        if iteriter > inter_now:
            old_iter = inter_now
            inter_now = num_of_iters_list[idex_for_num_of_iterations + 1]
            idex_for_num_of_iterations += 1
            for p in p_options:
                for rho in rho_options:
                    for threshold in thresholds:
                        key = "{}_{}_{}_{}".format(p, rho, threshold, inter_now)
                        old_key = "{}_{}_{}_{}".format(p, rho, threshold, old_iter)
                        opt_erros_dict[key] = opt_erros_dict[old_key]
                        opt_R_dict[key] = np.copy(opt_R_dict[old_key])
                        opt_T_dict[key] = np.copy(opt_T_dict[old_key])
        idxes_p = np.random.choice(n, size=d, replace=False);
        idxes_q = np.random.choice(n, size=d, replace=False)
        i_p = idxes_p[0]
        i_q = idxes_q[0]

        P_in = P[:, idxes_p[1:]] - P[:, i_p].reshape(-1, 1)
        Q_in = Q[:, idxes_q[1:]] - Q[:, i_q].reshape(-1, 1)
        R = get_rotation(P_in, Q_in)
        t = np.matmul(R, P[:, i_p]) - Q[:, i_q]

        Qtag = np.matmul(R, P) - t.reshape(-1, 1)

        for p in p_options:
            for rho in rho_options:
                for threshold in thresholds:

                    key = "{}_{}_{}_{}".format(p, rho, threshold, inter_now)
                    error = Get_NN_Error(Qtag, Q, rho=rho, p=p, threshold=threshold, use_best_k=use_best_k)

                    if key not in opt_erros_dict.keys() or opt_erros_dict[key] > error:
                        opt_erros_dict[key] = error
                        opt_R_dict[key] = np.copy(R)
                        opt_T_dict[key] = np.copy(t)

    if use_icp:
        for p in p_options:
            for rho in rho_options:
                for threshold in thresholds:
                    for iter in num_of_iters_list:

                        key = "{}_{}_{}_{}".format(p, rho, threshold, iter)

                        Qtag = np.matmul(opt_R_dict[key], P) - opt_T_dict[key].reshape(-1, 1)

                        _, _, _, R_icp, t_icp = icp_NN(Qtag.T, Q.T, init_pose=None, max_iterations=icp_iters,
                                                       tolerance=0.00001)
                        Qtag = t_icp.reshape(-1, 1) + np.dot(R_icp, Qtag)
                        icp_error = Get_NN_Error(Qtag, Q, rho=rho, p=p, threshold=threshold)
                        if key not in opt_erros_dict_with_icp.keys() or opt_erros_dict_with_icp[key] > icp_error:
                            opt_erros_dict_with_icp[key] = icp_error
                            opt_R_dict_with_icp[key] = np.copy(R_icp.dot(opt_R_dict[key]))
                            opt_T_dict_with_icp[key] = np.copy(
                                R_icp.dot(opt_T_dict[key].reshape(-1, 1)) - t_icp.reshape(-1, 1))

    if return_dict is None:
        return opt_erros_dict, opt_R_dict, opt_T_dict, opt_erros_dict_with_icp, opt_R_dict_with_icp, opt_T_dict_with_icp

    return_dict[idx] = {"errors": opt_erros_dict,
                        "R": opt_R_dict,
                        "T": opt_T_dict,
                        "errors_with_icp": opt_erros_dict_with_icp,
                        "R_with_icp": opt_R_dict_with_icp,
                        "T_with_icp": opt_T_dict_with_icp}


# The following function runs the previous function (with the unknown matching) in parallel, on procs CPU.
def approx_registration_rand_parallel(P, Q, num_of_iters_list, procs=1,
                                         p_options=[2], rho_options=[2],
                                         thresholds=[np.infty], use_icp=False, icp_iters=20, use_best_k=None):

    num_of_iters_per_proc = (np.array(num_of_iters_list) / procs).astype(int) + 1

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for idx in range(procs):
        porcces = Process(target=approx_registration_rand,
                          args=(P, Q, num_of_iters_per_proc,
                                return_dict, idx, p_options,
                                rho_options, thresholds, use_icp, icp_iters, use_best_k))
        jobs.append(porcces)
        porcces.start()
    for porcces in jobs: porcces.join()

    opt_erros_dict = {}
    opt_R_dict = {}
    opt_T_dict = {}

    opt_erros_dict_with_icp = {}
    opt_R_dict_with_icp = {}
    opt_T_dict_with_icp = {}
    for p in p_options:
        for rho in rho_options:
            for threshold in thresholds:
                for iter in num_of_iters_list:

                    key_to_load = "{}_{}_{}_{}".format(p, rho, threshold, int(iter / procs) + 1)
                    key_to_save = "{}_{}_{}_{}".format(p, rho, threshold, iter)

                    for i, val in return_dict.items():

                        error = val["errors"][key_to_load]
                        R = val["R"][key_to_load]
                        t = val["T"][key_to_load]

                        if key_to_save not in opt_erros_dict.keys() or opt_erros_dict[key_to_save] > error:
                            opt_erros_dict[key_to_save] = error
                            opt_R_dict[key_to_save] = np.copy(R)
                            opt_T_dict[key_to_save] = np.copy(t)

                        if use_icp:
                            error_with_icp = val["errors_with_icp"][key_to_load]
                            R_with_icp = val["R_with_icp"][key_to_load]
                            t_with_icp = val["T_with_icp"][key_to_load]

                            if key_to_save not in opt_erros_dict_with_icp.keys() or opt_erros_dict_with_icp[
                                key_to_save] > error_with_icp:
                                opt_erros_dict_with_icp[key_to_save] = error_with_icp
                                opt_R_dict_with_icp[key_to_save] = np.copy(R_with_icp)
                                opt_T_dict_with_icp[key_to_save] = np.copy(t_with_icp)

    return opt_erros_dict, opt_R_dict, opt_T_dict, opt_erros_dict_with_icp, opt_R_dict_with_icp, opt_T_dict_with_icp


################################################
# The following function computes an approximated rotation matrix that minimizes the sum of l_p distances to the power of rho.
# It basically iterates over all subsets of P, and computes the rotation matrix that aligns this subset to the corresponding
# subset of Q, using our novel method. It then returns the best such matrix.
def approx_rotation(P, Q, idx_options=None, error_func=None, p=2, rho=2):
    d, n = P.shape
    if idx_options is None: idx_options = list(combinations(range(n), d - 1))

    min_error = np.infty
    for i, idxes in enumerate(idx_options):
        for idxes_q in idx_options:
            P_in, Q_in = P[:, idxes], Q[:, idxes_q]
            R = get_rotation(P_in, Q_in)

            Qtag = np.matmul(R, P)

            distance_array = cdist(Qtag.T, Q.T, 'minkowski', p=p)  # ;print (rho)
            if not rho == 1: distance_array = distance_array ** rho
            row_ind, col_ind = linear_sum_assignment(distance_array)
            # error_R = norm(Q[:, col_ind] - Qtag, 'fro') ** rho
            error_R = np.sum(norm(Q[:, col_ind] - Qtag, axis=0, ord=p) ** rho)

            if min_error > error_R:
                min_error = error_R
                opt_R = R
    return opt_R, min_error


if __name__ == "__main__":
    np.random.seed(0)
    ##########################################################################
    ############## An example run when the matching is unknown ###############
    ##########################################################################
    n = 100
    d = 3
    Q = np.random.random((d, n))*10
    R_random = rand_rotation_matrix(d)
    t = np.random.random((d, 1))
    P = np.matmul(R_random, Q) + t
    np.random.shuffle(P.T)

    num_iterations = 3000
    procs = 2  # parallelize procs cpus
    p_options = [2]  # norms options
    rho_options = [1]  # sum of dists to the power of rho
    thresholds = [np.infty]
    use_icp = True  # run ICP on our opt solution

    opt_erros_dict, opt_R_dict, opt_T_dict, opt_erros_dict_with_icp, opt_R_dict_with_icp, opt_T_dict_with_icp = approx_registration_rand_parallel(
        P, Q, num_of_iters_list=[num_iterations], procs=procs,
        p_options=p_options, rho_options=rho_options,
        thresholds=thresholds, use_icp=use_icp, icp_iters=40, use_best_k=None)

    list_of_pairs = []
    lids_of_pairs_icp = []

    print('REGISTRATION TEST: ')
    for p in p_options:
        for rho in rho_options:
            for threshold in thresholds:
                key = "{}_{}_{}_{}".format(p, rho, threshold, num_iterations)
                Qtag = np.matmul(opt_R_dict[key], P) - opt_T_dict[key].reshape(-1, 1)
                Qtag_icp = np.matmul(opt_R_dict_with_icp[key], P) - opt_T_dict_with_icp[key].reshape(-1, 1)
                print(f'smallest recovered sum of l_{p} distances to the power of {rho} is {Get_NN_Error(Qtag, Q, rho, p=p, threshold=np.infty, use_best_k=None)}')
                print(f'smallest recovered sum of l_{p} distances to the power of {rho}, with ICP refinement, is {Get_NN_Error(Qtag_icp, Q, rho, p=p, threshold=np.infty, use_best_k=None)}')


    ##########################################################################
    ############## An example run when the matching is given ###############
    ##########################################################################
    n = 100
    d = 3
    p = 2
    rho = 1
    Q = np.random.random((d, n))*10
    R_random = rand_rotation_matrix(d)
    t = np.random.random((d, 1))
    P = np.matmul(R_random, Q) + t

    R_approx, t_approx, cost = approx_alignment_rand(P, Q, num_of_iters=1000, return_dict=None, idx=None, rho=rho, p=p)
    print('ALIGNMENT TEST: ')
    print(f'Smallest recovered sum of l_{p} distances to the power of {rho} is {cost}')
