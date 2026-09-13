import numpy as np
from lge_cnn.ym.numba_target import use_cuda, myjit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

if use_cuda:
    import numba.cuda as cuda
import lge_cnn.ym.su as su
import lge_cnn.ym.lattice as l
import math
import numba

pi = np.pi


class Simulation:
    def __init__(self, dims, beta, seed):
        self.beta = beta
        self.dims = np.array(dims, dtype=np.int)
        self.acc = np.append(np.cumprod(self.dims[::-1])[::-1], 1)
        self.n_dims = len(dims)
        self.sites = np.int(np.prod(dims))
        self.u = np.zeros((self.sites, self.n_dims, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.u[:, :, 0] = 1.0
        self.u_swap = np.zeros((self.sites, self.n_dims, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_u = self.u
        self.d_u_swap = self.u_swap
        self.d_dims = self.dims
        self.d_acc = self.acc

        self.threads_per_block = 256
        self.blocks = math.ceil(self.sites / self.threads_per_block)
        self.rng = create_xoroshiro128p_states(self.threads_per_block * self.blocks, seed=seed)
        self.copy_to_device()

    def copy_to_device(self):
        self.d_u = cuda.to_device(self.u)
        self.d_u_swap = cuda.to_device(self.u_swap)
        self.d_dims = cuda.to_device(self.dims)
        self.d_acc = cuda.to_device(self.acc)

    def copy_to_host(self):
        self.d_u.copy_to_host(self.u)
        self.d_u_swap.copy_to_host(self.u_swap)
        self.d_dims.copy_to_host(self.dims)
        self.d_acc.copy_to_host(self.acc)

    def apply_to_swap(self):
        copy_u_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.d_u_swap, self.d_dims)

    def swap(self):
        self.d_u, self.d_u_swap = self.d_u_swap, self.d_u

    def init(self, steps=10, use_flips=True):
        init_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.rng, steps, self.d_dims, use_flips)

    def metropolis(self, steps, updates_per_link=10, amplitude=0.5):
        for i in range(steps):
            for d in range(len(self.dims)):
                for cell_type in [0, 1]:
                    metropolis_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.rng, self.beta,
                                                                           cell_type, d, updates_per_link, amplitude, self.d_dims,
                                                                           self.d_acc)
                    numba.cuda.synchronize()

    def cooling(self, steps, use_buffer=False):
        if use_buffer:
            # save current config into u_swap
            self.apply_to_swap()

            # set references
            u0, u1 = self.d_u_swap, self.d_u
        else:
            # set references
            u0, u1 = self.d_u, self.d_u

        for i in range(steps):
            for d in range(len(self.dims)):
                for cell_type in [0, 1]:
                    # calculate cooling based on u0, write to u1
                    buffered_cooling_kernel[self.blocks, self.threads_per_block](self.sites, u0, u1, cell_type, d, self.d_dims, self.d_acc)
                    numba.cuda.synchronize()

            if use_buffer:
                # apply results from u1 (d_u) to u0 (d_u_swap)
                self.apply_to_swap()
                numba.cuda.synchronize()

    def flow(self, steps, step_size):

        # set references
        u0, u1 = self.d_u_swap, self.d_u

        for i in range(steps):
            # save current config into u_swap
            self.apply_to_swap()

            # write to u1 (which is d_u)
            # based on u0 (which is d_u_swap)
            flow_kernel[self.blocks, self.threads_per_block](self.sites, u0, u1, step_size, self.d_dims, self.d_acc)
            numba.cuda.synchronize()

    def normalize(self):
        normalize_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.d_dims)

    """
        Observables
    """

    def wilson_large(self, mu, nu, Lmu, Lnu):
        trW = np.zeros((self.sites), dtype=su.GROUP_TYPE_COMPLEX)
        d_trW = cuda.to_device(trW)
        wilson_large_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_trW, mu, nu, Lmu, Lnu, self.d_dims, self.d_acc)
        d_trW.copy_to_host(trW)
        return trW

    def topological_charge(self, mode=0):
        q = np.zeros((self.sites), dtype=su.GROUP_TYPE_REAL)
        d_q = cuda.to_device(q)
        topological_charge_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_q, self.d_dims, self.d_acc, mode)
        d_q.copy_to_host(q)
        return q

    def action_density(self, mode=0):
        sd = np.zeros(self.sites, dtype=su.GROUP_TYPE_REAL)
        d_sd = cuda.to_device(sd)
        if mode == 0:
            action_density_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_sd, self.beta, self.d_dims, self.d_acc)
        elif mode == 1:
            action_density_clover_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_sd, self.beta, self.d_dims, self.d_acc)
        else:
            print("Action density: unknown mode")
            return None
        d_sd.copy_to_host(sd)
        return sd

    """
        Configuration generation
    """

    def get_links(self):
        # switch to complex matrix representation
        out_u = np.zeros((self.sites, self.n_dims, su.NC, su.NC), dtype=su.GROUP_TYPE_COMPLEX)
        d_out_u = cuda.to_device(out_u)
        to_matrix_kernel[self.blocks, self.threads_per_block](self.sites, self.n_dims, self.d_u, d_out_u)
        d_out_u.copy_to_host(out_u)

        return out_u

    def get_plaquettes(self, include_hermitian=False):
        if include_hermitian:
            n_w = self.n_dims * (self.n_dims - 1)
        else:
            n_w = self.n_dims * (self.n_dims - 1) // 2

        # compute 1x1 Wilson loops (plaquettes)
        w = np.zeros((self.sites, n_w, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE_REAL)
        d_w = cuda.to_device(w)
        wilson_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_w, self.d_dims, self.d_acc, include_hermitian)

        # switch to complex matrix representation
        out_w = np.zeros((self.sites, n_w, su.NC, su.NC), dtype=su.GROUP_TYPE_COMPLEX)
        d_out_w = cuda.to_device(out_w)
        to_matrix_kernel[self.blocks, self.threads_per_block](self.sites, n_w, d_w, d_out_w)
        d_out_w.copy_to_host(out_w)

        return out_w

    def get_polyakov_loops(self, include_hermitian=False):
        if include_hermitian:
            n_p = 2 * self.n_dims
        else:
            n_p = self.n_dims

        # compute polyakov loops
        p = np.zeros((self.sites, n_p, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE_REAL)
        d_p = cuda.to_device(p)
        polyakov_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_p, self.d_dims, self.d_acc,
                                                             include_hermitian)

        # switch to complex matrix representation
        out_p = np.zeros((self.sites, n_p, su.NC, su.NC), dtype=su.GROUP_TYPE_COMPLEX)
        d_out_p = cuda.to_device(out_p)
        to_matrix_kernel[self.blocks, self.threads_per_block](self.sites, n_p, d_p, d_out_p)
        d_out_p.copy_to_host(out_p)

        return out_p

    def load_config(self, u_matrices):
        d_u_matrices = cuda.to_device(u_matrices)
        to_repr_kernel[self.blocks, self.threads_per_block](self.sites, self.n_dims, d_u_matrices, self.d_u)

    def random_gauge_transform(self, steps=10, amplitude=1.0):
        v = np.zeros((self.sites, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        d_v = cuda.to_device(v)
        random_gauge_transformation_kernel[self.blocks, self.threads_per_block](self.sites, self.rng, steps, amplitude, d_v)
        gauge_transform_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_v, self.d_dims, self.d_acc)

@cuda.jit
def copy_u_kernel(n, u, u_swap, dims):
    xi = cuda.grid(1)
    if xi < n:
        for d in range(len(dims)):
            for i in range(su.GROUP_ELEMENTS):
                u_swap[xi, d, i] = u[xi, d, i]

@cuda.jit
def init_kernel(n, u, rng, steps, dims, use_flips=True):
    """
    Initialization routine which randomizes all gauge links (warm start). If steps = 0, then all links are set to
    unit matrices.

    :param n:       number of lattice sizes
    :param u:       gauge link array
    :param rng:     random number states
    :param steps:   randomization steps per link
    :param dims:    lattice size array
    :param use_flips:   option to flip initial links from 1 to -1
    :return:
    """
    xi = cuda.grid(1)
    da = 3
    if xi < n:
        X_vals = numba.cuda.local.array(da, np.float32)
        for d in range(len(dims)):
            su.store(u[xi, d], su.unit())

            if use_flips:
                r = xoroshiro128p_uniform_float32(rng, xi)
                if r < 0.5:
                    su.store(u[xi, d], su.mul_s(u[xi, d], -1))

            for i in range(steps):
                for j in range(su.ALGEBRA_ELEMENTS):
                    X_vals[j] = xoroshiro128p_normal_float32(rng, xi)
                X = su.mexp(su.get_algebra_element(X_vals))
                su.store(u[xi, d], su.mul(X, u[xi, d]))


@cuda.jit
def normalize_kernel(n, u, dims):
    """
    Normalizes all gauge links, i.e. repairs unitarity and determinant.

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param dims:    lattice size array
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        for d in range(len(dims)):
            su.normalize(u[xi, d])


@cuda.jit
def metropolis_kernel(n, u, rng, beta, checkerboard_mode, d, updates, amplitude, dims, acc):
    """
    Performs a single metropolis sweep over the lattice in a checkerboard pattern.

    :param n:                   number of lattice sites
    :param u:                   gauge link array
    :param rng:                 random number states
    :param beta:                coupling constant
    :param checkerboard_mode:   checkerboard pattern: 'white' or 'black' cells
    :param updates:             number of consecutive updates per link
    :param dims:                lattice size array
    :param acc:                 cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    da = 3 # WARNING!! ONLY SU(2)
    if xi < n:
        X_components = numba.cuda.local.array(da, numba.float32)
        if checkerboard(xi, dims) == checkerboard_mode:
            staples = l.staple_sum(xi, d, u, dims, acc)

            # compute previous plaquette sum
            P0 = su.mul(u[xi, d], staples)

            for k in range(updates):
                # generate updated link
                for j in range(su.ALGEBRA_ELEMENTS):
                    X_components[j] = amplitude * xoroshiro128p_normal_float32(rng, xi)
                X = su.mexp(su.get_algebra_element(X_components))
                new_u = su.mul(X, u[xi, d])

                # compute updated plaquette sum
                P1 = su.mul(new_u, staples)

                # compute action difference
                delta_S = - beta / su.NC * (su.tr(P1).real - su.tr(P0).real)

                # propose update
                r = xoroshiro128p_uniform_float32(rng, xi)
                if r <= math.exp(- delta_S):
                    su.store(u[xi, d], new_u)
                    P0 = P1


@cuda.jit
def cooling_kernel(n, u, checkerboard_mode, d, dims, acc):
    xi = cuda.grid(1)
    if xi < n:
        if checkerboard(xi, dims) == checkerboard_mode:
            staples = l.staple_sum(xi, d, u, dims, acc)
            staples = su.dagger(staples)
            det_staples = math.sqrt(su.det(staples))
            new_u = su.mul_s(staples, 1.0 / det_staples)
            su.store(u[xi, d], new_u)


@cuda.jit
def buffered_cooling_kernel(n, u0, u1, checkerboard_mode, d, dims, acc):
    xi = cuda.grid(1)
    if xi < n:
        if checkerboard(xi, dims) == checkerboard_mode:
            staples = l.staple_sum(xi, d, u0, dims, acc)
            staples = su.dagger(staples)
            det_staples = math.sqrt(su.det(staples))
            new_u = su.mul_s(staples, 1.0 / det_staples)
            su.store(u1[xi, d], new_u)


@cuda.jit
def flow_kernel(n, u0, u1, dt, dims, acc):
    xi = cuda.grid(1)
    if xi < n:
        for d in range(len(dims)):
            old_u = u0[xi, d]
            omega = l.plaquettes(xi, d, u0, dims, acc)
            omega_ah = su.mul_s(su.ah(omega), -dt)
            exp_staples = su.mexp(omega_ah)
            new_u = su.mul(exp_staples, old_u)
            su.store(u1[xi, d], new_u)


@myjit
def checkerboard(xi, dims):
    """
    Tests whether a lattice site is 'white' or 'black'

    :param xi:      lattice index
    :param dims:    lattice size array
    :return:        0 or 1 depending on 'white' or 'black'
    """
    res = 0
    x0 = xi
    for d in range(len(dims) - 1, -1, -1):
        cur_pos = x0 % dims[d]
        res += cur_pos
        x0 -= cur_pos
        x0 /= dims[d]
    return res % 2


@cuda.jit
def polyakov_kernel(n, u, p, dims, acc, include_hermitian=False):
    """
    Computes the Polyakov loop in every direction at every lattice site

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param p:       output array for polyakov loops
    :param dims:    lattice size array
    :param acc:     cumulative product of lattice sizes
    :param include_hermitian:   option to include hermitian conjugates of polyakov loops
    :return:
    """
    xi = cuda.grid(1)
    D = len(dims)
    if xi < n:
        p_i = 0
        for i in range(D):
            x0 = xi

            # compute polyakov loop starting at xi in direction i
            P = su.load(u[x0, i])
            for t in range(dims[i] - 1):
                x0 = l.shift(x0, i, +1, dims, acc)
                P = su.mul(P, u[x0, i])

            # store in array
            su.store(p[xi, p_i], P)

            # also store hermitian conjugate
            if include_hermitian:
                su.store(p[xi, D + p_i], su.dagger(P))

            p_i += 1


@cuda.jit
def wilson_kernel(n, u, w, dims, acc, include_hermitian=False):
    """
    Computes all 1x1 Wilson loops on the lattice

    :param n:                   number of lattice sites
    :param u:                   gauge link array
    :param w:                   output array for wilson loops
    :param dims:                lattice size array
    :param acc:                 cumulative product of lattice sizes
    :param include_hermitian:   option to include hermitian conjugates of wilson loops
    :return:
    """
    xi = cuda.grid(1)
    D = len(dims)
    if xi < n:
        w_i = 0
        for i in range(D):
            for j in range(i):
                W = l.plaq(u, xi, i, j, 1, 1, dims, acc)
                su.store(w[xi, w_i], W)

                # also store hermitian conjugate
                if include_hermitian:
                    su.store(w[xi, D * (D - 1) // 2 + w_i], su.dagger(W))

                w_i += 1


@cuda.jit
def wilson_large_kernel(n, u, w, mu, nu, Lmu, Lnu, dims, acc):
    """
    Computes all Lmu x Lnu Wilson loops on the lattice

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param w:       output array for wilson loops
    :param dims:    lattice size array
    :param acc:     cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        x0 = xi

        x1 = x0
        U = su.unit()
        for x in range(Lmu):
            U = su.mul(U, u[x1, mu])
            x1 = l.shift(x1, mu, +1, dims, acc)

        x2 = x1
        for y in range(Lnu):
            U = su.mul(U, u[x2, nu])
            x2 = l.shift(x2, nu, +1, dims, acc)

        x3 = x2
        for x in range(Lmu):
            x3 = l.shift(x3, mu, -1, dims, acc)
            U = su.mul(U, su.dagger(u[x3, mu]))

        x4 = x3
        for y in range(Lnu):
            x4 = l.shift(x4, nu, -1, dims, acc)
            U = su.mul(U, su.dagger(u[x4, nu]))

        w[xi] = su.tr(U) / su.NC


@cuda.jit
def to_matrix_kernel(n, num_components, arr, out):
    """
    Converts an array 'arr' of parametrized SU(N_c) matrices to
    complex N_c x N_c matrices (fundamental representation).
    """
    x = cuda.grid(1)
    if x < n:
        for c in range(num_components):
            a = arr[x, c]
            c_matrix = su.to_matrix(a)
            for i in range(su.NC):
                for j in range(su.NC):
                    out[x, c, i, j] = c_matrix[i * su.NC + j]


@cuda.jit
def to_repr_kernel(n, num_components, arr, out):
    """
    Converts an array 'arr' of complex N_c x N_c matrices (fundamental
    representation, flattened) to parametrized SU(N_c) matrices.
    """
    x = cuda.grid(1)
    if x < n:
        for c in range(num_components):
            matrix = arr[x, c]
            repr = su.to_repr(matrix)
            su.store(out[x, c], repr)


@cuda.jit
def random_gauge_transformation_kernel(n, rng, steps, amplitude, v):
    """
    Generates a random gauge transformation and stores it in 'v'
    """
    xi = cuda.grid(1)
    da = 3 # WARNING!! ONLY SU(2)
    if xi < n:
        X_vals = numba.cuda.local.array(da, numba.float32)
        su.store(v[xi], su.unit())
        for i in range(steps):
            for j in range(su.ALGEBRA_ELEMENTS):
                X_vals[j] = amplitude * xoroshiro128p_normal_float32(rng, xi)
            X = su.mexp(su.get_algebra_element(X_vals))
            su.store(v[xi], su.mul(X, v[xi]))


@cuda.jit
def gauge_transform_kernel(n, u, v, dims, acc):
    """
    Applies the gauge transformation 'v' to the links 'u'
    """
    xi = cuda.grid(1)
    if xi < n:
        for i in range(len(dims)):
            xs = l.shift(xi, i, +1, dims, acc)
            new_u = su.mul(v[xi], u[xi, i])
            new_u = su.mul(new_u, su.dagger(v[xs]))
            su.store(u[xi, i], new_u)

@myjit
def lc_2(a, b):
    """
    Levi-Civita in 2D
    """
    if a == b:
        return 0
    elif a < b:
        return -1
    else:
        return +1

@myjit
def lc_4(a, b, c, d):
    """
    Levi-Civita in 4D
    """
    return lc_2(b, a) * lc_2(c, b) * lc_2(c, a) * lc_2(d, c) * lc_2(d, b) * lc_2(d, a)

@cuda.jit
def topological_charge_kernel(n, u, q, dims, acc, mode):
    """
    Computes the topological charge density q

    mode 0: plaquette discretization
    mode 1: clover discretization
    """
    xi = cuda.grid(1)
    if xi < n:
        if mode == 0:
            tr_c_0312 = l.c_plaq_abcd(u, xi, 0, 3, 1, 2, dims, acc)
            tr_c_0213 = l.c_plaq_abcd(u, xi, 0, 2, 1, 3, dims, acc)
            tr_c_0123 = l.c_plaq_abcd(u, xi, 0, 1, 2, 3, dims, acc)
        elif mode == 1:
            tr_c_0312 = l.c_clov_abcd(u, xi, 0, 3, 1, 2, dims, acc)
            tr_c_0213 = l.c_clov_abcd(u, xi, 0, 2, 1, 3, dims, acc)
            tr_c_0123 = l.c_clov_abcd(u, xi, 0, 1, 2, 3, dims, acc)
        elif mode == 2:
            tr_c_0312 = l.c_plaq2_abcd(u, xi, 0, 3, 1, 2, dims, acc)
            tr_c_0213 = l.c_plaq2_abcd(u, xi, 0, 2, 1, 3, dims, acc)
            tr_c_0123 = l.c_plaq2_abcd(u, xi, 0, 1, 2, 3, dims, acc)
        else:
            return None

        local_q = tr_c_0312 - tr_c_0213 + tr_c_0123
        q[xi] = local_q.real / (4.0 * pi ** 2)


@cuda.jit
def action_density_kernel(n, u, sd, beta, dims, acc):
    """
        Action density based on the Wilson action
    """
    xi = cuda.grid(1)
    if xi < n:
        sd_local = 0.0
        for mu in range(len(dims)):
            for nu in range(mu+1, len(dims)):
                plaq = l.plaq(u, xi, mu, nu, +1, +1, dims, acc)
                sd_local += 1.0 - su.tr(plaq).real / su.NC
        sd[xi] = beta * sd_local


@cuda.jit
def action_density_clover_kernel(n, u, sd, beta, dims, acc):
    """
        Action density based on the clover discretization
    """
    xi = cuda.grid(1)
    if xi < n:
        sd_local = 0.0
        for mu in range(len(dims)):
            for nu in range(mu+1, len(dims)):
                clov = su.ah(l.clov(u, xi, mu, nu, dims, acc))
                clov = su.mul_s(clov, 0.25)
                sd_local -= su.tr(su.mul(clov, clov)).real
        sd[xi] = beta * sd_local / (2 * su.NC)
