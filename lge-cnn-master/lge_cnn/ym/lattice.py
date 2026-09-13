"""
    General group and algebra functions
    Grid functions
"""
from lge_cnn.ym.numba_target import myjit
import lge_cnn.ym.su as su

import math

"""
    SU(2) group & algebra functions
"""

# product of 4 matrices
@myjit
def mul4(a, b, c, d):
    ab = su.mul(a, b)
    cd = su.mul(c, d)
    abcd = su.mul(ab, cd)
    return abcd

# group add: g0 = g0 + f * g1
@myjit
def add_mul(g0, g1, f):  # TODO: inline explicitly everywhere and remove this function
    return su.add(g0, su.mul_s(g1, f))

# adjoint action a -> u a u^t
@myjit
def act(u, a):
    buffer1 = su.mul(u, a)
    result =  su.mul(buffer1, su.dagger(u))
    return result

# commutator of two su(2) elements
@myjit
def comm(a, b):
    buffer1 = su.mul(a, b)
    buffer2 = su.mul(b, a)
    result = add_mul(buffer1, buffer2, -1)
    return result


"""
    Plaquette functions
"""

# compute 'positive' plaquette U_{x, i, j}
@myjit
def plaq_pos(u, x, i, j, dims, acc):
    x1 = shift(x, i, 1, dims, acc)
    x2 = shift(x, j, 1, dims, acc)

    # U_{x, i} * U_{x+i, j} * U_{x+j, i}^t * U_{x, j}^t
    plaquette = mul4(u[x, i], u[x1, j], su.dagger(u[x2, i]), su.dagger(u[x, j]))
    return plaquette

# compute 'negative' plaquette U_{x, i, -j}
@myjit
def plaq_neg(u, x, i, j, dims, acc):
    x0 = x
    x1 = shift(shift(x0, i, 1, dims, acc), j, -1, dims, acc)
    x2 = shift(x1, i, -1, dims, acc)
    x3 = x2

    # U_{x, i} * U_{x+i-j, j}^t * U_{x-j, i}^t * U_{x-j, j}
    return mul4(u[x0, i], su.dagger(u[x1, j]), su.dagger(u[x2, i]), u[x3, j])


# compute general plaquette U_{x, oi*i, oj*j}
@myjit
def plaq(u, x, i, j, oi, oj, dims, acc):
    x0 = x
    x1 = shift(x0, i, oi, dims, acc)
    x2 = shift(x1, j, oj, dims, acc)
    x3 = shift(x2, i, -oi, dims, acc)

    u0 = get_link(u, x0, i, oi, dims, acc)
    u1 = get_link(u, x1, j, oj, dims, acc)
    u2 = get_link(u, x2, i, -oi, dims, acc)
    u3 = get_link(u, x3, j, -oj, dims, acc)

    # U_{x, i} * U_{x+i, j} * U_{x+i+j, -i} * U_{x+j, -j}
    return mul4(u0, u1, u2, u3)


# compute clover Q_{x, i, j}
@myjit
def clov(u, x, i, j, dims, acc):
    clov1 = plaq(u, x, i, j, +1, +1, dims, acc)
    clov2 = plaq(u, x, j, i, +1, -1, dims, acc)
    clov3 = plaq(u, x, i, j, -1, -1, dims, acc)
    clov4 = plaq(u, x, j, i, -1, +1, dims, acc)
    return su.add(su.add(clov1, clov2), su.add(clov3, clov4))


# compute Tr[C_ab C_cd] for topological charge (clover)
@myjit
def c_clov_abcd(u, x, a, b, c, d, dims, acc):
    C_ab = su.im(clov(u, x, a, b, dims, acc))
    C_cd = su.im(clov(u, x, c, d, dims, acc))
    return su.tr(su.mul(C_ab, C_cd)) / 4 ** 2

# compute Tr[C_ab C_cd] for topological charge (plaquette)
@myjit
def c_plaq_abcd(u, x, a, b, c, d, dims, acc):
    C_ab = su.im(plaq(u, x, a, b, +1, +1, dims, acc))
    C_cd = su.im(plaq(u, x, c, d, +1, +1, dims, acc))
    return su.tr(su.mul(C_ab, C_cd))

# compute Tr[C_ab C_cd] for topological charge (plaquette w/ orientation)
@myjit
def c_plaq2_abcd(u, x, a, b, c, d, dims, acc):
    result = 0.0
    for oa in [-1, 1]:
        for ob in [-1, 1]:
            C_ab = plaq(u, x, a, b, oa, ob, dims, acc)
            for oc in [-1, 1]:
                for od in [-1, 1]:
                    C_cd = plaq(u, x, c, d, oc, od, dims, acc)
                    result += oa * ob * oc * od * su.tr(su.mul(C_ab, C_cd))
    return result / 2.0 ** 4


@myjit
def get_link(u, x, i, oi, dims, acc):
    if oi > 0:
        return su.load(u[x, i])
    else:
        xs = shift(x, i, oi, dims, acc)
        return su.dagger(u[xs, i])


# compute a staple
@myjit
def staple(u, x, i, j, oj, dims, acc):
    x0 = x
    x1 = shift(x0, i, +1, dims, acc)
    x2 = shift(x1, j, oj, dims, acc)
    x3 = shift(x2, i, -1, dims, acc)

    u1 = get_link(u, x1, j, +oj, dims, acc)
    u2 = get_link(u, x2, i, -1, dims, acc)
    u3 = get_link(u, x3, j, -oj, dims, acc)

    return su.mul(su.mul(u1, u2), u3)


# compute sum over staples
@myjit
def staple_sum(x, d, u, dims, acc):
    result = su.zero()
    for j in range(len(dims)):
        if j != d:
            for oj in [-1, 1]:
                s = staple(u, x, d, j, oj, dims, acc)
                result = su.add(result, s)
    return result


# compute plaquette sum
@myjit
def plaquettes(x, d, u, dims, acc):
    res = su.zero()
    for j in range(len(dims)):
        if j != d:
            p1 = plaq(u, x, d, j, 1, +1, dims, acc)
            p2 = plaq(u, x, d, j, 1, -1, dims, acc)
            res = su.add(res, p1)
            res = su.add(res, p2)
    return res


"""
    Parallel transport of 'scalar' fields (aeta, peta)
"""

@myjit
def transport(f, u, x, i, o, dims, acc):
    xs = shift(x, i, o, dims, acc)
    if o > 0:
        u1 = u[x, i]  # np-array
        result = act(u1, f[xs])
    else:
        u2 = su.dagger(u[xs, i])  # tuple
        result = act(u2, f[xs])
    return result


"""
    Shift on d-dimensional grid
"""

@myjit
def shift(xi, i, o, dims, acc):
    res = xi
    di = xi // acc[i + 1]
    wdi = di % dims[i]
    if o > 0:
        if wdi == dims[i] - 1:
            res -= acc[i]
        res += acc[i+1]
    else:
        if wdi == 0:
            res += acc[i]
        res -= acc[i+1]
    return res

@myjit
def get_index(pos, dims, acc):
    index = pos[0]
    for d in range(1, len(dims)):
        index = index * dims[d] + pos[d]
    return index
