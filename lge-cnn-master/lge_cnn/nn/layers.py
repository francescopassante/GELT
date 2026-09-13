import torch
from torch import einsum, roll, Tensor
import numpy as np
import time

# a few options
# switch between einsum calculations with re/im split and complex
use_fast_mm = True

# switch for various equivalent implementations of the bilinear layer
bilin_implementation = 2


class BetterLConvBilin(torch.nn.Module):
    def __init__(self, dims, kernel_size, n_in, n_out, nc):
        super(BetterLConvBilin, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc

        # determine number of input channels
        self.n_in1 = self.n_in

        # number of transported matrices
        D = len(dims)
        self.n_in2 = self.n_in * (1 + 2 * D * (self.kernel_size - 1))

        # create weight tensors
        # bias: w0 [n_out]
        self.weight0 = torch.nn.Parameter(data=torch.Tensor(self.n_out,), requires_grad=True)
        # conv: w1 [n_out, n_in2, 2]
        self.weight1 = torch.nn.Parameter(data=torch.Tensor(self.n_out, self.n_in2, 2), requires_grad=True)
        # bilin-conv: w2 [n_out, n_in, n_in2, 2, 2]
        self.weight2 = torch.nn.Parameter(data=torch.Tensor(self.n_out, self.n_in1, self.n_in2, 2, 2), requires_grad=True)

        # set initial weights
        torch.nn.init.constant_(self.weight0.data, 0.0)
        torch.nn.init.normal_(self.weight1.data, std=0.1 / np.sqrt(2 * self.n_in2))
        torch.nn.init.normal_(self.weight2.data, std=0.1 / np.sqrt(4 * self.n_in1 * self.n_in2))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.view_as_complex(torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)).clone()
        self.unit_shapes = []
        self.unit_tensors = {}

    def forward(self, x):
        global use_fast_bl
        u, w = unpack_x(x, len(self.dims))
        uc = torch.view_as_complex(u)

        # input1: local terms
        w1 = torch.view_as_complex(w)

        # input2: local + transported terms
        w2 = [w1]
        D = len(self.dims)
        for mu in range(D): # lattice axis
            for o in [+1, -1]: # orientation
                wt = w1
                for k in range(self.kernel_size - 1): # kernel_size
                    wt = complex_transport(uc, wt, mu, o, self.dims)
                    w2.append(wt)
        # combine terms into a single tensor
        w2 = torch.cat(w2, dim=2)

        # if we want to achieve very cheap isotropy, we should have mu-independent weights
        # this can be achieved if all contributions with the same mu and k are summed together into a channel component
        # TODO: try this out?
        
        # conv terms
        p = self.weight1.type(w1.dtype) # cast weight tensor to complex
        w_out =  torch.einsum('vw, bxwij -> bxvij', p[..., 0], w2)
        w_out += torch.einsum('vw, bxwji -> bxvij', p[..., 1], w2.conj())
        
        # bilinear conv terms
        p = self.weight2.type(w1.dtype)
        w_out += torch.einsum('bxvij, bxuvjk -> bxuik', w1,         torch.einsum('uvw, bxwij -> bxuvij', p[..., 0, 0], w2))
        w_out += torch.einsum('bxvji, bxuvjk -> bxuik', w1.conj(),  torch.einsum('uvw, bxwij -> bxuvij', p[..., 1, 0], w2))
        w_out += torch.einsum('bxvij, bxuvjk -> bxuik', w1,         torch.einsum('uvw, bxwji -> bxuvij', p[..., 0, 1], w2.conj()))
        w_out += torch.einsum('bxvji, bxuvjk -> bxuik', w1.conj(),  torch.einsum('uvw, bxwji -> bxuvij', p[..., 1, 1], w2.conj()))

        # bias term
        p = self.weight0.type(w1.dtype)

        # update unit tensor
        unit_shape = (w.shape[0], w.shape[1], self.nc, self.nc)
        if unit_shape not in self.unit_shapes:
            print("Recreate unit_tensor")
            self.unit_shapes.append(unit_shape)
            unit_matrix = self.unit_matrix.to(w.device)
            self.unit_tensors[unit_shape] = unit_matrix.expand(unit_shape)
        unit_tensor = self.unit_tensors[unit_shape]
        
        w_out += torch.einsum('v, bxij -> bxvij', p, unit_tensor)

        return repack_x(u, torch.view_as_real(w_out))

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

class LConvBilin(torch.nn.Module):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True, use_symmetric=True):
        super(LConvBilin, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.use_symmetric = use_symmetric

        # determine kernel structure from `kernel_size` argument
        # example 1: kernel_size=2, symmetric=True -> kernel_range = [-1, 0, +1] * DIM
        # example 2: kernel_size=2, symmetric=False -> kernel_range = [0, +1] * DIM
        # example 3: kernel_size=[-1, +1] -> kernel_range = [-1, 0, +1] * DIM
        self.kernel_range = None
        if isinstance(kernel_size, int):
            if kernel_size > 0:
                if not use_symmetric:
                    self.kernel_range = [[0, kernel_size-1]] * len(dims)
                else:
                    self.kernel_range = [[-(kernel_size-1), kernel_size-1]] * len(dims)
            else:
                raise ValueError("kernel_size should be a positive integer. Got {}.".format(kernel_size))
        elif isinstance(kernel_size, list):
            if len(kernel_size) == 2:
                a, b = kernel_size[0], kernel_size[1]
                if isinstance(a, int) and isinstance(b, int) and a <= 0 and b >= 0:
                    self.kernel_range = [[a, b]] * len(dims)
                else:
                    raise ValueError("kernel_size should be a list containing two integers a,b with a <= 0 and b >= "
                                     "0. Got {}.".format(kernel_size))
            else:
                raise ValueError("kernel_size should be a list containing two integers a,b with a <= 0 and b >= 0. "
                                 "Got {}.".format(kernel_size))
        else:
            raise ValueError("Invalid kernel_size. Got {}.".format(kernel_size))

        # initialize weights
        w_in_size = self.n_in
        t_w_size = self.n_in * (1 + sum([(abs(x[0]) + abs(x[1])) for x in self.kernel_range]))
        w_out_size = self.n_out

        # double channels for complex conjugation
        w_in_size = 2 * w_in_size
        t_w_size = 2 * t_w_size

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        variance = 1.0 / (w_in_size * t_w_size)
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        """
        print("*** L-ConvBilin layer")
        print(f"kernel_range: {self.kernel_range}")
        print(f"weight.shape: {self.weight.shape}")
        print(f"Number of weights: {np.prod(self.weight.shape)}")
        """

        # set residual and bias terms to zero
        torch.nn.init.constant_(self.weight.data[:, :, -1], 0.0)
        torch.nn.init.constant_(self.weight.data[:, -1, :], 0.0)

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

        # caches for unit_tensor
        self.unit_tensors = {}

    def get_unit(self, shape, device):
        shape = tuple(shape)

        if shape not in self.unit_tensors.keys():
            unit_matrix = self.unit_matrix.to(device)
            unit_tensor = unit_matrix.expand(shape)
            self.unit_tensors[shape] = unit_tensor
        
        return self.unit_tensors[shape]


    def forward(self, x):
        global bilin_implementation, transport_implementation
        u, w = unpack_x(x, len(self.dims))

        # transported terms
        t_w = [w]
        # gather all terms along lattice axes up to kernel_size
        for axis in range(len(self.dims)):
            for i, o in zip([0, 1], [-1, +1]):
                w_transport = w
                kernel_size = abs(self.kernel_range[axis][i])
                for d in range(kernel_size):
                    # get transported terms
                    for step in range(self.dilation):
                        # TODO: use complex implementation here
                        w_transport = transport(u, w_transport, axis=axis, orientation=o, dims=self.dims)
                    # and add to list
                    t_w.append(w_transport)
        # combine terms into a single tensor
        t_w = torch.cat(t_w, dim=2)

        # enlarge tensors by complex conjugates
        # not required for bilin_implementation == 3
        if bilin_implementation != 3:
            w_c, t_w_c = cconj(w), cconj(t_w)
            w = repack_x(w, w_c)
            t_w = repack_x(t_w, t_w_c)

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_tensor = self.get_unit(unit_shape, device=w.device)

            # not required for bilin_implementation == 3
            if bilin_implementation != 3:
                w   = repack_x(w,   unit_tensor)
                t_w = repack_x(t_w, unit_tensor)
        
        # v0: perform multiplication with re/im split, two einsums, multiply local and transported Ws first
        if bilin_implementation == 0:
            w = complex_einsum('bxvij, bxwjk -> bxvwik', w, t_w)
            weight = self.weight
            w = einsum('uvw, bxvwijc -> bxuijc', weight, w)

        # v1: complex multiplication, single einsum
        elif bilin_implementation == 1:
            w_c, t_w_c = torch.view_as_complex(w), torch.view_as_complex(t_w)
            weight_c = self.weight.type(w_c.dtype)
            w_c = einsum('uvw, bxvij, bxwjk -> bxuik', weight_c, w_c, t_w_c)
            w = torch.view_as_real(w_c)
        
        # v2: complex multiplication, two einsums, multiply transported Ws first [the good one]
        elif bilin_implementation == 2:
            w_c, t_w_c = torch.view_as_complex(w), torch.view_as_complex(t_w)
            weight_c = self.weight.type(w_c.dtype)
            tmp = einsum('uvw, bxwjk -> bxuvjk', weight_c, t_w_c)
            w_c = einsum('bxvij, bxuvjk -> bxuik', w_c, tmp)
            w = torch.view_as_real(w_c)

        # v3: complex multiplication, multiple einsums, transported Ws first, no precomputed conjugates, limited explicit unit elements
        elif bilin_implementation == 3:
            w_c, t_w_c = torch.view_as_complex(w), torch.view_as_complex(t_w)
            weight_c = self.weight.type(w_c.dtype)

            # get number of channels 
            n0 = self.n_out
            n1 = self.n_in
            n2 = self.n_in * (1 + sum([(abs(x[0]) + abs(x[1])) for x in self.kernel_range]))

            # list for storing results (cache this in self?)
            ws = torch.zeros(size=(w_c.shape[0], w_c.shape[1], n0, self.nc, self.nc), dtype=w_c.dtype, device=w_c.device)

            # V.W
            p = weight_c[:, :n1, :n2]
            ws += einsum('bxvij, bxuvjk -> bxuik', w_c,        einsum('uvw, bxwjk -> bxuvjk', p, t_w_c))

            # V*.W
            p = weight_c[:, n1:2*n1, :n2]
            ws += einsum('bxvji, bxuvjk -> bxuik', w_c.conj(), einsum('uvw, bxwjk -> bxuvjk', p, t_w_c))

            # V.W*
            p = weight_c[:, :n1, n2:2*n2]
            ws += einsum('bxvij, bxuvjk -> bxuik', w_c,        einsum('uvw, bxwkj -> bxuvjk', p, t_w_c.conj()))

            # V*-W*
            p = weight_c[:, n1:2*n1, n2:2*n2]
            ws += einsum('bxvji, bxuvjk -> bxuik', w_c.conj(), einsum('uvw, bxwkj -> bxuvjk', p, t_w_c.conj()))

            if self.use_unit_elements:
                # V.1
                p = weight_c[:, :n1, -1]
                ws += einsum('uv, bxvij -> bxuij', p, w_c)

                # V*.1
                p = weight_c[:, n1:2*n1, -1]
                ws += einsum('uv, bxvji -> bxuij', p, w_c.conj())

                # 1.W
                p = weight_c[:, -1, :n2]
                ws += einsum('uv, bxvij -> bxuij', p, t_w_c)
                
                # 1.W*
                p = weight_c[:, -1, n2:2*n2]
                ws += einsum('uv, bxvji -> bxuij', p, t_w_c.conj())

                # 1.1 (annoying ...)
                p = weight_c[:, -1, -1]
                unit = torch.view_as_complex(self.unit_matrix.to(w.device)).expand(ws.shape)

                ws += einsum('u, bxuij -> bxuij', p, unit)


            # cast back to real-valued tensor
            w = torch.view_as_real(ws)

        # v4: complex multiplication, two einsums, Ws first
        elif bilin_implementation == 4:
            w_c, t_w_c = torch.view_as_complex(w), torch.view_as_complex(t_w)
            weight_c = self.weight.type(w_c.dtype)
            tmp = einsum('uvw, bxvjk -> bxuwjk', weight_c, w_c)
            w_c = einsum('bxuwij, bxwjk -> bxuik', tmp, t_w_c)
            w = torch.view_as_real(w_c)

        # v-1: turn off bilinear layer (testing only)
        elif bilin_implementation == -1:
            w_shape = list(w.shape)
            w_shape[2] = self.n_out
            w = torch.zeros(size=w_shape, device=w.device, dtype=w.dtype)

        return repack_x(u, w)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

class LConv(torch.nn.Module):
    """
        A lattice gauge equivariant convolution
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True):
        super(LConv, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements

        # initialize weights
        w_in_size = self.n_in
        t_w_size = self.n_in * (2 * (self.kernel_size - 1) * len(self.dims) + 1)
        w_out_size = self.n_out

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        # TODO: CHECK THIS
        variance = 1.0 / t_w_size
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, u, w):
        # local term
        transported_terms = [w.clone()]

        # gather all terms along lattice axes up to kernel_size
        for orientation in [+1, -1]:
            for axis in range(len(self.dims)):
                w_transport = w.clone()
                for d in range(1, self.kernel_size):
                    # get transported terms
                    for step in range(self.dilation + 1):
                        w_transport = transport(u, w_transport, axis=axis, orientation=orientation, dims=self.dims)
                    # and add to list
                    transported_terms.append(w_transport)

        # combine terms into a single tensor
        t_w = torch.cat(transported_terms, dim=2)

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(t_w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device)
            unit_tensor = unit_matrix.expand(unit_shape)
            t_w = repack_x(t_w, unit_tensor)

        # perform multiplication and apply weights
        w = einsum('uv, bxvijc -> bxuijc', self.weight, t_w)

        return u, w

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

class LBilin(torch.nn.Module):
    """
        A lattice gauge equivariant bilinear layer
    """
    def __init__(self, dims, n_in_1, n_in_2, n_out, nc, init_w=1.0, use_unit_elements=True):
        super(LBilin, self).__init__()
        self.dims = dims
        self.n_in_1 = n_in_1
        self.n_in_2 = n_in_2
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements

        # initialize weights
        w_in_1_size = self.n_in_1
        w_in_2_size = self.n_in_2
        w_out_size = self.n_out

        if self.use_unit_elements:
            w_in_1_size += 1
            w_in_2_size += 1

        variance = 1.0 / (w_in_1_size * w_in_2_size)
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, w_in_1_size, w_in_2_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, u, w1, w2):
        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w1.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w1.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w1 = repack_x(w1, unit_tensor)
            w2 = repack_x(w2, unit_tensor)

        # perform multiplication and apply weights
        w = complex_einsum('bxvij, bxwjk -> bxvwik', w1, w2)
        w = einsum('uvw, bxvwijc -> bxuijc', self.weight, w)

        return u, w

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

class LLin(torch.nn.Module):
    """
        A lattice gauge equivariant linear layer
    """
    def __init__(self, dims, n_in, n_out, nc, init_w=1.0, use_conjugated=False, use_unit_elements=False):
        super(LLin, self).__init__()
        self.dims = dims
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_conjugated = use_conjugated
        self.use_unit_elements = use_unit_elements

        # initialize weights
        w_in_size = self.n_in
        w_out_size = self.n_out

        if self.use_conjugated:
            w_in_size *= 2

        if self.use_unit_elements:
            w_in_size += 1

        variance = 1.0 / (w_in_size)
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, w_in_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))

        # enlarge tensors by hermitian conjugates
        if self.use_conjugated:
            w_c = cconj(w)
            w = repack_x(w, w_c)

        # enlarge tensors by unit matrices (adds bias)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w = repack_x(w, unit_tensor)
            
        # perform multiplication and apply weights
        w = einsum('vw, bxwijc -> bxvijc', self.weight, w)

        return repack_x(u, w)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

class LTrace(torch.nn.Module):
    """
        Computes the trace of Wilson loops.
        This renders the output gauge-invariant.
    """
    def __init__(self, dims):
        self.update_dims(dims)
        super(LTrace, self).__init__()

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))
        tr = einsum('bxwiic -> bxwc', w)

        return tr

    def update_dims(self, dims):
        self.dims = dims

class LConvBilin2(torch.nn.Module):
    """
        A module that combines separate instances of LConv and LBilin into one module.
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_inter, n_out, nc, use_unit_elements=True, extended=False):
        super(LConvBilin2, self).__init__()
        self.dims = dims
        self.lconv1 = LConv(dims, kernel_size, dilation, n_in, n_inter, nc, use_unit_elements=use_unit_elements)
        self.extended = extended
        if not self.extended:
            self.lbilin = LBilin(dims, n_inter, n_inter, n_out, nc, use_unit_elements=use_unit_elements)
        else:
            self.lbilin = LBilin(dims, n_in + n_inter, n_in + n_inter, n_out, nc, use_unit_elements=use_unit_elements)

    def forward(self, x):
        u, w1 = unpack_x(x, len(self.dims))
        u, w2 = self.lconv1(u, w1)
        if not self.extended:
            u, w3 = self.lbilin(u, w2, w2)
        else:
            # combine w1 and w2
            wn = repack_x(w1, w2)
            u, w3 = self.lbilin(u, wn, wn)
        return repack_x(u, w3)

    def update_dims(self, dims):
        self.dims = dims
        self.lconv.update_dims(dims)
        self.lbilin.update_dims(dims)

class Plaq(torch.nn.Module):
    """
        A module that computes plaquettes from a link configuration
    """

    def __init__(self, dims, nc, orientation=-1):
        super(Plaq, self).__init__()
        self.dims = dims
        self.nc = nc
        self.orientation = orientation

    def get_number_of_channels(self):
        n_dims = len(self.dims)
        n_plaqs = n_dims * (n_dims - 1) // 2
        if self.orientation == -1 or self.orientation == +1:
            return n_plaqs
        else:
            return 2*n_plaqs

    def forward(self, u):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)
        n_dims = len(self.dims)
        plaqs = []
        for mu in range(n_dims):
            
            if self.orientation == -1:
                nu0 = 0
                nu1 = mu
            elif self.orientation == +1:
                nu0 = mu+1
                nu1 = n_dims
            else:
                nu0 = 0
                nu1 = n_dims
            
            for nu in range(nu0, nu1):
                if mu != nu:
                    u1 = u[:, :, mu, :, :, :]
                    u_shift_mu = shift(u, mu, -1, self.dims)
                    u2 = u_shift_mu[:, :, nu, :, :, :]
                    u_shift_nu = shift(u, nu, -1, self.dims)
                    u3 = u_shift_nu[:, :, mu, :, :, :]
                    u4 = u[:, :, nu, :, :, :]

                    u12 = complex_einsum('bxij, bxjk -> bxik', u1, u2)
                    u34 = complex_einsum('bxji, bxkj -> bxik', u3, u4, conj_a=-1, conj_b=-1)
                    plaqs.append(complex_einsum('bxij, bxjk -> bxik', u12, u34))

        return torch.stack(plaqs, dim=2)
     
    def update_dims(self, dims):
        self.dims = dims   

class PlaqClover(torch.nn.Module):
    """
        A module that computes all clover plaquettes from a link configuration
    """
    def __init__(self, dims, nc):
        super(PlaqClover, self).__init__()
        self.dims = dims
        self.nc = nc

    def get_number_of_channels(self):
        n_dims = len(self.dims)
        n_plaqs = n_dims * (n_dims - 1) // 2
        n_clovers = 4 * n_plaqs
        return n_clovers
    
    def forward(self, u):
        n_dims = len(self.dims)
        plaqs = []

        for mu in range(n_dims):
            for nu in range(0, mu):
                # four plaquettes, all with the same orientation
                # clover 0: U_x,mu,nu
                plaqs.append(plaq(u, mu, nu, +1, +1, self.dims))

                # clover 1: U_x,nu,-mu
                plaqs.append(plaq(u, nu, mu, +1, -1, self.dims))

                # clover 2: U_x,-mu,-nu
                plaqs.append(plaq(u, mu, nu, -1, -1, self.dims))

                # clover 3: U_x,-nu,mu
                plaqs.append(plaq(u, nu, mu, -1, +1, self.dims))
   
        return torch.stack(plaqs, dim=2)

    def update_dims(self, dims):
        self.dims = dims   

class LAH(torch.nn.Module):
    """
        A module that computes the anti-hermitian traceless part of a batch of matrices
    """

    def __init__(self, dims, nc):
        super(LAH, self).__init__()
        self.dims = dims
        self.nc = nc

    def forward(self, a):
        a = torch.view_as_complex(a)
        a_shape = a.shape
        a = 1/(2j) * (a - torch.transpose(a, dim0=-1, dim1=-2).conj())
        unit = torch.eye(self.nc, dtype=a.dtype, device=a.device).view([*[1]*len(a_shape[:-2]), self.nc, self.nc])
        a_trace = torch.einsum('...ii -> ...', a).view([*a_shape[:-2], 1, 1])
        a = (a - a_trace * unit / self.nc)
        return torch.view_as_real(a)

    def update_dims(self, dims):
        self.dims = dims   

class LStout(torch.nn.Module):
    def __init__(self, dims, nc, rho, learnable=False):
        super(LStout, self).__init__()
        self.dims = dims
        self.nc = nc
        self.lah = LAH(dims, nc)
        self.rho = rho
        self.initial_rho = rho
        self.learnable = learnable

        if self.learnable:
            self.weight = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
            self.weight.data = torch.Tensor([rho])
            self.rho = self.weight

    def update_dims(self, dims):
        self.dims = dims   
        self.lah.update_dims(dims)

    def forward(self, u):
        w = torch.zeros_like(u, device=u.device)
        for mu in range(len(self.dims)):
            for nu in range(len(self.dims)):
                if nu != mu:
                    w[:, :, mu, :, :, :] += plaq(u, mu, nu, +1, +1, self.dims)
                    w[:, :, mu, :, :, :] += plaq(u, mu, nu, +1, -1, self.dims)
        w_ah = torch.view_as_complex(self.lah(w))
        #print("w_ah", w_ah.shape)
        w_exp = torch.view_as_real(torch.linalg.matrix_exp(1j * self.rho * w_ah))
        #print("w_exp", w_exp.shape)
        #print("u", u.shape)
        u_new = complex_einsum('bxdij, bxdjk -> bxdik', w_exp, u)
        return u_new

class LActPoly(torch.nn.Module):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization
    """
    def __init__(self, dims, n_in, nc, init_w=1.0, degree_range=[], use_relu=[]):
        super(LActPoly, self).__init__()
        self.dims = dims
        self.n_in = n_in
        self.nc = nc
        self.init_w = init_w
        self.degree_range = degree_range
        self.use_relu = use_relu

        # initialize weights
        w_size = self.n_in
        w_degree_range = degree_range

        if use_relu:
            w_degree_range += 1 # one more item for relu
            self.relu = torch.nn.ReLU()

        variance = 1.0 / (w_size * w_degree_range)
        self.weight = torch.nn.Parameter(data=Tensor(w_size, w_degree_range), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))

        tr_w = einsum('bxwiic -> bxwc', w) # trace
        re_tr_w = tr_w[..., 0] # real part

        w_p = w
        w_powers = [w] # if self.degree_range > 0 else []

        for i in range(self.degree_range - 1):
            w_p = einsum('bxw, bxwijc -> bxwijc', re_tr_w, w_p)
            w_powers.append(w_p)

        if self.use_relu:
            relu = self.relu(re_tr_w)
            w_p = einsum('bxw, bxwijc -> bxwijc', relu, w)
            w_powers.append(w_p)

        # combine terms into a single tensor
        w_powers = torch.stack(w_powers, dim=3)

        w_powers_c = torch.view_as_complex(w_powers)
        weight_c = self.weight.type(w_powers_c.dtype)

        w_c = einsum('ws, bxwsij -> bxwij', weight_c, w_powers_c)
        w = torch.view_as_real(w_c)

        return repack_x(u, w)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

"""
    Various helper functions
"""

@torch.compile
def unpack_x(x, n_dims):
    u = x[:, :, 0:n_dims]
    w = x[:, :, n_dims:]
    return u, w

def repack_x(u, w):
    x = torch.cat((u, w), dim=2)
    return x

def cconj(w):
    w = w.transpose(dim0=3, dim1=4).clone()
    w[:, :, :, :, :, 1] = -w[:, :, :, :, :, 1]
    return w

def cconj2(u):
    u = u.transpose(dim0=3, dim1=4).clone()
    u[:, :, :, :, 1] = -u[:, :, :, :, 1]
    return u

def complex_transport(u, w, mu, o, dims):
    ws = shift(w, mu, -o, dims)
    ua = u.select(dim=2, index=mu)

    if o > 0:
        wt = torch.einsum('bxij, bxwjk, bxlk -> bxwil', ua, ws, ua.conj())
    else:
        # apply shift
        ua = shift(ua, mu, +1, dims)
        wt = torch.einsum('bxji, bxwjk, bxkl -> bxwil', ua.conj(), ws, ua)

    return wt

def transport(u, w, axis, orientation, dims):
    ws = shift(w, axis, -orientation, dims)

    # select links of appropriate axis
    ua = u.select(dim=2, index=axis)

    if orientation > 0:
        # U W U^dagger
        # A (3,3), B (3,3)
        # C = AB -> ij jk 
        # A (n, m, 3, 3), B(n, m, 3, 3)
        #  
        wt = complex_einsum('bxij,bxwjk -> bxwik', ua, ws)
        wt = complex_einsum('bxwij,bxkj -> bxwik', wt, ua, conj_b=-1)
    else:
        # apply shift
        ua = shift(ua, axis, +1, dims)
        wt = complex_einsum('bxji,bxwjk -> bxwik', ua, ws, conj_a=-1)
        wt = complex_einsum('bxwij,bxjk -> bxwik', wt, ua)

    return wt

def shift(a, axis, orientation, dims):
    # tensor layout size: [B, N^D, ..., N_C, N_C, 2]
    # add +1 to axis to skip batch dimension

    # old and new shapes for torch.roll
    o_s = tuple(a.shape)
    n_s = (o_s[0], *dims, *tuple(a.shape[2:]))
    a_shift = roll(a.view(*n_s), orientation, axis + 1).view(*o_s)

    return a_shift

def get_links(u, mu, o_mu, dims):
    # returns links in mu direction and shifted u array at end point
    # must still perform dagger!
    if o_mu > 0:
        u_shift = shift(u, mu, -1, dims)
        links = u[:, :, mu, :, :, :]
    else:
        u_shift = shift(u, mu, +1, dims)
        links = u_shift[:, :, mu, :, :, :]
    return links, u_shift

def plaq(u, mu, nu, o_mu, o_nu, dims):
    l1, us = get_links(u,  mu, +o_mu, dims)
    l2, us = get_links(us, nu, +o_nu, dims)
    l3, us = get_links(us, mu, -o_mu, dims)
    l4, us = get_links(us, nu, -o_nu, dims)
    
    l12 = complex_mm(l1, l2, +o_mu, +o_nu)
    l34 = complex_mm(l3, l4, -o_mu, -o_nu)

    return complex_mm(l12, l34, +1, +1)

def complex_einsum(pattern, a, b, conj_a=1, conj_b=1):
    global use_fast_mm
    if use_fast_mm:
        # faster implementation via complex tensors
        a_c, b_c = torch.view_as_complex(a), torch.view_as_complex(b)
        a_c = a_c.conj() if conj_a == -1 else a_c
        b_c = b_c.conj() if conj_b == -1 else b_c
        c_c = einsum(pattern, a_c, b_c)
        c = torch.view_as_real(c_c)
    else:
        aR, aI = a.select(-1, 0), conj_a * a.select(-1, 1)
        bR, bI = b.select(-1, 0), conj_b * b.select(-1, 1)

        cR = einsum(pattern, aR, bR) - einsum(pattern, aI, bI)
        cI = einsum(pattern, aR, bI) + einsum(pattern, aI, bR)
        c = torch.stack((cR, cI), dim=-1)

    return c

def complex_mm(a, b, conj_a=1, conj_b=1):
    if conj_a == +1 and conj_b == +1:
        return complex_einsum('...ij, ...jk -> ...ik', a, b, conj_a, conj_b)
    elif conj_a == +1 and conj_b == -1:
        return complex_einsum('...ij, ...kj -> ...ik', a, b, conj_a, conj_b)
    elif conj_a == -1 and conj_b == +1:
        return complex_einsum('...ji, ...jk -> ...ik', a, b, conj_a, conj_b)
    elif conj_a == -1 and conj_b == -1:
        return complex_einsum('...ji, ...kj -> ...ik', a, b, conj_a, conj_b)
    else:
        raise Exception(f"Invalid call of complex_mm(a, b, conj_a, conj_b) with a.shape: {a.shape}, b.shape: {b.shape}, conj_a: {conj_a}, conj_b: {conj_b}.")

def update_kernel_size(kernel_size):
    new_kernel_size = kernel_size
    if type(kernel_size) == int:
        new_kernel_size = [kernel_size, kernel_size]
    elif type(kernel_size) == list:
        if len(kernel_size) == 1:
            new_kernel_size = [kernel_size[0], kernel_size[0]]
    return new_kernel_size


class CConv2d(torch.nn.Module):
    """
        A wrapper for Conv2d which correctly implements circular padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(CConv2d, self).__init__()

        # Conv2d without padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

        # padding size
        self.padding = []

        # Check and fix size of kernel_size
        kernel_size = update_kernel_size(kernel_size)

        # for some reason `padding` is in reverse order compared to kernel_size (see also pytorch repository)
        for k in reversed(kernel_size):
            if k % 2 == 0:
                # even kernel
                self.padding.append((k - 1) // 2)
                self.padding.append(k // 2)
            else:
                # odd kernel)
                self.padding.append(k // 2)
                self.padding.append(k // 2)

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, self.padding, mode='circular')
        x_conv = self.conv(x_pad)
        return x_conv
