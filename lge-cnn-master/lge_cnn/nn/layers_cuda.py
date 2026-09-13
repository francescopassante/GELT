import torch
from torch import Tensor
import numpy as np

from lge_cnn.ym.numba_target import myjit, mycudajit, my_parallel_loop
import lge_cnn.ym.lattice as lattice
import lge_cnn.ym.su as su
#from lge_cnn.ym.nn.layers import unpack_x, repack_x

from numba import cuda, jit
import lge_cnn.nn.layers as layers
from torch.autograd.function import once_differentiable

# TODO: So far only tested with SU(3). Use proper ym.su.py interfaces and make usable for SU(2) and SU(3)


###############################################################################################################################################
class LConvBilinCuda2(torch.nn.Module):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization.

        This implementation requires only a fraction of GPU memory, but is slower compared to LConvBilin 
        and only supports first derivatives, and no derivatives with respect to weights.
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True, use_symmetric=True):
        super(LConvBilinCuda2, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.use_symmetric = use_symmetric

        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1))  # TODO: Clean-up and provide functionality through nn.lattice.py

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
        self.weight = torch.nn.Parameter(data=torch.Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
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

        # construct transport w list:
        # each list contains instruction entries of the following format:
        #   [0]: axis
        #   [1]: steps
        #   [2]: orientation
        self._transport_list = []

        # add local term
        self._transport_list.append((0, 0, 0))

        # gather all terms along lattice axes up to kernel_size
        for axis in range(len(self.dims)):
            for i, o in zip([0, 1], [-1, +1]):
                kernel_size = abs(self.kernel_range[axis][i])
                for d in range(kernel_size):
                    steps = self.dilation * (d + 1)
                    self._transport_list.append((axis, steps, o))

        self._transport_list = tuple(self._transport_list)
        # TODO: test whether it is better to use a constant tuple or a GPU array? 

    def forward(self, x):
        # x contains (u, w)
        u, w = layers.unpack_x(x, len(self.dims))

        w2 = LConvBilinCudaFunc2.apply(u, w, self.weight, self.n_out, self._transport_list, self.dims, self.acc, self.use_unit_elements)

        x_out = layers.repack_x(u, w2)
        return x_out

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims
        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1)) 
    
class LConvBilinCudaFunc2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, w, weight, n_out, transport_list, dims, acc, use_unit_elements):

        # Save the input for backward
        ctx.save_for_backward(u, w, weight)
        ctx.parameters = transport_list, dims, acc, n_out, use_unit_elements

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())
        w = convert_to_su_matrix(w.detach())

        weight = weight.detach()

        # reserve space for w_out
        channel_size = n_out
        w_out_shape_cuda = (*w.shape[0:2], channel_size, *w.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        w_out = torch.zeros(w_out_shape_cuda, dtype = w.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        w_out_len = weight.shape[0]

        w_out_block_size = 1
        w_out_block_max = (w_out_len + w_out_block_size - 1) // w_out_block_size

        idx_max = n_batch * n_sites * w_out_block_max

        my_parallel_loop(lbilinconv_kernel_2, idx_max, u, w, weight, w_out, transport_list, dims, acc, use_unit_elements, w_out_block_size, w_out_block_max)

        w_out = convert_from_su_matrix(w_out)
        return w_out
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        
        # Retrieve the input
        u, w, weight = ctx.saved_tensors
        transport_list, dims, acc, n_out, use_unit_elements = ctx.parameters

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())
        w = convert_to_su_matrix(w.detach())
        grad_out = convert_to_su_matrix(grad_out.detach())

        weight = weight.detach()

        grad_u = torch.zeros(u.shape, dtype = w.dtype, device="cuda")
        grad_w = torch.zeros(w.shape, dtype = w.dtype, device="cuda")
        # grad_weight = torch.zeros(weight.shape, dtype = weight.dtype, device="cuda")
        grad_weight = None  # Maybe later

        n_batch = u.shape[0]
        n_sites = u.shape[1]

        axis_len = len(dims_d)
        # axis_block_size = axis_len
        axis_block_size = 1
        axis_block_max = (axis_len + axis_block_size - 1) // axis_block_size

        # Construct offset of transport_list entries per axis.
        # It is assumed that transport_list items of same axis follow consecutively
        transport_list_entries = []
        previous_axis = -1
        for i, (axis, steps, orientation) in enumerate(transport_list):
            if steps > 0:
                if axis > previous_axis:
                    transport_list_entries.append(i)
                    previous_axis = axis
                elif axis < previous_axis:
                    raise ValueError("Axis not in consecutive order in transport-list", axis, previous_axis)
        transport_list_entries.append(len(transport_list)) # End of list
        transport_list_entries = tuple(transport_list_entries)

        idx_max = n_batch * n_sites * axis_block_max

        # Calculate grad_u
        my_parallel_loop(lbilinconv_backward_u_kernel_2, idx_max, u, w, weight, grad_out, grad_u, transport_list, dims, acc, use_unit_elements, axis_block_size, axis_block_max, transport_list_entries)


        w_len = w.shape[2]
        # w_block_size = w_len
        w_block_size = 1
        w_block_max = (w_len + w_block_size - 1) // w_block_size

        idx_max = n_batch * n_sites * w_block_max
        # Calculate grad_w
        my_parallel_loop(lbilinconv_backward_w_kernel_2, idx_max, u, w, weight, grad_out, grad_w, transport_list, dims, acc, use_unit_elements, w_block_size, w_block_max)

        grad_u = convert_from_su_matrix(grad_u)
        grad_w = convert_from_su_matrix(grad_w)

        # Return elements correspond to arguments of forward function:
        # def forward(ctx, u, w, weight, n_out, transport_list, dims, acc, use_unit_elements):
        return grad_u, grad_w, grad_weight, None, None, None, None, None


# Obtain a single contribution of lconvunit_kernel
@myjit
def get_w_transported_2(b, x, w_index, u, w, transport_list, dims, acc):
    """
    x: position index
    b: batch index
    w_index: w index (called k in the outer loop)
    """
    w_len = w.shape[2]

    w_i = w_index % w_len # w-index
    t = w_index // w_len # transport_list index
    
    # transport_list format:
    #   [0]: axis
    #   [1]: steps
    #   [2]: orientation
    axis, steps, orientation = transport_list[t]

    # perform parallel transport operation (LConv)
    
    # construct parallel transport for all steps
    t_x = x # transported position x
    links = su.unit() # product of all links up to t_x
    for j in range(steps): # TODO: if necessary, another optimization could be possible here: 
                            #       continue to construct links from last location instead of always starting from x
                            #       (but maybe this is not a bottleneck, especially for small kernel sizes)
        
        u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
        links = su.mul(links, u0)
        t_x = lattice.shift(t_x, axis, orientation, dims, acc)
    
    # Parallel transport w field:
    w_t_x = su.load(w[b, t_x, w_i]) # w at position t_x
    t_w = su.mul(su.mul(links, w_t_x), su.dagger(links)) # transported w

    return t_w

# Minimize write output
@myjit
def lbilinconv_kernel_2(idx, u, w, weight, w_out, transport_list, dims, acc, use_unit_elements, w_out_block_size, w_out_block_max):
    n_batch = w.shape[0]
    n_sites = w.shape[1]
    w1_len = w.shape[2]
    w2_len = w.shape[2] * len(transport_list)
    w_out_len = w_out.shape[2]
    
    mm = idx // 1
    x = mm // w_out_block_max
    b = x // n_sites

    mm = mm % w_out_block_max
    x = x % n_sites 
    b = b % n_batch
    
    # Multiply matrices with the correct weights and add to result
    # for m in range(w_out_len):
    for m in range(w_out_block_size * mm, min(w_out_block_size * mm + w_out_block_size, w_out_len)):

        out_add = su.zero()

        for k in range(w2_len):
            w2m = get_w_transported_2(b, x, k, u, w, transport_list, dims, acc)
        
            # perform bilinear layer operation (LBilin)
            for l in range(w1_len):
                w1m = su.load(w[b, x, l])

                # w and t_w undaggered
                out_add = add_bilin_values_2(m, l, k, False, False, w1m, w2m, weight, out_add)

                # daggered w
                out_add = add_bilin_values_2(m, w1_len + l, k, True, False, w1m, w2m, weight, out_add)

                # daggered t_w
                out_add = add_bilin_values_2(m, l, w2_len + k, False, True, w1m, w2m, weight, out_add)

                # daggered w and t_w
                out_add = add_bilin_values_2(m, w1_len + l, w2_len + k, True, True, w1m, w2m, weight, out_add)

                if use_unit_elements:
                    if l == 0:
                        # 1 * t_w
                        out_add = add_bilin_values_2(m, 2 * w1_len, k, False, False, su.unit(), w2m, weight, out_add)

                        # 1 * dagger(t_w)
                        out_add = add_bilin_values_2(m, 2 * w1_len, w2_len + k, False, True, su.unit(), w2m, weight, out_add)
                    
                    if k == 0:
                        # w * 1
                        out_add = add_bilin_values_2(m, l, 2 * w2_len, False, False, w1m, su.unit(), weight, out_add)

                        # dagger(w) * 1
                        out_add = add_bilin_values_2(m, w1_len + l, 2 * w2_len, True, False, w1m, su.unit(), weight, out_add)

                    if l == 0 and k == 0:
                        # 1 * 1
                        out_add = add_bilin_values_2(m, 2 * w1_len, 2 * w2_len, False, False, su.unit(), su.unit(), weight, out_add)


        su.atomic_add(w_out[b, x, m], out_add)

@myjit
def add_bilin_values_2(m, l, k, dagger_w1, dagger_w2, w1m, w2m, weight, out_add):

    factor = weight[m, l, k]
    
    if dagger_w1:
        w1m = su.dagger(w1m)

    if dagger_w2:
        w2m = su.dagger(w2m)
    
    out = su.mul_s(su.mul(w1m, w2m), factor)
    out_add = su.add(out_add, out)

    return out_add

@myjit
def lbilinconv_backward_u_kernel_2(idx, u, w, weight, grad_out, grad_u, transport_list, dims, acc, use_unit_elements, axis_block_size, axis_block_max, transport_list_entries):
    n_batch = w.shape[0]
    n_sites = w.shape[1]
    w_len = w.shape[2]
    w1_len = w_len
    w2_len = w_len * len(transport_list)
    w_out_len = weight.shape[0]
    axis_len = u.shape[2]
    
    axis_thread = idx // 1
    x = axis_thread // axis_block_max
    b = x // n_sites

    axis_thread = axis_thread % axis_block_max
    x = x % n_sites 
    b = b % n_batch

    #for axis_out in range(axis_len):
    for axis_out in range(axis_block_size * axis_thread, min(axis_block_size * axis_thread + axis_block_size, axis_len)):
        grad_u_add = su.zero()
        
        #for t in range(len(transport_list)):
        for t in range(transport_list_entries[axis_out], transport_list_entries[axis_out + 1]):
            axis, steps, orientation = transport_list[t]

            #    if axis == axis_out and steps > 0:  ############### This should be fulfilled by restricting t

            bt_x = x # backward transported position
            if orientation < 0:
                # Start shifted in positive direction for derivative of daggered quantity
                bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

            # a link U appears once along the parallel transported path
            for i in range(steps): # additional loop for derivative
                
                # a link U appears once along the parallel transported path

                # construct parallel transport for all steps
                t_x = bt_x # x # transported position x
                links_A = su.unit() # product of all links before position i
                link_g = su.unit() # link at position i (where also g is located)
                links_B = su.unit() # product of all links after position i up to position t_x
                for j in range(steps): 
                    
                    u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
                    if j < i:
                        links_A = su.mul(links_A, u0)
                    elif j > i:
                        links_B = su.mul(links_B, u0)
                    else: # j == i
                        link_g = u0
                    t_x = lattice.shift(t_x, axis, orientation, dims, acc)
                
                links = su.mul(su.mul(links_A, link_g), links_B)
                

                for k_w in range(w_len):
        
                    w2m = su.load(w[b, t_x, k_w])

                    k = t * w_len + k_w # Index for w2 loop

                    # perform bilinear layer operation (LBilin)
                    for l in range(w1_len):
                        w1m = su.load(w[b, bt_x, l])
                        
                        # Multiply matrices with the correct weights and add to result
                        # # w and t_w undaggered
                        grad_u_add = add_lconvbilin_grad_u_2(b, l, k, False, False, False, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                        # # daggered w
                        grad_u_add = add_lconvbilin_grad_u_2(b, w1_len + l, k, True, False, False, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                        # # daggered t_w
                        grad_u_add = add_lconvbilin_grad_u_2(b, l, w2_len + k, False, True, False, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                        # # daggered w and t_w
                        grad_u_add = add_lconvbilin_grad_u_2(b, w1_len + l, w2_len + k, True, True, False, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                        # If there is a unit element, then the links cancel, so there is no net derivative with respect to the links.

                        if use_unit_elements:
                            if l == 0:
                                # 1 * t_w
                                grad_u_add = add_lconvbilin_grad_u_2(b, 2 * w1_len, k, False, False, True, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                                # 1 * dagger(t_w)
                                grad_u_add = add_lconvbilin_grad_u_2(b, 2 * w1_len, w2_len + k, False, True, True, False, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)
                            
                            if k == 0:
                                # w * 1
                                grad_u_add = add_lconvbilin_grad_u_2(b, l, 2 * w2_len, False, False, False, True, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                                # dagger(w) * 1
                                grad_u_add = add_lconvbilin_grad_u_2(b, w1_len + l, 2 * w2_len, True, False, False, True, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)

                            if l == 0 and k == 0:
                                # 1 * 1
                                grad_u_add = add_lconvbilin_grad_u_2(b, 2 * w1_len, 2 * w2_len, False, False, True, True, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x)


                # Adjust start position in opposite direction for next step in i loop
                bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

        su.atomic_add(grad_u[b, x, axis_out], grad_u_add)

# Calculate gradient with respect to u
@myjit
def add_lconvbilin_grad_u_2(b, l_weight, k_weight, dagger_w1, dagger_w2, unit_w1, unit_w2, orientation, links, links_A, links_B, w1m, w2m, weight, grad_out, grad_u_add, bt_x):
    w_out_len = weight.shape[0]

    # Multiply W_1 * U * U * ... * U * W_2 * U^\dagger * U^\dagger * ... * U^\dagger
    if unit_w1:
        w1m_ = su.unit()
    else:
        w1m_ = w1m # su.load(w[b, bt_x, l])
        if dagger_w1:
            w1m_ = su.dagger(w1m)

    if unit_w2:
        w2m_ = su.unit()
    else:
        w2m_ = w2m # su.load(w[b, t_x, w2_i])
        if dagger_w2:
            w2m_ = su.dagger(w2m)

    # Multiply: w1 * links_A * [U-derivative] * links_B * w2 * links^\dagger

    # Derivative along forward path: (w1 * links_A)^\dagger * grad_out * (links_B * w2 * links^\dagger)^\dagger
    #                                = A * grad_out * B
    A = su.dagger(su.mul(w1m_, links_A))
    B = su.dagger(su.mul(su.mul(links_B, w2m_), su.dagger(links)))
    
    # Along backward path:
    # Multiply: w1 * links * w2 * links_B^\dagger * [U-dagger-derivative] * links_A^\dagger
    
    # Derivative along backward path: (w1 * links * w2 * links_B^dagger)^\dagger * grad_out * (links_A^\dagger)^\dagger
    #                                 = C * grad_out * D
    C = su.dagger(su.mul(su.mul(w1m_, links), su.mul(w2m_, su.dagger(links_B))))
    D = su.dagger(su.dagger(links_A)) # compiler simplifies this hopefully
    
    res_u = su.zero()
    grad_out_sum = su.zero()

    for m in range(w_out_len):
        factor = weight[m, l_weight, k_weight]

        g_bt_x = su.load(grad_out[b, bt_x, m])

        grad_out_sum = su.add(grad_out_sum, su.mul_s(g_bt_x, factor))

    # Derivative along forward path: (w1 * links_A)^\dagger * grad_out * (links_B * w2 * links^\dagger)^\dagger
    #                                = A * grad_out * B
    du = su.mul(su.mul(A, grad_out_sum), B)
    if orientation < 0:
        # calculate daggered quantity
        du = su.dagger(du)
    res_u = su.add(res_u, du)

    # Derivative along backward path: (w1 * links * w2 * links_B^dagger)^\dagger * grad_out * (links_A^\dagger)^\dagger
    #                                 = C * grad_out * D
    du = su.mul(su.mul(C, grad_out_sum), D)
    if orientation > 0: # This time calculate the daggered quantity for positive orientation
        # calculate daggered quantity
        du = su.dagger(du)
    res_u = su.add(res_u, du)
            
    grad_u_add = su.add(grad_u_add, res_u)
    
    return grad_u_add

@myjit
def lbilinconv_backward_w_kernel_2(idx, u, w, weight, grad_out, grad_w, transport_list, dims, acc, use_unit_elements, w_block_size, w_block_max):
    n_batch = w.shape[0]
    n_sites = w.shape[1]
    w_len = w.shape[2]
    w1_len = w_len
    w2_len = w_len * len(transport_list)
    w_out_len = weight.shape[0]
    
    w_thread = idx // 1
    x = w_thread // w_block_max
    b = x // n_sites

    w_thread = w_thread % w_block_max
    x = x % n_sites 
    b = b % n_batch
    
    grad_w1_add = su.zero()
    grad_w2_add = su.zero()

    # perform bilinear layer operation (LBilin)
    #for l in range(w1_len):
    for l in range(w_block_size * w_thread, min(w_block_size * w_thread + w_block_size, w_len)):
    
        #for k in range(w2_len):
        for k_w in range(w_len):
                
            for t in range(len(transport_list)):
                k = t * w_len + k_w # Index for w2 loop

                # Swapped coordinates
                kk = k_w
                ll = t * w_len + l
                
                # Multiply matrices with the correct weights and add to result
                # # w and t_w undaggered
                grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, l, k, False, False, False, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, kk, ll, False, False, False, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                # # daggered w
                grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, w1_len + l, k, True, False, False, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, w1_len + kk, ll, True, False, False, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                # # daggered t_w
                grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, l, w2_len + k, False, True, False, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, kk, w2_len + ll, False, True, False, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                # # daggered w and t_w
                grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, w1_len + l, w2_len + k, True, True, False, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, w1_len + kk, w2_len + ll, True, True, False, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                # If there is a unit element, then the links cancel, so there is no net derivative with respect to the links.

                if use_unit_elements:
                    if ll == 0: # implies l == 0
                        # 1 * t_w
                        grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, 2 * w1_len, k, False, False, True, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                        grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, 2 * w1_len, ll, False, False, True, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                        # 1 * dagger(t_w)
                        grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, 2 * w1_len, w2_len + k, False, True, True, False, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                        grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, 2 * w1_len, w2_len + ll, False, True, True, False, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)
                    
                    if k == 0: # implies kk = 0
                        # w * 1
                        grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, l, 2 * w2_len, False, False, False, True, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                        grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, kk, 2 * w2_len, False, False, False, True, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                        # dagger(w) * 1
                        grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, w1_len + l, 2 * w2_len, True, False, False, True, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                        grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, w1_len + kk, 2 * w2_len, True, False, False, True, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)

                    if ll == 0 and k == 0:
                        # 1 * 1
                        grad_w1_add = add_lconvbilin_grad_w1_2(b, x, l, k, 2 * w1_len, 2 * w2_len, False, False, True, True, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc)
                        grad_w2_add = add_lconvbilin_grad_w2_2(b, x, kk, ll, 2 * w1_len, 2 * w2_len, False, False, True, True, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc)


        #    su.atomic_add(grad_w[b, x, k], grad_w2_add) # actually, the original index of w2 is k. By swapping indices, we can write them both in the outermost loop.
        # su.atomic_add(grad_w[b, x, l], grad_w2_add) # w2i==k # swap meaning of l and k
        # grad_w2_add = su.zero()
        
        # su.atomic_add(grad_w[b, x, l], grad_w1_add)
        # grad_w1_add = su.zero()

        su.atomic_add(grad_w[b, x, l], su.add(grad_w1_add, grad_w2_add)) # we can add w1 and w2, because w2 has swapped coordinates
        grad_w1_add = su.zero()
        grad_w2_add = su.zero()

@myjit
def add_lconvbilin_grad_w1_2(b, x, l, k, l_weight, k_weight, dagger_w1, dagger_w2, unit_w1, unit_w2, u, w, weight, grad_out, grad_w1_add, transport_list, dims, acc):
    w_out_len = weight.shape[0]

    # Derivative with respect to U:
    # Gradients can appear on the forward path or on the backward path
    
    w_len = w.shape[2]
    w2_i = k % w_len # w2-index
    t = k // w_len # transport_list index

    # transport_list format:
    #   [0]: axis
    #   [1]: steps
    #   [2]: orientation
    axis, steps, orientation = transport_list[t]

    res_w1 = su.zero()

    # Calculate derivatives with respect to u

    bt_x = x # backward transported position
    if orientation < 0:
        # Start shifted in positive direction for derivative of daggered quantity
        bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

    # Now calculate derivatives with respect to first w
    # Multiply: [W-derivative] * links * w2 * links^\dagger
    # Everything is shifted such that W-derivative is at position x.
    
    if not unit_w1:
        g_x_sum = su.zero()
        for m in range(w_out_len):
            g_x = su.load(grad_out[b, x, m])
            factor = weight[m, l_weight, k_weight]
            g_x_sum = su.add(g_x_sum, su.mul_s(g_x, factor))

        # Construct links from starting position x
        t_x = x
        links = su.unit()
        for i in range(steps):         
            u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
            links = su.mul(links, u0)
            t_x = lattice.shift(t_x, axis, orientation, dims, acc)

        if unit_w2:
            w2m = su.unit()
        else:
            w2m = su.load(w[b, t_x, w2_i])
            if dagger_w2:
                w2m = su.dagger(w2m)

        # Derivative: grad_out * (links * w2 * links^\dagger)^\dagger
        B = su.dagger(su.mul(su.mul(links, w2m), su.dagger(links)))
        dw1 = su.mul(g_x_sum, B)
        if dagger_w1:
            # calculate daggered quantity
            dw1 = su.dagger(dw1)
        res_w1 = su.add(res_w1, dw1)

    grad_w1_add = su.add(grad_w1_add, res_w1)    
    return grad_w1_add

@myjit
def add_lconvbilin_grad_w2_2(b, x, l, k, l_weight, k_weight, dagger_w1, dagger_w2, unit_w1, unit_w2, u, w, weight, grad_out, grad_w2_add, transport_list, dims, acc):
    w_out_len = weight.shape[0]

    # Derivative with respect to U:
    # Gradients can appear on the forward path or on the backward path
    
    w_len = w.shape[2]
    w2_i = k % w_len # w2-index
    t = k // w_len # transport_list index

    # transport_list format:
    #   [0]: axis
    #   [1]: steps
    #   [2]: orientation
    axis, steps, orientation = transport_list[t]

    res_w2 = su.zero()

    # Calculate derivatives with respect to u

    bt_x = x # backward transported position
    if orientation < 0:
        # Start shifted in positive direction for derivative of daggered quantity
        bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

    # # Now calculate derivatives with respect to second w
    # # Multiply: w1 * links * [W-derivative] * links^\dagger
    # # Everything is shifted such that W-derivative is at position x.

    if not unit_w2:

        # Construct links from t_x up to position x:
        # Do this by constructing links in opposite direction,
        # and then dagger the result
        t_x = x
        links = su.unit()
        for i in range(steps): # additional loop for derivative          
            u0 = lattice.get_link(u[b], t_x, axis, -orientation, dims, acc) # negative orientation
            links = su.mul(links, u0)
            t_x = lattice.shift(t_x, axis, -orientation, dims, acc) # negative orientation

        links = su.dagger(links) # convert backlink from x to t_x into forward link from t_x to x

        if unit_w1:
            w1m = su.unit()
        else:
            w1m = su.load(w[b, t_x, l])
            if dagger_w1:
                w1m = su.dagger(w1m)

        g_t_x_sum = su.zero()
        for m in range(w_out_len):
            g_t_x = su.load(grad_out[b, t_x, m]) # pick grad_out from shifted position
            factor = weight[m, l_weight, k_weight]
            g_t_x_sum = su.add(g_t_x_sum, su.mul_s(g_t_x, factor))

        # Derivative: (w1 * links)^\dagger * grad_out * (links^\dagger)^\dagger
        A = su.dagger(su.mul(w1m, links))
        B = su.dagger(su.dagger(links))
        dw2 = su.mul(su.mul(A, g_t_x_sum), B)
        if dagger_w2:
            # calculate daggered quantity
            dw2 = su.dagger(dw2)
        res_w2 = su.add(res_w2, dw2)

    grad_w2_add = su.add(grad_w2_add, res_w2)
    return grad_w2_add



###############################################################################################################################################

class LConvBilinCuda(torch.nn.Module):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True, use_symmetric=True):
        super(LConvBilinCuda, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.use_symmetric = use_symmetric

        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1))  # TODO: Clean-up and provide functionality through nn.lattice.py

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
        self.weight = torch.nn.Parameter(data=torch.Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
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

        # construct transport w list:
        # each list contains instruction entries of the following format:
        #   [0]: axis
        #   [1]: steps
        #   [2]: orientation
        self._transport_list = []

        # add local term
        self._transport_list.append((0, 0, 0))

        # gather all terms along lattice axes up to kernel_size
        for axis in range(len(self.dims)):
            for i, o in zip([0, 1], [-1, +1]):
                kernel_size = abs(self.kernel_range[axis][i])
                for d in range(kernel_size):
                    steps = self.dilation * (d + 1)
                    self._transport_list.append((axis, steps, o))

        self._transport_list = tuple(self._transport_list)
        # TODO: test whether it is better to use a constant tuple or a GPU array? 

    def forward(self, x):
        # x contains (u, w)
        u, w = layers.unpack_x(x, len(self.dims))

        w2 = LConvUnitCudaFunc.apply(u, w, self._transport_list, self.dims, self.acc)
        # f = torch.sum(torch.isnan(w2.flatten())).item()
        # if f > 0:
        #     print("nan's detected in w2", f)
        #     print("dims", self.dims)
        #     print("w2", w2.shape)
        
        w3 = LBilinCudaFunc.apply(w, w2, self.weight, self.n_out, self.use_unit_elements)
        # f = torch.sum(torch.isnan(w3.flatten())).item()
        # if f > 0:
        #     print("nan's detected in w3", f)
        #     print("dims", self.dims)
        #     print("w3", w3.shape)


        x_out = layers.repack_x(u, w3)
        return x_out

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims
        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1)) 

class LConvUnitCudaFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, w, transport_list, dims, acc):
        # Save the input for backward
        ctx.save_for_backward(u, w)
        ctx.parameters = transport_list, dims, acc

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())
        w = convert_to_su_matrix(w.detach())

        # reserve space for transported w_out
        w_channels = w.shape[2]
        w_out_channels = len(transport_list) * w_channels
        w_out_shape_cuda = (*w.shape[0:2], w_out_channels, *w.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        w_out = torch.zeros(w_out_shape_cuda, dtype = w.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites * len(transport_list)

        my_parallel_loop(lconvunit_kernel, idx_max, u, w, w_out, transport_list, dims_d, acc)

        w_out = convert_from_su_matrix(w_out)
        return w_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        # Retrieve the input
        u, w = ctx.saved_tensors
        transport_list, dims, acc = ctx.parameters

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())
        w = convert_to_su_matrix(w.detach())
        grad_out = convert_to_su_matrix(grad_out.detach())

        grad_u = torch.zeros(u.shape, dtype = w.dtype, device="cuda")
        grad_w = torch.zeros(w.shape, dtype = w.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites * len(transport_list)

        my_parallel_loop(lconvunit_backward_kernel, idx_max, u, w, grad_out, grad_u, grad_w, transport_list, dims_d, acc)

        grad_u = convert_from_su_matrix(grad_u)
        grad_w = convert_from_su_matrix(grad_w)

        return grad_u, grad_w, None, None, None

@myjit
def lconvunit_kernel(idx, u, w, w_out, transport_list, dims, acc):
    n_batch = u.shape[0]
    n_sites = u.shape[1]
    w_len = w.shape[2]

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    t = b // n_batch # transport list index
    b = b % n_batch

    k_base = w_len * t # sum of number of all transported w fields so far

    # transport_list format:
    #   [0]: axis
    #   [1]: steps
    #   [2]: orientation
    axis, steps, orientation = transport_list[t]

    # perform parallel transport operation (LConv)
    
    # construct parallel transport for all steps
    t_x = x # transported position x
    links = su.unit() # product of all links up to t_x
    for j in range(steps): # TODO: if necessary, another optimization could be possible here: 
                            #       continue to construct links from last location instead of always starting from x
                            #       (but maybe this is not a bottleneck, especially for small kernel sizes)
        
        u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
        links = su.mul(links, u0)
        t_x = lattice.shift(t_x, axis, orientation, dims, acc)
    
    # Obtain all parallel transported w fields:
    for k in range(w_len):
        w_t_x = su.load(w[b, t_x, k]) # w at position t_x
        t_w = su.mul(su.mul(links, w_t_x), su.dagger(links)) # transported w
        su.store(w_out[b, x, k_base + k], t_w)


@myjit
def lconvunit_backward_kernel(idx, u, w, grad_out, grad_u, grad_w, transport_list, dims, acc):
    n_batch = u.shape[0]
    n_sites = u.shape[1]
    w_len = w.shape[2]
    n_dims = len(dims)

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    t = b // n_batch # transport list index
    b = b % n_batch

    k_base = w_len * t # sum of number of all transported w fields so far

    # transport_list format:
    #   [0]: axis
    #   [1]: steps
    #   [2]: orientation
    axis, steps, orientation = transport_list[t]

    # perform parallel transport operation (LConv)

    bt_x = x # backward transported position
    if orientation < 0:
        # Start shifted in positive direction for derivative of daggered quantity
        bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

    # a link U appears once along the parallel transported path
    for i in range(steps): # additional loop for derivative
    
        # construct parallel transport for all steps
        t_x = bt_x # x # transported position x
        links_A = su.unit() # product of all links before position i
        link_g = su.unit() # link at position i (where also g is located)
        links_B = su.unit() # product of all links after position i up to position t_x
        for j in range(steps): # TODO: if necessary, another optimization could be possible here: 
                                #       continue to construct links from last location instead of always starting from x
                                #       (but maybe this is not a bottleneck, especially for small kernel sizes)
            
            u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
            if j < i:
                links_A = su.mul(links_A, u0)
            elif j > i:
                links_B = su.mul(links_B, u0)
            else: # j == i
                link_g = u0
            t_x = lattice.shift(t_x, axis, orientation, dims, acc)
        
        links = su.mul(su.mul(links_A, link_g), links_B)
        
        # Obtain all parallel transported w fields:
        for k in range(w_len):
            res = su.zero()

            w_t_x = su.load(w[b, t_x, k]) # w at position t_x
            g_bt_x = su.load(grad_out[b, bt_x, k_base + k])

            # Derivative along forward path: (links_A)^\dagger * grad_out * (links B * w * links^\dagger)^\dagger
            A = su.dagger(links_A)
            B = su.dagger(su.mul(su.mul(links_B, w_t_x), su.dagger(links)))
            du = su.mul(su.mul(A, g_bt_x), B)
            if orientation < 0:
                # calculate daggered quantity
                du = su.dagger(du)
            res = su.add(res, du)
            
            # Derivative along backward path: (links * w * links_B^dagger)^\dagger * grad_out * (links_A^\dagger)^\dagger
            A = su.dagger(su.mul(su.mul(links, w_t_x), su.dagger(links_B)))
            B = su.dagger(su.dagger(links_A)) # compiler simplifies this hopefully
            du = su.mul(su.mul(A, g_bt_x), B)
            if orientation > 0: # This time calculate the daggered quantity for positive orientation
                # calculate daggered quantity
                du = su.dagger(du)
            res = su.add(res, du)

            su.atomic_add(grad_u[b, x, axis], res)

        bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

    # Calculate gradients with respect to w

    bt_x = x # backward transported position
    for i in range(steps): # additional loop for derivative
        bt_x = lattice.shift(bt_x, axis, -orientation, dims, acc)

    # construct parallel transport for all steps
    t_x = bt_x # transported position x
    links = su.unit() # product of all links up to t_x
    for j in range(steps):
        
        u0 = lattice.get_link(u[b], t_x, axis, orientation, dims, acc)
        links = su.mul(links, u0)
        t_x = lattice.shift(t_x, axis, orientation, dims, acc)
    
    # Obtain all parallel transported w fields:
    for k in range(w_len):
        g_bt_x = su.load(grad_out[b, bt_x, k_base + k])
        # Derivative with respect to w: links^\dagger * grad_out * (links^\dagger)^\dagger
        A = su.dagger(links)
        B = su.dagger(su.dagger(links)) # compiler simplifies this hopefully
        dw = su.mul(su.mul(A, g_bt_x), B)
        su.atomic_add(grad_w[b, x, k], dw)


class LBilinCudaFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w1, w2, weight, n_out, use_unit_elements):
        
        # Save the input for backward
        ctx.save_for_backward(w1, w2, weight)
        ctx.parameters = n_out, use_unit_elements

        w1 = convert_to_su_matrix(w1.detach())
        w2 = convert_to_su_matrix(w2.detach())
        weight = weight.detach()

        # reserve space for w_out
        channel_size = n_out
        w_out_shape_cuda = (*w1.shape[0:2], channel_size, *w1.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        w_out = torch.zeros(w_out_shape_cuda, dtype = w1.dtype, device="cuda")

        n_batch = w1.shape[0]
        n_sites = w1.shape[1]
        w2_len = w2.shape[2]
        idx_max = n_batch * n_sites * w2_len

        # print('w1', w1.shape, 'w2', w2.shape, 'w', weight.shape, 'wo',w_out.shape)

        my_parallel_loop(lbilin_kernel, idx_max, w1, w2, weight, w_out, use_unit_elements)

        w_out = convert_from_su_matrix(w_out)
        return w_out
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):

        # Retrieve the input
        w1, w2, weight = ctx.saved_tensors
        n_out, use_unit_elements = ctx.parameters

        w1 = convert_to_su_matrix(w1.detach())
        w2 = convert_to_su_matrix(w2.detach())
        grad_out = convert_to_su_matrix(grad_out.detach())

        weight = weight.detach()
        
        grad_w1 = torch.zeros(w1.shape, dtype = w1.dtype, device="cuda")
        grad_w2 = torch.zeros(w2.shape, dtype = w2.dtype, device="cuda")
        grad_weight = torch.zeros(weight.shape, dtype = weight.dtype, device="cuda")

        n_batch = w1.shape[0]
        n_sites = w1.shape[1]
        w2_len = w2.shape[2]
        idx_max = n_batch * n_sites * w2_len

        my_parallel_loop(lbilin_backward_kernel, idx_max, w1, w2, weight, grad_out, grad_w1, grad_w2, grad_weight, use_unit_elements)

        grad_w1 = convert_from_su_matrix(grad_w1)
        grad_w2 = convert_from_su_matrix(grad_w2)

        return grad_w1, grad_w2, grad_weight, None, None

@myjit
def lbilin_kernel(idx, w1, w2, weight, w_out, use_unit_elements):
    n_batch = w1.shape[0]
    n_sites = w1.shape[1]
    w1_len = w1.shape[2]
    w2_len = w2.shape[2]
    w_out_len = w_out.shape[2]

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    kk = b // n_batch # loop index for w2 channels
    b = b % n_batch

    # for k in range(w2_len):
    for k in range(kk, kk+1): # equivalent to k = kk without the loop
        w2m = su.load(w2[b, x, k])
    
        # perform bilinear layer operation (LBilin)
        for l in range(w1_len):
            w1m = su.load(w1[b, x, l])

            # Multiply matrices with the correct weights and add to result
            for m in range(w_out_len):
                out_add = su.zero()

                # w and t_w undaggered
                out_add = add_bilin_values(m, l, k, False, False, w1m, w2m, weight, out_add)

                # daggered w
                out_add = add_bilin_values(m, w1_len + l, k, True, False, w1m, w2m, weight, out_add)

                # daggered t_w
                out_add = add_bilin_values(m, l, w2_len + k, False, True, w1m, w2m, weight, out_add)

                # daggered w and t_w
                out_add = add_bilin_values(m, w1_len + l, w2_len + k, True, True, w1m, w2m, weight, out_add)

                if use_unit_elements:
                    if l == 0:
                        # 1 * t_w
                        out_add = add_bilin_values(m, 2 * w1_len, k, False, False, w1m, w2m, weight, out_add) # FIXME: w1m -> su.unit()  ((see lbilin_backward_kernel))

                        # 1 * dagger(t_w)
                        out_add = add_bilin_values(m, 2 * w1_len, w2_len + k, False, True, w1m, w2m, weight, out_add) # FIXME: w1m -> su.unit()  ((see lbilin_backward_kernel))
                    
                    if k == 0:
                        # w * 1
                        out_add = add_bilin_values(m, l, 2 * w2_len, False, False, w1m, w2m, weight, out_add) # FIXME: w2m -> su.unit()  ((see lbilin_backward_kernel))

                        # dagger(w) * 1
                        out_add = add_bilin_values(m, w1_len + l, 2 * w2_len, True, False, w1m, w2m, weight, out_add) # FIXME: w2m -> su.unit()  ((see lbilin_backward_kernel))

                    if l == 0 and k == 0:
                        # 1 * 1
                        out_add = add_bilin_values(m, 2 * w1_len, 2 * w2_len, False, False, w1m, w2m, weight, out_add) # FIXME: w1m, w2m -> su.unit()  ((see lbilin_backward_kernel))

                su.atomic_add(w_out[b, x, m], out_add)

@jit(nopython=True)
def add_bilin_values(m, l, k, dagger_w1, dagger_w2, w1m, w2m, weight, out_add):

    factor = weight[m, l, k]
    
    if dagger_w1:
        w1m = su.dagger(w1m)

    if dagger_w2:
        w2m = su.dagger(w2m)

    out = su.mul_s(su.mul(w1m, w2m), factor)
    out_add = su.add(out_add, out)
    return out_add

@myjit
def lbilin_backward_kernel(idx, w1, w2, weight, grad_out, grad_w1, grad_w2, grad_weight, use_unit_elements):
    n_batch = w1.shape[0]
    n_sites = w1.shape[1]
    w1_len = w1.shape[2]
    w2_len = w2.shape[2]
    w_out_len = grad_out.shape[2]

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    kk = b // n_batch # loop index for w2 channels
    b = b % n_batch

    # for k in range(w2_len):
    for k in range(kk, kk+1): # equivalent to k = kk without the loop
        w2m = su.load(w2[b, x, k])
        grad_w2_add = su.zero()
    
        # perform bilinear layer operation (LBilin)
        for l in range(w1_len):
            w1m = su.load(w1[b, x, l])
            grad_w1_add = su.zero()

            # Multiply matrices with the correct weights and add to result
            for m in range(w_out_len):
                g_x = su.load(grad_out[b, x, m])

                # # w and t_w undaggered
                grad_w1_add, grad_w2_add = add_bilin_gradients(m, l, k, False, False, w1m, w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                # # daggered w
                grad_w1_add, grad_w2_add = add_bilin_gradients(m, w1_len + l, k, True, False, w1m, w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                # # daggered t_w
                grad_w1_add, grad_w2_add = add_bilin_gradients(m, l, w2_len + k, False, True, w1m, w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                # # daggered w and t_w
                grad_w1_add, grad_w2_add = add_bilin_gradients(m, w1_len + l, w2_len + k, True, True, w1m, w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                if use_unit_elements:
                    if l == 0:
                        # 1 * t_w
                        grad_w1_add, grad_w2_add = add_bilin_gradients(m, 2 * w1_len, k, False, False, su.unit(), w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                        # 1 * dagger(t_w)
                        grad_w1_add, grad_w2_add = add_bilin_gradients(m, 2 * w1_len, w2_len + k, False, True, su.unit(), w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight)
                    
                    if k == 0:
                        # w * 1
                        grad_w1_add, grad_w2_add = add_bilin_gradients(m, l, 2 * w2_len, False, False, w1m, su.unit(), weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                        # dagger(w) * 1
                        grad_w1_add, grad_w2_add = add_bilin_gradients(m, w1_len + l, 2 * w2_len, True, False, w1m, su.unit(), weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

                    if l == 0 and k == 0:
                        # 1 * 1
                        grad_w1_add, grad_w2_add = add_bilin_gradients(m, 2 * w1_len, 2 * w2_len, False, False, su.unit(), su.unit(), weight, g_x, grad_w1_add, grad_w2_add, grad_weight)

            su.atomic_add(grad_w1[b, x, l], grad_w1_add)

        su.atomic_add(grad_w2[b, x, k], grad_w2_add)

@jit(nopython=True)
def add_bilin_gradients(m, l, k, dagger_w1, dagger_w2, w1m, w2m, weight, g_x, grad_w1_add, grad_w2_add, grad_weight):

    factor = weight[m, l, k]
    
    if dagger_w1:
        w1m = su.dagger(w1m)

    if dagger_w2:
        w2m = su.dagger(w2m)

    # Original multiplication:
    # w_out = w1m * w2m * factor
    # out_add = su.mul_s(su.mul(w1m, w2m), factor)
    # out = su.add(out, out_add)
    
    # Derivative with respect to w1
    
    # grad_w1 = g * (w2 * f)^\dagger
    g_w1= su.mul(g_x, su.dagger(su.mul_s(w2m, factor)))
    if dagger_w1:
        g_w1 = su.dagger(g_w1)
    grad_w1_add = su.add(grad_w1_add, g_w1)
    
    # Derivative with respect to w2
    
    # grad_w2 = w1^\dagger * g * f^\dagger
    #         = w1^\dagger * g * f   ((f is real))
    g_w2= su.mul_s(su.mul(su.dagger(w1m), g_x),factor)
    if dagger_w2:
        g_w2 = su.dagger(g_w2)
    grad_w2_add = su.add(grad_w2_add, g_w2)
    
    # Derivative with respect to weight

    # matrix M,  factor f with out = M * f
    # -> grad_f = tr(M * grad_out)
    #
    # Both Wirtinger derivatives added give real quantity.
    # grad_weight = tr[(w1 * w2)^\dagger * g]/2 + tr[((w1 * w2)^\dagger * g)^\dagger]/2
    #             = real(tr[(w1*w2)^\dagger * g])
    grad_weight_add = su.tr(su.mul(su.dagger(su.mul(w1m, w2m)), g_x)).real
    cuda.atomic.add(grad_weight, (m, l, k), grad_weight_add)

    return grad_w1_add, grad_w2_add

def convert_to_su_matrix(x):
    #x = x.contiguous() # view_ax_complex requires contiguous tensors
    x = torch.view_as_complex(x)
    # for SU(3): convert from 3x3 matrix to 9 matrix elements 
    # # TODO: Use proper su.py interface to convert matrices?
    x_shape_cuda = (*x.shape[0:-2], 9)
    x = x.view(x_shape_cuda)
    return x

def convert_from_su_matrix(x):
    x_shape = (*x.shape[0:-1], 3, 3) # for SU(3): convert back to 3x3 matrix
    x = x.view(x_shape)
    x = torch.view_as_real(x)
    return x


###############################################################################################################################################
class PlaqCuda(torch.nn.Module):
    """
        A module that computes plaquettes from a link configuration
    """
    def __init__(self, dims, nc, orientation=-1):
        super(PlaqCuda, self).__init__()
        self.dims = dims
        self.nc = nc
        self.orientation = orientation
        
        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1)) # TODO: Clean-up and provide functionality through nn.lattice.py
        self.n_plaqs = self._calculate_number_of_channels()

    def get_number_of_channels(self):
        return self.n_plaqs

    def _calculate_number_of_channels(self):
        n_dims = len(self.dims)
        n_plaqs = n_dims * (n_dims - 1) // 2
        if self.orientation == -1 or self.orientation == +1:
            return n_plaqs
        else:
            return 2*n_plaqs

    def forward(self, u):
        plaqs = PlaqCudaFunc.apply(u, self.n_plaqs, self.orientation, self.dims, self.acc)
        return plaqs

    def update_dims(self, dims):
        self.dims = dims   

class PlaqCudaFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, u, n_plaqs, orientation, dims, acc):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)

        # Save the input for backward
        ctx.save_for_backward(u)
        ctx.parameters = n_plaqs, orientation, dims, acc

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())

        plaqs_shape_cuda = (*u.shape[0:2], n_plaqs, *u.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        plaqs = torch.zeros(plaqs_shape_cuda, dtype = u.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_kernel, idx_max, u, plaqs, orientation, dims_d, acc)

        plaqs = convert_from_su_matrix(plaqs)
        return plaqs
     
    @staticmethod
    def backward(ctx, grad_plaqs):
        
        # Retrieve the input
        u, = ctx.saved_tensors
        n_plaqs, orientation, dims, acc = ctx.parameters

        du = PlaqCudaBackward_du.apply(grad_plaqs, u, n_plaqs, orientation, dims, acc)

        return du, None, None, None, None

class PlaqCudaBackward_du(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grad_plaqs, u, n_plaqs, orientation, dims, acc):

        # Save the input for backward
        ctx.save_for_backward(grad_plaqs, u)
        ctx.parameters = n_plaqs, orientation, dims, acc

        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)
        dims_d = tuple(dims)
        
        u = convert_to_su_matrix(u.detach())
        grad_plaqs = convert_to_su_matrix(grad_plaqs.detach())

        du = torch.zeros(u.shape, dtype = u.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_backward_kernel, idx_max, u, grad_plaqs, du, orientation, dims_d, acc)

        du = convert_from_su_matrix(du)

        return du
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_du):

        grad_plaqs, u = ctx.saved_tensors
        n_plaqs, orientation, dims, acc = ctx.parameters
        
        dims_d = tuple(dims)
        
        u = convert_to_su_matrix(u.detach())
        grad_plaqs = convert_to_su_matrix(grad_plaqs.detach())
        grad_du = convert_to_su_matrix(grad_du.detach())

        dgrad_plaqs = torch.zeros(grad_plaqs.shape, dtype=grad_plaqs.dtype, device="cuda")
        #du = torch.zeros(u.shape, dtype=u.dtype, device="cuda")
        
        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_backward_backward_kernel, idx_max, u, grad_du, dgrad_plaqs, orientation, dims_d, acc)

        dgrad_plaqs = convert_from_su_matrix(dgrad_plaqs)

        # TODO: for true double backward, we would need to implement the proper derivative with respect to u
        du = None 

        return dgrad_plaqs, du, None, None, None, None

@myjit
def plaq_kernel(idx, u, plaqs, orientation, dims, acc):
    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)
    c = 0 # channel index of plaquettes

    # TODO: check whether shifting these loops to idx can be faster
    for mu in range(n_dims): 
        
        if orientation == -1:
            nu0 = 0
            nu1 = mu
        elif orientation == +1:
            nu0 = mu+1
            nu1 = n_dims
        else:
            nu0 = 0
            nu1 = n_dims
        
        for nu in range(nu0, nu1):
            if mu != nu:
                p = lattice.plaq(u[b], x, mu, nu, +1, +1, dims, acc)  
                su.store(plaqs[b, x, c], p)
                c += 1

@myjit
def plaq_backward_kernel(idx, u, grad_plaqs, du, orientation, dims, acc):
    # first derivative:
    #   grad_out = grad_plaqs (same shape as plaqs)
    #   d_in = du (same shape as u)

    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)

    # TODO: The following loop structure seems highly inefficient - the loops
    #       could be organized in a different way, with p_idx stored
    #       in a table beforehand. Then again, maybe this is not a bottleneck.
    for mu_u in range(n_dims):
        du_tmp = su.zero()
        c = 0 # channel index of plaquettes
        for mu in range(n_dims):
            
            if orientation == -1:
                nu0 = 0
                nu1 = mu
            elif orientation == +1:
                nu0 = mu+1
                nu1 = n_dims
            else:
                nu0 = 0
                nu1 = n_dims
            
            for nu in range(nu0, nu1):
                if mu != nu:
                    du_tmp_add = dplaq(u[b], grad_plaqs[b, :, c], x, mu_u, mu, nu, +1, +1, dims, acc)
                    du_tmp = su.add(du_tmp, du_tmp_add)
                    c += 1

        su.store(du[b, x, mu_u], du_tmp)

@myjit
def plaq_backward_backward_kernel(idx, u, grad_du, dgrad_plaqs, orientation, dims, acc):
    # second derivative:
    #   grad_out = grad_du (same shape as u)
    #   d_in = dgrad_plaqs (same shape as plaqs)

    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)

    c = 0 # channel index of plaquettes
    for mu in range(n_dims):
        
        if orientation == -1:
            nu0 = 0
            nu1 = mu
        elif orientation == +1:
            nu0 = mu+1
            nu1 = n_dims
        else:
            nu0 = 0
            nu1 = n_dims
        
        for nu in range(nu0, nu1):
            if mu != nu:
                du_tmp = d2plaq(u[b], grad_du[b], x, mu, nu, +1, +1, dims, acc)
                su.store(dgrad_plaqs[b, x, c], du_tmp)
                c += 1

# compute general derivative of plaquette d_k U_{x, oi*i, oj*j}
@myjit
def dplaq(u, g, x, k, i, j, oi, oj, dims, acc):
    res = su.zero()
    
    # Original Plaquette:
    # u0 * u1 * u2 * u3 
    # = U_{x, i} * U_{x+i, j} * U_{x+i+j, -i} * U_{x+j, -j}
    if (k == i):
        # Vary with respect to first link
        x0 = x
        if oi < 0:
            x0 = lattice.shift(x0, i, -oi, dims, acc)
        x1 = lattice.shift(x0, i, oi, dims, acc)
        x2 = lattice.shift(x1, j, oj, dims, acc)
        x3 = lattice.shift(x2, i, -oi, dims, acc)

        g0 = su.load(g[x0]) 
        u0 = lattice.get_link(u, x0, i, oi, dims, acc)
        u1 = lattice.get_link(u, x1, j, oj, dims, acc)
        u2 = lattice.get_link(u, x2, i, -oi, dims, acc)
        u3 = lattice.get_link(u, x3, j, -oj, dims, acc)

        # g_{x} * (u1 * u2 * u3)^\dagger (first derivative)
        # dx_{x} * (u1 * u2 * u3) (second derivative)
        B = su.mul(u1, su.mul(u2, u3))
        B = su.dagger(B)
        res_add = su.mul(g0, B)
        if oi < 0:
            # (g_{x} * (u1 * u2 * u3)^\dagger)^\dagger (first derivative)
            res_add = su.dagger(res_add)
        res = su.add(res, res_add)
        
        # Vary with respect to third link
        # Shift loop to starting position x0 = x - j*oj
        x0 = lattice.shift(x, j, -oj, dims, acc)
        if oi < 0:
            # derivative of daggered quantity
            x0 = lattice.shift(x0, i, -oi, dims, acc)
        x1 = lattice.shift(x0, i, oi, dims, acc)
        x2 = lattice.shift(x1, j, oj, dims, acc)
        x3 = lattice.shift(x2, i, -oi, dims, acc)

        g0 = su.load(g[x0])
        u0 = lattice.get_link(u, x0, i, oi, dims, acc)
        u1 = lattice.get_link(u, x1, j, oj, dims, acc)
        u2 = lattice.get_link(u, x2, i, -oi, dims, acc)
        u3 = lattice.get_link(u, x3, j, -oj, dims, acc)

        # (u0 * u1)^\dagger * g_{x} * u3^\dagger
        A = su.mul(u0, u1)
        B = u3
        A = su.dagger(A)
        B = su.dagger(B)
        res_add = su.mul(su.mul(A, g0), B)
        if oi > 0:
            # ((u0 * u1)^\dagger * g_{x} * u3^\dagger))^\dagger
            res_add = su.dagger(res_add)
        res = su.add(res, res_add)

    if (k == j):
        # Vary with respect to second link
        # Shift loop to starting position x0 = x - i*oi
        x0 = lattice.shift(x, i, -oi, dims, acc)
        if oj < 0:
            # derivative of daggered quantity
            x0 = lattice.shift(x0, j, -oj, dims, acc)
        x1 = lattice.shift(x0, i, oi, dims, acc)
        x2 = lattice.shift(x1, j, oj, dims, acc)
        x3 = lattice.shift(x2, i, -oi, dims, acc)

        g0 = su.load(g[x0])
        u0 = lattice.get_link(u, x0, i, oi, dims, acc)
        u1 = lattice.get_link(u, x1, j, oj, dims, acc)
        u2 = lattice.get_link(u, x2, i, -oi, dims, acc)
        u3 = lattice.get_link(u, x3, j, -oj, dims, acc)

        # u0^\dagger * g_{x} * (u2 * u3)^\dagger
        A = u0
        B = su.mul(u2, u3)
        A = su.dagger(A)
        B = su.dagger(B)
        res_add = su.mul(su.mul(A, g0), B)
        if oj < 0:
            # (u0^\dagger * g_{x} * (u2 * u3)^\dagger)^\dagger
            res_add = su.dagger(res_add)
        res = su.add(res, res_add)

        # Vary with respect to fourth link
        # Shift loop to starting position x0 = x
        x0 = x
        if oj < 0:
            # derivative of daggered quantity
            x0 = lattice.shift(x0, j, -oj, dims, acc)
        x1 = lattice.shift(x0, i, oi, dims, acc)
        x2 = lattice.shift(x1, j, oj, dims, acc)
        x3 = lattice.shift(x2, i, -oi, dims, acc)

        g0 = su.load(g[x0])
        u0 = lattice.get_link(u, x0, i, oi, dims, acc)
        u1 = lattice.get_link(u, x1, j, oj, dims, acc)
        u2 = lattice.get_link(u, x2, i, -oi, dims, acc)
        u3 = lattice.get_link(u, x3, j, -oj, dims, acc)

        # (u0 * u1 * u2)^\dagger * g_{x}
        A = su.mul(su.mul(u0, u1), u2)
        A = su.dagger(A)
        res_add = su.mul(A, g0)
        if oj > 0:
            # ((u0 * u1 * u2)^\dagger * g_{x})^\dagger
            res_add = su.dagger(res_add)
        res = su.add(res, res_add)
    return res

# compute sum_k of all backward derivatives with respect to du_k
# of backward derivative of plaquette du_k = d_k U_{x, oi*i, oj*j}
@myjit
def d2plaq(u, grad_du, x, i, j, oi, oj, dims, acc):
    res = su.zero()
    
    x0 = x
    x1 = lattice.shift(x0, i, oi, dims, acc)
    x2 = lattice.shift(x1, j, oj, dims, acc)
    x3 = lattice.shift(x2, i, -oi, dims, acc)

    u0 = lattice.get_link(u, x0, i, oi, dims, acc)
    u1 = lattice.get_link(u, x1, j, oj, dims, acc)
    u2 = lattice.get_link(u, x2, i, -oi, dims, acc)
    u3 = lattice.get_link(u, x3, j, -oj, dims, acc)
    
    # Original Plaquette:
    # u0 * u1 * u2 * u3 
    # = U_{x, i} * U_{x+i, j} * U_{x+i+j, -i} * U_{x+j, -j}

    # Vary with respect to first link
    # dx_{x} * u1 * u2 * u3
    xg = x
    if oi < 0:
        xg = lattice.shift(x, i, oi, dims, acc)
    g0 = su.load(grad_du[xg, i]) 
    if oi < 0:
        g0 = su.dagger(g0)
    B = su.mul(u1, su.mul(u2, u3))
    res_add = su.mul(g0, B)
    res = su.add(res, res_add)
    
    # Vary with respect to second link
    # u0 * dx_{x+i} * u2 * u3

    xg = lattice.shift(x, i, oi, dims, acc)
    if oj < 0:
        xg = lattice.shift(xg, j, oj, dims, acc)
    g0 = su.load(grad_du[xg, j]) 
    if oj < 0:
        g0 = su.dagger(g0)
    A = u0
    B = su.mul(u2, u3)
    res_add = su.mul(su.mul(A, g0), B)
    res = su.add(res, res_add)

    # Vary with respect to third link
    # u0 * u1 * dx_{x+j} * u3
    xg = lattice.shift(x, j, oj, dims, acc)
    if oi < 0:
        xg = lattice.shift(xg, i, oi, dims, acc)
    g0 = su.load(grad_du[xg, i]) 
    if oi > 0:
        g0 = su.dagger(g0)
    A = su.mul(u0, u1)
    B = u3
    res_add = su.mul(su.mul(A, g0), B)
    res = su.add(res, res_add)

    # Vary with respect to fourth link
    # u0 * u1 * u2 * dx_{x}
    xg = x
    if oj < 0:
        xg = lattice.shift(xg, j, oj, dims, acc)
    g0 = su.load(grad_du[xg, j])
    if oj > 0:
        g0 = su.dagger(g0)
    A = su.mul(su.mul(u0, u1), u2)
    res_add = su.mul(A, g0)
    res = su.add(res, res_add)
    return res

###############################################################################################################################################
class PlaqCloverCuda(torch.nn.Module):
    """
        A module that computes plaquettes from a link configuration
    """

    def __init__(self, dims, nc):
        super(PlaqCloverCuda, self).__init__()
        self.dims = dims
        self.nc = nc
        
        self.acc = tuple(np.append(np.cumprod(self.dims[::-1])[::-1], 1)) # TODO: Clean-up and provide functionality through nn.lattice.py
        self.sites = int(np.prod(dims))  # == self.acc[0]
        self.n_clovers = self._calculate_number_of_channels()

    def get_number_of_channels(self):
        return self.n_clovers

    def _calculate_number_of_channels(self):
        n_dims = len(self.dims)
        n_plaqs = n_dims * (n_dims - 1) // 2
        n_clovers = 4 * n_plaqs
        return n_clovers

    def forward(self, u):
        plaqs = PlaqCloverCudaFunc.apply(u, self.n_clovers, self.dims, self.acc)
        return plaqs
     
    def update_dims(self, dims):
        self.dims = dims   

class PlaqCloverCudaFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, u, n_plaqs, dims, acc):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)

        # Save the input for backward
        ctx.save_for_backward(u)
        ctx.parameters = n_plaqs, dims, acc

        dims_d = tuple(dims)

        u = convert_to_su_matrix(u.detach())

        plaqs_shape_cuda = (*u.shape[0:2], n_plaqs, *u.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        plaqs = torch.zeros(plaqs_shape_cuda, dtype = u.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_clover_kernel, idx_max, u, plaqs, dims_d, acc)

        plaqs = convert_from_su_matrix(plaqs)
        return plaqs

    @staticmethod
    def backward(ctx, grad_plaqs):
        
        # Retrieve the input
        u, = ctx.saved_tensors
        n_plaqs, dims, acc = ctx.parameters

        du = PlaqCloverCudaBackward_du.apply(grad_plaqs, u, n_plaqs, dims, acc)

        return du, None, None, None

class PlaqCloverCudaBackward_du(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grad_plaqs, u, n_plaqs, dims, acc):

        # Save the input for backward
        ctx.save_for_backward(grad_plaqs, u)
        ctx.parameters = n_plaqs, dims, acc

        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)
        dims_d = tuple(dims)
        
        u = convert_to_su_matrix(u.detach())
        grad_plaqs = convert_to_su_matrix(grad_plaqs.detach())

        du = torch.zeros(u.shape, dtype = u.dtype, device="cuda")

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_clover_backward_kernel, idx_max, u, grad_plaqs, du, dims_d, acc)

        du = convert_from_su_matrix(du)

        return du

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_du):

        grad_plaqs, u = ctx.saved_tensors
        n_plaqs, dims, acc = ctx.parameters
        
        dims_d = tuple(dims)
        
        u = convert_to_su_matrix(u.detach())
        grad_plaqs = convert_to_su_matrix(grad_plaqs.detach())
        grad_du = convert_to_su_matrix(grad_du.detach())

        dgrad_plaqs = torch.zeros(grad_plaqs.shape, dtype=grad_plaqs.dtype, device="cuda")
        #du = torch.zeros(u.shape, dtype=u.dtype, device="cuda")
        
        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites

        my_parallel_loop(plaq_clover_backward_backward_kernel, idx_max, u, grad_du, dgrad_plaqs, dims_d, acc)

        dgrad_plaqs = convert_from_su_matrix(dgrad_plaqs)

        # TODO: for true double backward, we would need to implement the proper derivative with respect to u
        du = None 

        return dgrad_plaqs, du, None, None, None, None

@myjit
def plaq_clover_kernel(idx, u, clovers, dims, acc):
    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)
    c = 0 # channel index of plaquettes

    # TODO: check whether shifting these loops to idx can be faster
    for mu in range(n_dims):
        for nu in range(0, mu):

            # four plaquettes, all with the same orientation
            # clover 0: U_x,mu,nu
            p = lattice.plaq(u[b], x, mu, nu, +1, +1, dims, acc)
            su.store(clovers[b, x, c], p)
            c += 1
            
            # # clover 1: U_x,nu,-mu
            p = lattice.plaq(u[b], x, nu, mu, +1, -1, dims, acc)
            su.store(clovers[b, x, c], p)
            c += 1
            
            # # clover 2: U_x,-mu,-nu
            p = lattice.plaq(u[b], x, mu, nu, -1, -1, dims, acc)
            su.store(clovers[b, x, c], p)
            c += 1
            
            # # clover 3: U_x,-nu,mu
            p = lattice.plaq(u[b], x, nu, mu, -1, +1, dims, acc)
            su.store(clovers[b, x, c], p)
            c += 1

@myjit
def plaq_clover_backward_kernel(idx, u, grad_plaqs, du, dims, acc):
    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)

    # TODO: The following loop structure seems highly inefficient - the loops
    #       could be organized in a different way, with p_idx stored
    #       in a table beforehand. Then again, maybe this is not a bottleneck.
    for mu_u in range(n_dims): # mu_u: direction index of link u
        du_tmp = su.zero()
        c = 0 # channel index of plaquettes
        for mu in range(n_dims):
            for nu in range(0, mu):
                # clover 0: U_x,mu,nu
                du_tmp_add = dplaq(u[b], grad_plaqs[b, :, c], x, mu_u, mu, nu, +1, +1, dims, acc)
                du_tmp = su.add(du_tmp, du_tmp_add)
                c += 1
                
                # # clover 1: U_x,nu,-mu
                du_tmp_add = dplaq(u[b], grad_plaqs[b, :, c], x, mu_u, nu, mu, +1, -1, dims, acc)
                du_tmp = su.add(du_tmp, du_tmp_add)
                c += 1
                
                # # clover 2: U_x,-mu,-nu
                du_tmp_add = dplaq(u[b], grad_plaqs[b, :, c], x, mu_u, mu, nu, -1, -1, dims, acc)
                du_tmp = su.add(du_tmp, du_tmp_add)
                c += 1
                
                # # clover 3: U_x,-nu,mu
                du_tmp_add = dplaq(u[b], grad_plaqs[b, :, c], x, mu_u, nu, mu, -1, +1, dims, acc)
                du_tmp = su.add(du_tmp, du_tmp_add)
                c += 1

        su.store(du[b, x, mu_u], du_tmp)

@myjit
def plaq_clover_backward_backward_kernel(idx, u, grad_du, dgrad_plaqs, dims, acc):
    # second derivative:
    #   grad_out = grad_du (same shape as u)
    #   d_in = dgrad_plaqs (same shape as plaqs)

    n_sites = u.shape[1]
    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    n_dims = len(dims)

    c = 0 # channel index of plaquettes
    for mu in range(n_dims):
        for nu in range(0, mu):

            # clover 0: U_x,mu,nu
            du_tmp = d2plaq(u[b], grad_du[b], x, mu, nu, +1, +1, dims, acc)
            su.store(dgrad_plaqs[b, x, c], du_tmp)
            c += 1
            
            # # clover 1: U_x,nu,-mu
            du_tmp = d2plaq(u[b], grad_du[b], x, nu, mu, +1, -1, dims, acc)
            su.store(dgrad_plaqs[b, x, c], du_tmp)
            c += 1
            
            # # clover 2: U_x,-mu,-nu
            du_tmp = d2plaq(u[b], grad_du[b], x, mu, nu, -1, -1, dims, acc)
            su.store(dgrad_plaqs[b, x, c], du_tmp)
            c += 1
            
            # # clover 3: U_x,-nu,mu
            du_tmp = d2plaq(u[b], grad_du[b], x, nu, mu, -1, +1, dims, acc)
            su.store(dgrad_plaqs[b, x, c], du_tmp)
            c += 1


    
def convert_to_su_matrix(x):
    x = x.contiguous() # view_ax_complex requires contiguous tensors
    x = torch.view_as_complex(x)
    # for SU(3): convert from 3x3 matrix to 9 matrix elements 
    # # TODO: Use proper su.py interface to convert matrices?
    x_shape_cuda = (*x.shape[0:-2], 9)
    x = x.view(x_shape_cuda)
    return x

def convert_from_su_matrix(x):
    x_shape = (*x.shape[0:-1], 3, 3) # for SU(3): convert back to 3x3 matrix
    x = x.view(x_shape)
    x = torch.view_as_real(x)
    return x