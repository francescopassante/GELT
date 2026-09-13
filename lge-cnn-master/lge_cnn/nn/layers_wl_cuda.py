import torch
from torch import Tensor
import numpy as np

from lge_cnn.ym.numba_target import myjit, mycudajit, my_parallel_loop
from lge_cnn.nn.layers_cuda import convert_to_su_matrix, convert_from_su_matrix
# import lge_cnn.ym.lattice as lattice
# import lge_cnn.ym.su as su
#from lge_cnn.ym.nn.layers import unpack_x, repack_x

from numba import cuda, jit
# import lge_cnn.nn.layers as layers

import lge_cnn.ym.su as su

class DerivativeEntry:
    def __init__(self, indices_list, name='', parent=None, index=None, w_out_id=-1, adjust_locations=True, use_start_sum_list=False):
        self.indices_list = indices_list
        self.start_term_list_gpu, self.start_sum_list_gpu, self.indices_list_gpu, self.out_len = indices_list_to_gpu(self.indices_list)
        self.w_out_id = w_out_id

        self.name=name

        self.derivatives = None
        self.parent = parent
        self.index = index

        if adjust_locations == False:
            # TODO: The logic has to be enhanced: if adjust_locations=False, the code can be even
            #       faster (as fewer location shifts have to be calculated in the kernel), 
            #       but one needs to store separately, where the derivative (of grad_out)
            #       has to be written to.
            raise Exception("DerivativeEntry: option adjust_locations=False is not supported yet")
        
        self.adjust_locations = adjust_locations     # Default 'False' is fastest option
        self.use_start_sum_list = use_start_sum_list # Default 'False' is fastest option

        if self.parent == None:
            self.index_list = ()
        else:
            self.index_list = (*self.parent.index_list, index)
            self.adjust_locations = self.parent.adjust_locations
            self.use_start_sum_list = self.parent.use_start_sum_list
            self.name = self.parent.name


class LWilsonLineCuda(torch.nn.Module):
    """
        A module that computes products of Wilson lines from a link configuration
    """

    def __init__(self, dims, nc, line_list=None):
        super(LWilsonLineCuda, self).__init__()
        self.dims = tuple(dims)
        self.nc = nc

        self.line_list = None
        self.field_types = None
        self.derivative_tree_top = None

        # cache to prevent recalculating `derivative_tree_top`
        self.cache = {}

        if line_list:
            self.init_line_list(line_list)

    def init_line_list(self, line_list):
        # use dims as hash
        h = hash(self.dims)
        if h not in self.cache:
            #print(f"dims: {self.dims} -> generate derivative_tree_top")
            # compute derivative_tree_top, indices_list, etc
            field_types = [True] # Extension for later. True=u-field, False=w-field
            self.field_types = field_types
            self.line_list = line_list
            line_sum_list = convert_simple_line_list_to_line_sum_list(line_list)
            self.indices_list = create_sum_lines(line_sum_list, self.dims, field_types)
            self.derivative_tree_top = DerivativeEntry(self.indices_list, self.__class__.__name__)

            # add results to cache
            self.cache[h] = {
                'line_list': self.line_list,
                'indices_list': self.indices_list,
                'derivative_tree_top': self.derivative_tree_top
            }
        else:
            #print(f"dims: {self.dims} -> use cache")
            # grab results from cache
            self.line_list = self.cache[h]['line_list']
            self.indices_list = self.cache[h]['indices_list']
            self.derivative_tree_top = self.cache[h]['derivative_tree_top']
        
    def get_number_of_channels(self):
        return len(self.line_list)

    def forward(self, u):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)
        w = LWilsonLineCudaFunc.apply(self.derivative_tree_top, self.dims, u)
        return w

    def update_dims(self, dims):
        self.dims = tuple(dims)
        self.init_line_list(self.line_list) # uses cache if same dims have been used already

class LWilsonLineCudaFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, derivative_entry, dims, *uw_list):
        # Save the input for backward
        ctx.save_for_backward(*uw_list)
        ctx.parameters = derivative_entry, dims

        dims_d = tuple(dims)

        uw_list_converted = []
        for u in uw_list:
            u = convert_to_su_matrix(u.detach())
            uw_list_converted.append(u)
        uw_list_converted = tuple(uw_list_converted)

        # reserve space for transported w_out
        u0 = uw_list_converted[0]
        w_out_channels = derivative_entry.out_len
        w_out_shape_cuda = (*u0.shape[0:2], w_out_channels, *u0.shape[3:]) # represent 3x3 su(3) matrices as 9 entries
        w_out = torch.zeros(w_out_shape_cuda, dtype = u0.dtype, device="cuda")

        use_start_sum_list = derivative_entry.use_start_sum_list

        if use_start_sum_list:
            start_list_gpu = derivative_entry.start_sum_list_gpu
        else:
            start_list_gpu = derivative_entry.start_term_list_gpu

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites * len(start_list_gpu)

        # Simple workaround for the fact that we can not simply pass a list of differently sized
        # arrays in Numba.
        # TODO: Maybe it is possible to convert uw_list_converted into a list of pointers
        #       and reconstruct the lists within the cuda jitted function?
        #       https://stackoverflow.com/questions/58421549/passing-numpy-arrays-as-arguments-to-numba-cfunc     
        u0 = uw_list_converted[0]
        dummy = u0 # use any list of compatible shape as dummy
        u1 = uw_list_converted[1] if len(uw_list_converted) > 1 else dummy
        u2 = uw_list_converted[2] if len(uw_list_converted) > 2 else dummy
        u3 = uw_list_converted[3] if len(uw_list_converted) > 3 else dummy
        u4 = uw_list_converted[4] if len(uw_list_converted) > 4 else dummy
        if len(uw_list_converted) > 5:
            raise Exception("LWilsonLineCudaFunc: currently only 5 nested levels allowed")
    
        w_out_id = derivative_entry.w_out_id

        if len(start_list_gpu) > 0:
            my_parallel_loop(lwilsonline_kernel, idx_max, u0, u1, u2, u3, u4, w_out_id, w_out, start_list_gpu, derivative_entry.indices_list_gpu, dims_d, derivative_entry.use_start_sum_list)
        w_out = convert_from_su_matrix(w_out)
        return w_out

    @staticmethod
    def backward(ctx, grad_out):

        # Retrieve the input
        uw_list = ctx.saved_tensors
        derivative_entry, dims = ctx.parameters

        d_uw_list = list(uw_list) + [grad_out]
        d_w_list = []

        if derivative_entry.derivatives == None:
            # Precalculate derivative entries:

            derivative_entry.derivatives = []
            for field_entry, uw in enumerate(uw_list):
                # Calculate the derivative with respect to the input tensors
                field_len = uw.shape[2]
                grad_out_entry = len(uw_list)
                d_indices_list = calculate_derivative(derivative_entry.indices_list, field_entry, field_len, grad_out_entry, dims, adjust_locations=derivative_entry.adjust_locations)
                new_derivative_entry = DerivativeEntry(d_indices_list, parent=derivative_entry, index=field_entry, w_out_id=grad_out_entry)
                derivative_entry.derivatives.append(new_derivative_entry)
                # be quiet
                #print("Derivative table generated for", new_derivative_entry.name, new_derivative_entry.index_list)

        for field_entry, uw in enumerate(uw_list):
            # Calculate the derivative with respect to the input tensors            
            d_w = LWilsonLineCudaFunc.apply(derivative_entry.derivatives[field_entry], dims, *d_uw_list)
            d_w_list.append(d_w)

        return None, None, *d_w_list

@myjit
def lwilsonline_kernel(idx, u0, u1, u2, u3, u4, w_out_id, w_out, start_list_gpu, indices_list_gpu, dims, use_start_sum_list):
    n_batch = u0.shape[0]
    n_sites = u0.shape[1]
    w_len = w_out.shape[2]

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    t = b // n_batch # transport list index
    b = b % n_batch

    k = start_list_gpu[t]

    first_element = True
    out_pos = x
    while True:
        pos_shift = indices_list_gpu[k] # (Trick: the first few indices are arranged to contain the position, so we can use the same list)
        out_i = indices_list_gpu[k][-5]
        field = indices_list_gpu[k][-4]
        field_i = indices_list_gpu[k][-3]
        dagger = indices_list_gpu[k][-2]
        stop_bit = indices_list_gpu[k][-1]
        x_s = shift_multiple(x, pos_shift, dims)
        u_mu = su.zero()
        if field == 0:
            u_mu = su.add(u_mu, su.load(u0[b, x_s, field_i]))
        elif field == 1:
            u_mu = su.add(u_mu, su.load(u1[b, x_s, field_i]))
        elif field == 2:
            u_mu = su.add(u_mu, su.load(u2[b, x_s, field_i]))
        elif field == 3:
            u_mu = su.add(u_mu, su.load(u3[b, x_s, field_i]))
        elif field == 4:
            u_mu = su.add(u_mu, su.load(u4[b, x_s, field_i]))
        if dagger > 0:
            u_mu = su.dagger(u_mu)
        if first_element:
            u_product = u_mu
            first_element = False
        else:
            u_product = su.mul(u_product, u_mu)
        
        if w_out_id == field:
            # store position for output gradient if this is a derivative
 #           out_pos = x_s
            # TODO: The logic needs to be improved for the case adjust_locations=False
            #       In this case, the correct location for the output field has to be
            #       used. This is not stored yet properly in calculate_derivative.
            #       (probably one needs a separate field for this information, or
            #       one stores it together with the indices_list_gpu field as output instruction)
            pass
        
        if stop_bit > 0:
            su.atomic_add(w_out[b, out_pos, out_i], u_product)
            if use_start_sum_list:
                # Continue with next term in sum
                first_element = True
                out_pos = x
            else:
                # Every term gets a separate cuda thread
                break
            
        if stop_bit == 2:
            # Stop at end of sum list in any case
            break
        k += 1

class PlaqWLCuda(LWilsonLineCuda):
    """
        A module that computes plaquettes from a link configuration
    """

    def __init__(self, dims, nc, orientation=-1):
        super(PlaqWLCuda, self).__init__(dims, nc)
        self.orientation = orientation

        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []
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
                    i = mu + 1 # line_list entries start at 1
                    j = nu + 1

                    line_list += [(+i, +j, -i, -j)] # Plaquette

        return line_list

class PlaqCloverWLCuda(LWilsonLineCuda):
    """
        A module that computes all clover plaquettes from a link configuration
    """
    def __init__(self, dims, nc):
        super(PlaqCloverWLCuda, self).__init__(dims, nc)

        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        for mu in range(n_dims):
            for nu in range(0, mu):
                i = mu + 1 # line_list entries start at 1
                j = nu + 1

                # four plaquettes, all with the same orientation
                # clover 0: U_x,mu,nu
                line_list += [(+i, +j, -i, -j)]

                # clover 1: U_x,nu,-mu
                line_list += [(+j, -i, -j, +i)]

                # clover 2: U_x,-mu,-nu
                line_list += [(-i, -j, +i, +j)]

                # clover 3: U_x,-nu,mu
                line_list += [(-j, +i, +j, -i)]
   
        return line_list

class RectangleWLCuda(LWilsonLineCuda):
    """
        A module that computes all rectangles from a link configuration
    """
    def __init__(self, dims, nc):
        super(RectangleWLCuda, self).__init__(dims, nc)

        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        # choose positive orientation
        for mu in range(n_dims):
            for nu in range(mu+1, n_dims):
                i = mu + 1 # line_list entries start at 1
                j = nu + 1

                # first rectangle
                line_list += [(+i, +i, +j, -i, -i, -j)]

                # second rectangle
                # line_list += [(+j, +j, +i, -j, -j, -i)] # Loop with negative orientation
                line_list += [(+i, +j, +j, -i, -j, -j)] # Same loop with positive orientation

        return line_list

class ParallelogramWLCuda(LWilsonLineCuda):
    """
        A module that computes all parallelograms from a link configuration
    """
    def __init__(self, dims, nc):
        super(ParallelogramWLCuda, self).__init__(dims, nc)
        
        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        for mu in range(n_dims):
            for nu in range(0, n_dims):
                for sigma in range(0, n_dims):
                    
                    # three different directions
                    if mu != nu and nu != sigma and mu != sigma:

                        # swapping mu <-> sigma gives the hermitian loop,
                        # so only one of these loops is added to the list
                        if mu < sigma:
                            i = mu + 1 # line_list entries start at 1
                            j = nu + 1
                            k = sigma + 1

                            # parallelogram
                            line_list += [(+i, +j, +k, -i, -j, -k)]

                            # contribution outside the hypercube
                            if mu < nu and nu < sigma:
                                line_list += [(+i, -j, +k, -i, +j, -k)]   ###  ((( corresponds to [+y, -x, +z, -y, +x, -z] )))
   
        return line_list

class ChairWLCuda(LWilsonLineCuda):
    """
        A module that computes all chairs from a link configuration
    """
    def __init__(self, dims, nc):
        super(ChairWLCuda, self).__init__(dims, nc)
        
        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        for mu in range(n_dims):
            for nu in range(0, n_dims):
                for sigma in range(0, n_dims):
                    
                    # three different directions
                    if mu != nu and nu != sigma and mu != sigma:

                        i = mu + 1 # line_list entries start at 1
                        j = nu + 1
                        k = sigma + 1

                        # chairs
                        line_list += [(+i, +j, +k, -j, -i, -k)]
                        line_list += [(+i, -j, +k, +j, -i, -k)]
   
        return line_list

class StaplesCUDA(LWilsonLineCuda):
    def __init__(self, dims, nc):
        super(StaplesCUDA, self).__init__(dims, nc)

        line_list = self.create_line_list()
        self.init_line_list(line_list)
        self.D = len(dims)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        # for each direction mu
        for mu in range(n_dims):
            # iterate over orthogonal directions
            for nu in range(n_dims):
                # both orientations of stapes
                for o in [+1, -1]:
                    if mu != nu:
                        i = mu + 1
                        j = nu + 1
                        line_list += [(o * j, +i, -o * j)]

        return line_list
    
    def forward(self, u):
        s = super().forward(u)
        s = torch.view_as_complex(s)

        # reshape into something sane
        s = s.reshape(*s.shape[0:2], self.D, self.D-1, 2, *s.shape[-2:])

        # sum over orientations
        s = torch.sum(s, dim=-3)
        
        return s

    
    def update_dims(self, dims):
        self.D = len(dims)
        return super().update_dims(dims)

class PlanarLoopWLCuda(LWilsonLineCuda):
    """
        A module that computes all rectangles from a link configuration
    """
    def __init__(self, dims, nc, M, N, W=1):
        super(PlanarLoopWLCuda, self).__init__(dims, nc)

        self.M, self.N, self.W = M, N, W
        line_list = self.create_line_list()
        self.init_line_list(line_list)
    
    def create_line_list(self):
        n_dims = len(self.dims)
        line_list = []

        # choose positive orientation
        for mu in range(n_dims):
            for nu in range(mu+1, n_dims):
                i = mu + 1 # line_list entries start at 1
                j = nu + 1

                # first loop
                loop1 = [(*([+i] * self.M), *([+j] * self.N), *([-i] * self.M), *([-j] * self.N))]
                wound_loop = []
                for w in range(self.W):
                    wound_loop.extend(loop1)
                line_list += wound_loop

                # second loop
                if self.M != self.N:
                    loop2 = [(*([+i] * self.N), *([+j] * self.M), *([-i] * self.N), *([-j] * self.M))]
                    wound_loop = []
                    for w in range(self.W):
                        wound_loop.extend(loop2)
                    line_list += wound_loop



        return line_list


@myjit
def shift_multiple(xi, pos_shift, dims):
    acc = 1
    xi_rest = xi
    xi_new = 0
    for i in range(len(dims)-1, -1, -1):
        p_new = (xi_rest + pos_shift[i]) % dims[i]
        xi_new += p_new * acc
        if i > 0:
            acc = acc * dims[i]
            xi_rest = xi_rest // dims[i]
    return xi_new

def convert_simple_line_list_to_line_sum_list(line_list):
    line_sum_list = []
    for line in line_list:
        line_sum_list.append([line])
    return line_sum_list

# line_list: list of Wilson lines
# field_types: True: u-field; False: w-field
def create_sum_lines(wilson_line_sum_list, dims, field_types):
    shifted_indices_list = []
    for wilson_line_sum in wilson_line_sum_list:
        shifted_indices_sublist = []
        for wilson_line in wilson_line_sum:
            shifted_indices = get_index_shifts(wilson_line, dims, field_types)
            shifted_indices_sublist.append(shifted_indices)
        shifted_indices_list.append(shifted_indices_sublist)
    return shifted_indices_list

# line examples: 
#   [1, 2, -1, -2], field types [True]  # Plaquette
#   [(0, 1), (0, 2), (0, -1), (0, -2)], field types [True]  # Same plaquette, explicitly denoting the default value of the field
#   [1, (1, 1), -1], field types [True, False]  # W[0]-field parallel transported
#   [1, (1, -1), -1], field types [True, False]  # (W[0]-field)^dagger parallel transported
# field_types: True: u-field; False: w-field
def get_index_shifts(line, dims, field_types):
    p = (0, ) * len(dims) # starting position
    object_list = []
    for item in line:
        if isinstance(item, list) or isinstance(item, tuple):
            field, ii = item
        else:
            field = 0 # default value
            ii = item

        if field_types[field] == True:
            # u field: shift position in the respective direction
            # convert from +-1, +-2, +-3, +-4 to 0, 1, 2, 3 and +-1
            i = abs(ii) - 1
            if ii > 0:
                oi = 1
                p0 = p
                dagger = 0
            else:
                oi = -1
                p0 = shift_position(p, i, oi, dims)
                dagger = 1
            object_list.append((field, p0, i, dagger))
            p = shift_position(p, i, oi, dims)
        else:
            # w field: local fields, do not shift position
            # convert from +-1, +-2, +-3, +-4 to 0, 1, 2, 3 and +-1
            i = abs(ii) - 1
            if ii > 0:
                dagger = 0
            else:
                dagger = 1
            object_list.append((field, p, i, dagger))
    return object_list

def shift_position(p, i, oi, dims):
    """
    Shifts a position along a specified dimension by a given number of steps.

    Args:
        p (tuple): The original position as a tuple of coordinates along different dimensions.
        i (int): The dimension along which to shift the position.
        oi (int): The number of steps to shift along the specified dimension.
        dims (tuple): The dimensions of the space along each dimension.

    Returns:
        tuple: The shifted position after applying the specified shift along the given dimension.
    """
    v = p[i]
    v = (v + oi) % dims[i]
    p = list(p)
    p[i] = v
    p = tuple(p)
    return p

def subtract_positions(p1, p2, dims):
    p3 = tuple((i1 - i2) % dim for (i1, i2, dim) in zip(p1, p2, dims))
    return p3

def negative_position(p, dims):
    p2 = tuple((-i) % dim for (i, dim) in zip(p, dims))
    return p2

def indices_list_to_gpu(indices_sum_list):
    start_term_list = []
    start_sum_list = []
    indices = []
    for out_i, indices_sum in enumerate(indices_sum_list):
        if len(indices_sum) > 0:
            start_sum_list.append(len(indices))
            for wilson_line in indices_sum:
                start_term_list.append(len(indices))
                # Add all elements except last
                for field, field_pos, field_i, dagger in wilson_line:
                    indices.append([*field_pos, out_i, field, field_i, dagger, 0])
                indices[-1][-1] = 1 # Add stop bit to last entry
            indices[-1][-1] = 2 # Add stop sum bit to last entry

    start_term_list_gpu = torch.asarray(start_term_list, device="cuda", dtype=torch.int16) # Increase dtype if an overflow occurs
    start_sum_list_gpu = torch.asarray(start_sum_list, device="cuda", dtype=torch.int16) # Increase dtype if an overflow occurs
    indices_list_gpu = torch.asarray(indices, device="cuda", dtype=torch.int8) # Increase dtype if an overflow occurs
    out_len = len(indices_sum_list)
    return start_term_list_gpu, start_sum_list_gpu, indices_list_gpu, out_len

def indices_list_from_gpu(start_term_list_gpu, start_sum_list_gpu, indices_list_gpu, out_len):
    indices_list = [[]] * out_len
    line = []
    line_sum = []
    for k in range(len(indices_list_gpu)):
        *field_pos, out_i, field, field_i, dagger, stop_bit = indices_list_gpu[k].cpu().tolist()
        line.append((field, tuple(field_pos), field_i, dagger))
        if stop_bit > 0:
            line_sum.append(line)
            line = []
        if stop_bit == 2:
            indices_list[out_i] = line_sum
            line_sum = []
    return indices_list


def calculate_derivative(indices_list, field_entry, field_len, grad_out_entry, dims, adjust_locations=True):
    # Calculate derivative for function
    d_indices_list = []
    for field_i_0 in range(field_len):
        d_indices_sumlist = []
        for grad_out_i, wilson_line_sum in enumerate(indices_list):
            for wilson_line in wilson_line_sum:
                for k, (field, field_pos, field_i, dagger) in enumerate(wilson_line):
                    if field == field_entry and field_i == field_i_0:
                        # Calculate Wilson line derivative with respect to this field
                        d_indices_wilson_line = []
                        if dagger == 0:
                            # (u_0 * u_1 * ... u_{k-1}))^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger
                            # = u_{k-1}^\dagger * ... u_1^\dagger * u_0^\dagger * grad_out * u_{n-1}^\dagger * ... * u_{k+1}
                            #
                            # Positions are:
                            # x_0, x_1, ... x_{k-1}, x(grad_out), x_{k+1}, ..., x_{n-1}
                            # where x(grad_out) is instead of the original link at x_k.
                            # Derivative gives 
                            for m in reversed(range(k)):
                                field_2, field_pos_2, field_i_2, dagger_2 = wilson_line[m]
                                if adjust_locations:
                                    field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                                d_indices_wilson_line.append((field_2, field_pos_2, field_i_2, 1 - dagger_2))
                            
                            if adjust_locations:
                                grad_out_pos = negative_position(field_pos, dims)
                            else:
         # ???                      grad_out_pos = field_pos
                                grad_out_pos = subtract_positions(field_pos, field_pos, dims)
                            d_indices_wilson_line.append((grad_out_entry, grad_out_pos, grad_out_i, 0))
                            
                            for m in reversed(range(k + 1, len(wilson_line))):
                                field_2, field_pos_2, field_i_2, dagger_2 = wilson_line[m]
                                if adjust_locations:
                                    field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                                d_indices_wilson_line.append((field_2, field_pos_2, field_i_2, 1 - dagger_2))
                        else:
                            # ((u_0 * u_1 * ... u_{k-1})^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger)^\dagger
                            # = u_{k+1} * ... * u_{n-1} * grad_out^\dagger * u_0 * u_1 * ... u_{k-1}
                            for m in range(k + 1, len(wilson_line)):
                                field_2, field_pos_2, field_i_2, dagger_2 = wilson_line[m]
                                if adjust_locations:
                                    field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                                d_indices_wilson_line.append((field_2, field_pos_2, field_i_2, dagger_2))
                            
                            if adjust_locations:
                       #         grad_out_pos = subtract_positions(field_pos, field_pos, dims)
                                grad_out_pos = negative_position(field_pos, dims)
        #                        grad_out_pos = field_pos
                            else:
       #  ???                       grad_out_pos = field_pos
                                grad_out_pos = subtract_positions(field_pos, field_pos, dims)
                            d_indices_wilson_line.append((grad_out_entry, grad_out_pos, grad_out_i, 1))
                            
                            for m in range(k):
                                field_2, field_pos_2, field_i_2, dagger_2 = wilson_line[m]
                                if adjust_locations:
                                    field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                                d_indices_wilson_line.append((field_2, field_pos_2, field_i_2, dagger_2))

                        d_indices_sumlist.append(d_indices_wilson_line)
        d_indices_list.append(d_indices_sumlist)
    return d_indices_list
