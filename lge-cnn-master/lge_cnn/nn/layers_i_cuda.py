"""
Module for CUDA layers that are based on lists of instructions.

These instruction lists are based on simple operations that are executed in a CUDA kernel (such as multiply a complex matrix, story temporary result, write output to field).

Derivatives of instruction lists can be taken to create new instruction lists for the derivatives.

High-level layers such as LPlaqICuda or LConvBilinICuda are expressed in terms of these instruction lists.

For backward compatibility with layers_wl_cuda.py, wilson line lists can be converted to instruction lists.
"""


import torch
import numpy as np

from lge_cnn.ym.numba_target import myjit, my_parallel_loop
from lge_cnn.nn.layers_cuda import convert_to_su_matrix, convert_from_su_matrix

import lge_cnn.ym.su as su

###############################################################################
## Definition of Instruction IDs used for GPU

# Instructions are lists of the following entries:
# - instruction_id, index of field (in case of multiple fields), (offset-)direction (+-1, +-2, ...) or index (0,1,2,3,4)

### Command operations
RESET_VALUES = 1 # Resets values
STOP_EXECUTION = 2 # Stops execution of current thread
RESET_POSITION = 3 # Reset internal position (for storing U fields at same location as reading them)

### Matrix operations
MULTIPLY_U = 4 # field index, offset-direction
MULTIPLY_U_DAGGER = 5  # field index, offset-direction; Only low-level instruction
MULTIPLY_W = 6 # Only high-level instruction. Replaced by MULTIPLY_U for GPU purposes.
MULTIPLY_W_DAGGER = 7 # Only high-level instruction. Replaced by MULTIPLY_U for GPU purposes.
MULTIPLY_WEIGHT = 8 # Multiply field value by weight
ADD_OUTPUT_U = 9 # Add U matrix to output field
ADD_OUTPUT_W = 10 # Add W matrix to output field (same as U, but includes offset). Replaced by ADD_OUTPUT_U for GPU purposes.
STORE_TEMPORARY = 11 # Store temporary U matrix to be retrieved later
RETRIEVE_TEMPORARY = 12 # Retrieve stored U matrix
TRACE_TO_SCALAR = 13 # Calculate trace and store in scalare value

### Scalar operations
MULTIPLY_SCALAR_WEIGHT = 13 # Multiply scalar value by weight
ADD_OUTPUT_SCALAR_WEIGHT = 14 # Add scalar value to weight matrix


###############################################################################
## Top level classes

class DerivativeEntry:
    def __init__(self, instructions, out_len, name='', parent=None, index=None):
        self.instructions = instructions
        self.out_len = out_len
        self.start_instructions_gpu, self.instructions_gpu = instruction_list_to_gpu(self.instructions)

        self.name=name

        self.derivatives_weight = None
        self.derivatives_uw = None
        self.parent = parent
        self.index = index

        if self.parent == None:
            self.index_list = ()
        else:
            self.index_list = (*self.parent.index_list, index)
            self.name = self.parent.name


class LInstructionsCuda(torch.nn.Module):
    """
        A module that executes instructions for gauge fields.
    """

    def __init__(self, dims, nc, line_list=None):
        super(LInstructionsCuda, self).__init__()
        self.dims = tuple(dims)
        self.nc = nc

        self.line_list = None
        self.field_types = None
        self.derivative_tree_top = None

        # cache to prevent recalculating `derivative_tree_top`
        self.cache = {}

        self.weight = None # Placeholder for weight

        if line_list:
            self.init_line_list(line_list)

    def init_line_list(self, line_list):
        # For compatibility with classes from layers_wl_cuda
        instructions, out_len = convert_wilson_lines_to_instructions(line_list)
        self.init_instructions(instructions, out_len)

    def init_instructions(self, instructions, out_len):
        self.original_instructions = instructions

        # use dims as hash
        h = hash(self.dims)
        if h not in self.cache:
            #print(f"dims: {self.dims} -> generate derivative_tree_top")
            # compute derivative_tree_top, indices_list, etc
            instructions = resolve_index_shifts(instructions, self.dims)
            self.instructions = instructions
            self.out_len = out_len
            self.derivative_tree_top = DerivativeEntry(self.instructions, self.out_len, self.__class__.__name__)

            # add results to cache
            self.cache[h] = {
                'instructions': self.instructions,
                'derivative_tree_top': self.derivative_tree_top
            }
        else:
            #print(f"dims: {self.dims} -> use cache")
            # grab results from cache
            self.instructions = self.cache[h]['instructions']
            self.derivative_tree_top = self.cache[h]['derivative_tree_top']
        
    def get_number_of_channels(self):
        return len(self.line_list)

    def forward(self, u):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)
        if self.weight is not None:
            weightlist = [self.weight]
        else:
            weightlist = []
        w = LInstructionsCudaFunc.apply(self.derivative_tree_top, self.dims, len(weightlist), *weightlist, u)
        return w

    def update_dims(self, dims):
        self.dims = tuple(dims)
        self.init_line_list(self.line_list) # uses cache if same dims have been used already

class LInstructionsCudaFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, derivative_entry, dims, weightlist_len, *weight_uw_list):
        """
        In the forward pass we receive weights and U/W fields containing the input and return
        a W field containing the output.
        """
        # Save the input for backward
        ctx.save_for_backward(*weight_uw_list)
        ctx.parameters = derivative_entry, dims, weightlist_len

        weight_list = weight_uw_list[:weightlist_len]
        uw_list = weight_uw_list[weightlist_len:]
        
        dims_d = tuple(dims)

        weight_list_converted = []
        for weight in weight_list:
            weight = weight.detach()
            weight = torch.flatten(weight, start_dim=1) # flatten weights but keep batch dimension separate
            weight_list_converted.append(weight)
        weight_list_converted = tuple(weight_list_converted)

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

        start_instructions_gpu = derivative_entry.start_instructions_gpu
        instructions_gpu = derivative_entry.instructions_gpu

        n_batch = u.shape[0]
        n_sites = u.shape[1]
        idx_max = n_batch * n_sites * len(start_instructions_gpu)

        # Simple workaround for the fact that we can not simply pass a list of differently sized
        # arrays in Numba.
        # TODO: Maybe it is possible to convert uw_list_converted into a list of pointers
        #       and reconstruct the lists within the cuda jitted function?
        #       https://stackoverflow.com/questions/58421549/passing-numpy-arrays-as-arguments-to-numba-cfunc
        dummy = torch.zeros((n_batch, 1), device="cuda") # use any list of compatible shape as dummy
        w0 = weight_list_converted[0] if len(weight_list_converted) > 0 else dummy
        w1 = weight_list_converted[1] if len(weight_list_converted) > 1 else dummy
        w2 = weight_list_converted[2] if len(weight_list_converted) > 2 else dummy
        w3 = weight_list_converted[3] if len(weight_list_converted) > 3 else dummy
        w4 = weight_list_converted[4] if len(weight_list_converted) > 4 else dummy
        if len(weight_list_converted) > 5:
            raise Exception("LWilsonLineCudaFunc: currently only 5 nested levels allowed")

        u0 = uw_list_converted[0]
        dummy = u0 # use any list of compatible shape as dummy
        u1 = uw_list_converted[1] if len(uw_list_converted) > 1 else dummy
        u2 = uw_list_converted[2] if len(uw_list_converted) > 2 else dummy
        u3 = uw_list_converted[3] if len(uw_list_converted) > 3 else dummy
        u4 = uw_list_converted[4] if len(uw_list_converted) > 4 else dummy
        if len(uw_list_converted) > 5:
            raise Exception("LWilsonLineCudaFunc: currently only 5 nested levels allowed")

        if len(start_instructions_gpu) > 0:
            my_parallel_loop(instructions_kernel, idx_max, w0, w1, w2, w3, w4, u0, u1, u2, u3, u4, w_out, start_instructions_gpu, instructions_gpu, dims)
        w_out = convert_from_su_matrix(w_out)
        return w_out

    @staticmethod
    def backward(ctx, grad_out):
        """
        In the backward pass we receive grad_out containing the 'gradient of the loss
        with respect to the output' (which has the same shape as the output), 
        and we need to compute the 'gradient of the loss with respect to the input'
        (which has the same shape as the input).
        """

        # Retrieve the input
        saved_tensors = ctx.saved_tensors
        derivative_entry, dims, weightlist_len = ctx.parameters
        weight_list = saved_tensors[:weightlist_len]
        uw_list = saved_tensors[weightlist_len:]

        # TODO: What happens if the derivative with respect to the weights is calculated? 
        # Add grad_out to uw_list (it has the same shape as the output of the forward pass)
        d_weight_list = list(weight_list) + [grad_out]
        d_uw_list = list(uw_list) + [grad_out]

        d_weight_uw_list = []

        # TODO: LET US SKIP DERIVATIVES WITH RESPECT TO WEIGHTS FOR NOW. This has to be implemented later.

        # if derivative_entry.derivatives_weight == None:
        #     # Precalculate derivative entries:

        #     derivative_entry.derivatives_weight = []
        #     for field_entry_weight, weight in enumerate(weight_list):
        #         # Calculate the derivative with respect to the weights
        #         field_len = weight.shape[1]
        #         grad_out_entry = len(weight_list)
        #         d_weight_instructions = calculate_derivative_weight(derivative_entry.instructions, field_entry_weight, grad_out_entry)
        #         new_derivative_entry = DerivativeEntry(d_weight_instructions, field_len, parent=derivative_entry, index=field_entry_weight)
        #         derivative_entry.derivatives_weight.append(new_derivative_entry)
        #         # be quiet
        #         #print("Derivative table generated for", new_derivative_entry.name, new_derivative_entry.index_list)

        if derivative_entry.derivatives_uw == None:
            # Precalculate derivative entries:

            derivative_entry.derivatives_uw = []
            for field_entry_uw, uw in enumerate(uw_list):
                # Calculate the derivative with respect to the input tensors
                field_len = uw.shape[2]
                grad_out_entry = len(uw_list)
                d_uw_instructions = calculate_derivative_uw(derivative_entry.instructions, field_entry_uw, grad_out_entry, dims)
                new_derivative_entry = DerivativeEntry(d_uw_instructions, field_len, parent=derivative_entry, index=field_entry_uw)
                derivative_entry.derivatives_uw.append(new_derivative_entry)
                # be quiet
                #print("Derivative table generated for", new_derivative_entry.name, new_derivative_entry.index_list)

        for field_entry_weight, weight in enumerate(weight_list):
            # # Calculate the derivative with respect to the weights
            # d_weight = LInstructionsCudaFunc.apply(derivative_entry.derivatives_weight[field_entry_weight], dims, len(d_weight_list), *d_weight_list, *uw_list)
            # d_weight_uw_list.append(d_weight)

            ### TODO: Skip for now, should implement proper derivative later
            d_weight_uw_list.append(None)

        for field_entry_uw, uw in enumerate(uw_list):
            # Calculate the derivative with respect to the input tensors
            d_uw = LInstructionsCudaFunc.apply(derivative_entry.derivatives_uw[field_entry_uw], dims, len(weight_list), *weight_list, *d_uw_list)
            d_weight_uw_list.append(d_uw)

        # return fields correspond derivatives with respect to forward arguments:
        # derivative_entry, dims, weightlist_len, *weight_uw_list
        return None, None, None, *d_weight_uw_list


###############################################################################
## Wilson line classes

class PlaqICuda(LInstructionsCuda):
    """
        A module that computes plaquettes from a link configuration
    """

    def __init__(self, dims, nc, orientation=-1):
        super(PlaqICuda, self).__init__(dims, nc)
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



###############################################################################
## Instruction classes

class LConvBilinICuda(LInstructionsCuda):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization
    """

    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True, use_symmetric=True):
        super(LConvBilinICuda, self).__init__(dims, nc)
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
        self.weight = torch.nn.Parameter(data=torch.Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        self.w_in_size = w_in_size
        self.t_w_size = t_w_size

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

        # Create instructions for LConvBilin
        instruction_list = self.create_instruction_list()

        # Instructions for generating U + W fields
        self.init_instructions(instruction_list, len(self.dims) + self.n_out)
    
    def create_instruction_list(self):
        """
        Create the instruction list for an LConvBilin layer.

        First the U fields are simply copied.

        For the W fields, the instruction lists consists of individual instruction blocks
        of the following form:

        ## Initialization
        - RESET_VALUES

        ## Multiply links
        Multiply links for a given Wilson line

        - MULTIPLY_U        # parallel transport (depending on kernel size)
        - MULTIPLY_U
        - MULTIPLY_W        # multiplication with W field
        - MULTIPLY_U_DAGGER # parallel transport back
        - MULTIPLY_U_DAGGER

        ## Multiply weights and store output
        Multiply by weight and store in output.
        One link configuration is written to all output positions with different
        weight factor.

        - STORE_TEMPORARY    # Store result of U U W U^\dagger U^\daagger
        - MULTIPLY_WEIGHT    # Multiply by weight factor
        - ADD_OUTPUT_W       # Write to output W field
        - RETRIEVE_TEMPORARY # Retrieve value of U U W U^\dagger U^\daagger
        - MULTIPLY_WEIGHT    # Proceed as above with next weight factor and next W field
        - ADD_OUTPUT_W
        - ...

        - STOP_EXECUTION     # Stop operation on this kernel
        """ 

        instructions = []

        # Output the U fields
        instructions += self.copy_U_fields()

        # Output the W fields
        for w_in_index in range(self.w_in_size):
            instructions_intro = []
            instructions_intro.append((RESET_VALUES, 0, 0))

            # Start with original W field (or dagger or unit)
            if w_in_index < self.n_in:
                # Start with W-field
                instructions_intro.append((MULTIPLY_W, 0, w_in_index))
            elif w_in_index < 2 * self.n_in:
                # Start with W-dagger-field
                instructions_intro.append((MULTIPLY_W_DAGGER, 0, w_in_index - self.n_in))
            else:
                # Start with unity - no command necessary
                pass

            t_w_index = 0

            # Multiply by parallel transported W field
            instructions, t_w_index = self.loop_over_W_fields(instructions, t_w_index, instructions_intro, 
                                                w_in_index, 0, 0, -1, MULTIPLY_W)

            # gather all terms along lattice axes up to kernel_size
            for axis in range(len(self.dims)):
                for i, o in zip([0, 1], [-1, +1]):
                    kernel_size = abs(self.kernel_range[axis][i])
                    for d in range(kernel_size):

                        instructions, t_w_index = self.loop_over_W_fields(instructions, t_w_index, instructions_intro, 
                                                            w_in_index, axis, o, (d+1) * self.dilation, MULTIPLY_W)
            
            # Multiply by parallel transported W-dagger field
            instructions, t_w_index = self.loop_over_W_fields(instructions, t_w_index, instructions_intro, 
                                                w_in_index, 0, 0, -1, MULTIPLY_W_DAGGER)
            
            # gather all terms along lattice axes up to kernel_size
            for axis in range(len(self.dims)):
                for i, o in zip([0, 1], [-1, +1]):
                    kernel_size = abs(self.kernel_range[axis][i])
                    for d in range(kernel_size):

                        instructions, t_w_index = self.loop_over_W_fields(instructions, t_w_index, instructions_intro, 
                                                            w_in_index, axis, o, (d+1) * self.dilation, MULTIPLY_W_DAGGER)
        
            # Multiply by unit element
            if self.use_unit_elements:
                # Build code
                instructions += instructions_intro

                # Now output everything multiplying by the weights
                instructions += self.multiply_weights_and_add_output(w_in_index, t_w_index)

                # Thread ends
                instructions.append((STOP_EXECUTION, 0, 0))

                t_w_index += 1

        return instructions
    
    def copy_U_fields(self):
        # Simply copy U fields from input to output
        instructions = []
        for uu in range(0,len(self.dims)):
            instructions.append((RESET_VALUES, 0, 0))
            instructions.append((MULTIPLY_U, 0, (uu + 1) * 1)) ## offset by one for direction/orientation information
            instructions.append((RESET_POSITION, 0, 0)) # Reset position for storing U fields at same location as reading them
            instructions.append((ADD_OUTPUT_U, 0, uu)) # ADD_OUTPUT_U does not use offset
            instructions.append((STOP_EXECUTION, 0, 0))
        return instructions
    
    def loop_over_W_fields(self, instructions, t_w_index, instructions_intro, w_in_index, axis, orientation, count, multiply_instruction):
        # Loop over all W fields:
        for ww in range(self.n_in):
            # Build code
            instructions += instructions_intro

            # Multiply by links d*self.dilation times
            instructions += self.multiply_many_U(axis, orientation, count)
            
            # Multiply by W
            instructions.append((multiply_instruction, 0, ww))  # Here is the only difference to above

            # Multiply by daggered links d*self.dilation times
            instructions += self.multiply_many_U(axis, -orientation, count)

            # Now output everything multiplying by the weights
            instructions += self.multiply_weights_and_add_output(w_in_index, t_w_index)

            # Thread ends
            instructions.append((STOP_EXECUTION, 0, 0))

            t_w_index += 1
            
        return instructions, t_w_index

    def multiply_many_U(self, axis, orientation, count):
        instructions = []
        for i in range(count):
            # Multiply by U-link pointing in direction axis and orientation o
            instructions.append((MULTIPLY_U, 0, (axis+1) * orientation))
        return instructions

    def multiply_weights_and_add_output(self, w_in_index, t_w_index):
        instructions = []
        instructions.append((STORE_TEMPORARY, 0, 0))
        first_time = True
        for w_out_index in range(self.n_out):
            if first_time:
                first_time = False
            else:
                instructions.append((RETRIEVE_TEMPORARY, 0, 0))
            flattened_weight_index = (w_out_index * self.w_in_size * self.t_w_size
                                    + w_in_index * self.t_w_size
                                    + t_w_index)
            instructions.append((MULTIPLY_WEIGHT, 0, flattened_weight_index))
            instructions.append((ADD_OUTPUT_W, 0, w_out_index))
        return instructions



###############################################################################
## CUDA kernel functions


@myjit
def instructions_kernel(idx, w0, w1, w2, w3, w4, u0, u1, u2, u3, u4, w_out, start_list_gpu, instructions_list_gpu, dims):
    """
    CUDA kernel that executes instruction lists.

    :param w0...w4: real fields
    :param u0...u4: SU(3) complex matrices fields
    :param w_out: SU(3) output field
    """
    n_batch = u0.shape[0]
    n_sites = u0.shape[1]

    x = idx % n_sites # position index
    b = idx // n_sites # batch index
    t = b // n_batch # transport list index
    b = b % n_batch

    k = start_list_gpu[t]

    first_element = True
    # scalar = 0.
    while True:
        pos_shift = instructions_list_gpu[k] # (Trick: the first few indices are arranged to contain the position, so we can use the same list)
        instruction = instructions_list_gpu[k][-3]
        field = instructions_list_gpu[k][-2]
        field_i = instructions_list_gpu[k][-1]

        # TODO: Possible optimization: Does it make a difference, in which order the instructions appear in the if/elif/elif/elif/elif/... block? Frequent/lightweight(?) ones could be on top.
        if instruction == RESET_VALUES:
            first_element = True
        elif instruction == MULTIPLY_U or instruction == MULTIPLY_U_DAGGER:
            # U and W fields are treated in the same way
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
            if instruction == MULTIPLY_U_DAGGER:
                u_mu = su.dagger(u_mu)
            if first_element:
                u_product = u_mu
                first_element = False
            else:
                u_product = su.mul(u_product, u_mu)
        elif instruction == MULTIPLY_WEIGHT: ### or instruction == MULTIPLY_SCALAR_WEIGHT:
            x_s = shift_multiple(x, pos_shift, dims)
            if field == 0:
                w = w0[b, field_i]
            elif field == 1:
                w = w1[b, field_i]
            elif field == 2:
                w = w2[b, field_i]
            elif field == 3:
                w = w3[b, field_i]
            elif field == 4:
                w = w4[b, field_i]
            ### if instruction == MULTIPLY_WEIGHT:
            if first_element:
                u_product = su.mul_s(su.unit(), w)
                first_element = False
            else:
                u_product = su.mul_s(u_product, w)
            # else: # instruction == MULTIPLY_SCALAR_WEIGHT
            #     pass
            #     ### TODO: Has to be implemented layter
            #     scalar *= w
        elif instruction == ADD_OUTPUT_U:
            su.atomic_add(w_out[b, x, field_i], u_product)
        # elif instruction == TRACE_TO_SCALAR:
        #     if first_element:
        #         raise ValueError("CUDA: Unexpected trace without matrix", instruction, "at pos", k)
        #     scalar = su.tr(u_product)
        # elif instruction == ADD_OUTPUT_SCALAR_WEIGHT:
        #     raise ValueError("CUDA: ADD_OUTPUT_SCALAR_WEIGHT not implemented yet, at pos", k)
        #     # su.atomic_acc(weight_out[b, field_i])    ####### need weight_out!!
        elif instruction == STORE_TEMPORARY:
            u_product_temp = u_product
        elif instruction == RETRIEVE_TEMPORARY:
            u_product = u_product_temp
        elif instruction == STOP_EXECUTION:
            break
        else:
            if x == 0 and b == 0:
                # raise ValueError("CUDA: Unexpected instruction", instruction, "at pos", k)
                print("CUDA: Unexpected instruction", instruction, "at pos", k)
        
        k += 1

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

###############################################################################
## Instruction handling functions

# Notes:
#   - U and W share same space, the only difference is that after loading (and multiplying) W, there is no automatic position shift.
#   - U/W and Scalar have separate index lists that both start at 0.
# TODO: list of instructions could be further optimized.
# - W * W-dagger: MULTIPLY_W + MULTIPLY_W_DAGGER on same location could be both removed. (similarly W-dagger * W)
# - STORE_TEMPORARY: could be removed if no RETRIEVE_TEMPORARY follows (only single weight - usually not the case)

# line_list: list of Wilson lines
def convert_wilson_lines_to_instructions(wilson_line_list):
    instructions = []
    for out_i, wilson_line in enumerate(wilson_line_list):
        instructions.append((RESET_VALUES, 0, 0))
        for ii in wilson_line:
            instructions.append((MULTIPLY_U, 0, ii))
        # Add output to output channel
        instructions.append((ADD_OUTPUT_U, 0, out_i))
        instructions.append((STOP_EXECUTION, 0, 0))
    out_len = len(wilson_line_list)
    return instructions, out_len

# line examples: 
#   [(LOAD_U, 0, 1), (LOAD_U, 0, 2), (LOAD_U, 0, -1), (LOAD_U, 0, -2), (ADD_OUTPUT, 0, 0)] # Plaquette of field 0
#   [(LOAD_U, f, 1), (LOAD_U, f, 2), (LOAD_U, f, -1), (LOAD_U, f, -2), (ADD_OUTPUT, 0, 0)]  # Plaquette of field #f
#   [(LOAD_U, 0, 1), (LOAD_W, 1, 1), (LOAD_U, 0, -1), (ADD_OUTPUT, 0, 0)]  # W[0]-field parallel transported: U_x(p) W_0(p+x) U_x^dagger(p)
#   [(LOAD_U, 0, 1), (LOAD_W, 1, -1), (LOAD_U, 0, -1), (ADD_OUTPUT, 0, 0)]  # (W[0]-field)^dagger parallel transported
#   [(LOAD_U, 0, 1), (LOAD_W, 1, k), (LOAD_U, 0, -1), (ADD_OUTPUT, 0, 0)]  # W[k]-field parallel transported
#   [(LOAD_U, 1, 1), (LOAD_W, 0, k+1), (LOAD_U, 1, -1), (STORE_TEMPORARY), (LOAD_SCALAR, 0, 0), (ADD_OUTPUT, 0, ), (RETRIEVE_TEMPORARY), (LOAD_SCALAR, 0, 1), (ADD_OUTPUT, 0, )]  # weight[m+1] U_x(p) W_k(p+x) U_x^dagger(p)

def resolve_index_shifts(instructions, dims):
    """Resolves index shifts in instructions with incremental line information to relative positions.
    e.g. Wilson loop [+1, +2, -1, -2] in dim=2 dimensions gets converted to the positions
    [[position: (0, 0), field direction: 0, dagger: 0], 
     [position: (1, 0), field direction: 1, dagger: 0],
     [position: (0, 1), field direction: 0, dagger: 1],
     [position: (0, 0), field direction: 1, dagger: 1]]

     For W fields or scalar fields, the original index is kept (no subtraction of 1).
    """
    p = (0, ) * len(dims) # starting position
    p_start = p
    instructions_list = []
    for item in instructions:
        instruction, field, ii = item

        if instruction == RESET_VALUES:
            ## field, p, and ii is ignored
            p = p_start
            instructions_list.append((instruction, field, p, ii))
        elif instruction == MULTIPLY_U:
            # convert index from +-1, +-2, +-3, +-4 to 0, 1, 2, 3 and +-1
            i = abs(ii) - 1
            if ii > 0:
                dagger = 0
            else:
                dagger = 1
                
            # u field: shift position in the respective direction
            if ii > 0:
                oi = 1
                p0 = p
            else:
                oi = -1
                p0 = shift_position(p, i, oi, dims)
            if dagger == 0:
                instructions_list.append((MULTIPLY_U, field, p0, i))
            else:
                instructions_list.append((MULTIPLY_U_DAGGER, field, p0, i))
            p = shift_position(p, i, oi, dims)
        elif instruction == MULTIPLY_W:
            # w field: local fields, do not shift position.
            # We can treat U and W fields from now on equally.
            # W fields use original index ii.
            # ii gets shifted by number of dimensions in combined U+W field:
            W_ii = len(dims) + ii
            instructions_list.append((MULTIPLY_U, field, p, W_ii))
        elif instruction == MULTIPLY_W_DAGGER:
            # w field: local fields, do not shift position.
            # We can treat U and W fields from now on equally.
            # W fields use original index ii.
            # ii gets shifted by number of dimensions in combined U+W field:
            W_ii = len(dims) + ii
            instructions_list.append((MULTIPLY_U_DAGGER, field, p, W_ii))
        elif instruction == MULTIPLY_WEIGHT:
            # Scalar field
             ## index p is ignored later. Weights just have original index ii
            instructions_list.append((instruction, field, p, ii))
        elif instruction == ADD_OUTPUT_U:
             ## index p is ignored later. Weights just have original index ii
            instructions_list.append((ADD_OUTPUT_U, field, p, ii))
        elif instruction == ADD_OUTPUT_W:
             ## index p is ignored later. Weights just have original index ii
            # ii gets shifted by number of dimensions in combined U+W field:
            W_ii = len(dims) + ii
            instructions_list.append((ADD_OUTPUT_U, field, p, W_ii))
        elif instruction == STORE_TEMPORARY:
             ## index p is ignored later.
             # TODO: Later may have multiple temporary storage space through index ii
             # TODO: Shall we also store current location p?
            instructions_list.append((instruction, field, p, ii))
        elif instruction == RETRIEVE_TEMPORARY:
             ## index p is ignored later.
             # TODO: Later may have multiple temporary storage space through index ii
             # TODO: Shall we also retrieve current location p?
            instructions_list.append((instruction, field, p, ii))
        elif instruction == STOP_EXECUTION:
            ## field, p, and ii is ignored
            instructions_list.append((instruction, field, p, ii))
        elif instruction == RESET_POSITION:
            # Restore original position (for storing U fields at same location as reading them)
            p = p_start
        else:
            raise ValueError('instruction ' + str(instruction) + ' unknown.')
        

    return instructions_list
def instruction_list_to_gpu(instructions_list):
    """Converts the instructions_list to be used in the wilsonline_kernel.
    The order of instruction items is switched.
    Also a list of starting points is constructed."""
    start_instructions_list = []
    instructions_list_converted = []
    start_new_instruction = True
    for instruction_line in instructions_list:
        if start_new_instruction:
            start_instructions_list.append(len(instructions_list_converted))
            start_new_instruction = False
        instruction, field, field_pos, field_i = instruction_line
        instructions_list_converted.append([*field_pos, instruction, field, field_i])
        if instruction == STOP_EXECUTION:
            start_new_instruction = True

    start_instructions_list_gpu = torch.asarray(start_instructions_list, device="cuda", dtype=torch.int32) # Increase dtype if an overflow occurs
    instructions_list_gpu = torch.asarray(instructions_list_converted, device="cuda", dtype=torch.int32) # int8 Increase dtype if an overflow occurs
    return start_instructions_list_gpu, instructions_list_gpu

###############################################################################
## Helper functions for position calculations

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


###############################################################################
## Calculating derivatives of instructions

def split_instructions(instructions):
    """Split instructions at the STOP_EXECUTION command in sublists."""
    instructions_split = []
    current_sublist = []

    for instruction_line in instructions:
        current_sublist.append(instruction_line)
        if instruction_line[0] == STOP_EXECUTION:
            instructions_split.append(current_sublist)
            current_sublist = []

    if current_sublist:
        instructions_split.append(current_sublist)
    
    return instructions_split

def split_instruction_block(instruction_block):
    """Split instruction block into a init, fields, and weights instructions.
    
    Typical instruction block of instruction_list is split in the following way:
    ### instructions_init ###
    - RESET_VALUES

    ### instructions_fields ###
    - MULTIPLY_U
    - MULTIPLY_U
    - MULTIPLY_U_DAGGER
    - ...

    ### instructions_rest ###
    - STORE_TEMPORARY
    - RETRIEVE_TEMPORARY
    - MULTIPLY_WEIGHT
    - ADD_OUTPUT_U
    - RETRIEVE_TEMPORARY
    - MULTIPLY_WEIGHT
    - ADD_OUTPUT_U
    - ...
    - STOP_EXECUTION

    More complicated orders of mixing store/retrieve temporary with multiply_u are not supported.
    """
    instructions_init = []
    instructions_fields = []
    instructions_rest = []

    mode = 0

    for instruction_line in instruction_block:
        instruction = instruction_line[0]
        if mode == 0:
            if instruction == MULTIPLY_U or instruction == MULTIPLY_U_DAGGER:
                # switch to next mode
                mode = 1
        elif mode == 1:
            if instruction != MULTIPLY_U and instruction != MULTIPLY_U_DAGGER:
                # switch to last mode
                mode = 2
        else:
            if instruction == MULTIPLY_U or instruction == MULTIPLY_U_DAGGER:
                # There should not be any MULTIPLY in the last block, otherwise the logic is too complicated for us to handle.
                raise ValueError('extract_fields_from_instruction_block: Unexpected MULTIPLY_U(_DAGGER)')
        
        if mode == 0:
            instructions_init.append(instruction_line)
        elif mode == 1:
            instructions_fields.append(instruction_line)
        else:
            instructions_rest.append(instruction_line)
    
    return instructions_init, instructions_fields, instructions_rest

def extract_weight_and_output_blocks(instructions_rest):
    """Split instructions rest into individual weight and output blocks.
    
    Extract either
    - MULTIPLY_WEIGHT
    - ADD_OUTPUT_U

    or just
    - ADD_OUTPUT_U

    Other combinations are not supported and ignored.
    """
    weight_and_output_list = []
    subblock = []

    for instruction_line in instructions_rest:
        instruction = instruction_line[0]
        if instruction == MULTIPLY_WEIGHT:
            if len(subblock) > 0:
                # Subblock should be empty. Multiple weights are not supported yet.
                print(instructions_rest)
                raise ValueError("extract_weight_add_blocks: subblock should be empty", instruction_line)
            subblock.append(instruction_line)
        elif instruction == ADD_OUTPUT_U:
            subblock.append(instruction_line)
            weight_and_output_list.append(subblock)
            subblock = []
        elif instruction == RETRIEVE_TEMPORARY or instruction == STOP_EXECUTION or instruction == STORE_TEMPORARY:
            # These are allowed commands - ignore them
            pass
        else:
            raise ValueError("extract_weight_add_blocks: unexpected command", instruction_line)
    
    if subblock:
        # The last command should have been ADD_OUTPUT_U which empties a subblock
        raise ValueError("extract_weight_add_blocks: subblock not empty", instruction_line)
        
    return weight_and_output_list

def calculate_derivative_uw(instructions, field_entry, grad_out_entry, dims):
    """Calculate derivative with respect to u or w fields.

    Original expression is of the form:
    U_temp = U[field_1, i_1] * U[field_2, i_2] * U[field_3, i_3]^dagger; U_temp * weight_1, U_temp * weight_2, U_temp * weight_3, ...

    -> We apply the rules for U and U^dagger derivatives to the first part, and append the weight part unmodified.

    -> For example, if field_1 and field_3 agree with field_entry, but field_2 is different (field_1 = field_3 = field_entry != field_2)
       then field_2 entry is dropped, and the others are changed to grad_out_entry:

    For a chain of U links of the following form:
    u_0 * u_1 * ... * u_{k-1} * u_{k} * u_{k+1} * ... u_{n-1}

    the rule for U derivative at position k is:
    (u_0 * u_1 * ... u_{k-1}))^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger
    = u_{k-1}^\dagger * ... u_1^\dagger * u_0^\dagger * grad_out * u_{n-1}^\dagger * ... * u_{k+1}

    For a chain of U links of the following form:
    u_0 * u_1 * ... * u_{k-1} * u_{k}^\dagger * u_{k+1} * ... u_{n-1}

    the rule for U derivative at position k is: 
    ((u_0 * u_1 * ... u_{k-1})^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger)^\dagger
    = u_{k+1} * ... * u_{n-1} * grad_out^\dagger * u_0 * u_1 * ... u_{k-1}

    Typical instruction block of instruction_list:
    ### instructions_init ###
    - RESET_VALUES

    ### instructions_fields ###
    - MULTIPLY_U
    - MULTIPLY_U
    - MULTIPLY_U_DAGGER <-- field_i becomes field_i_0 below. Use here grad_out_i from below.
    - ...

    ### instructions_rest ###
    - STORE_TEMPORARY

    - RETRIEVE_TEMPORARY
    - MULTIPLY_WEIGHT
    - ADD_OUTPUT_U  <-- field_i becomes grad_out_i above. Use here field_i_0 from above.

    - RETRIEVE_TEMPORARY
    - MULTIPLY_WEIGHT
    - ADD_OUTPUT_U
    - ...
    - STOP_EXECUTION

    More complicated orders of mixing store/retrieve temporary with multiply_u are not supported.
    """
    d_instructions = []

    instructions_split = split_instructions(instructions)

    for instruction_block in instructions_split:
        instructions_init, instructions_fields, instructions_rest = split_instruction_block(instruction_block)

        # instructions_rest -> aufspalten in einzelne weight_and_output
        weight_and_output_list = extract_weight_and_output_blocks(instructions_rest)

        for weight_and_output in weight_and_output_list:
            # Last line defines ADD_OUTPUT_U
            output_instruction, output_field, output_field_pos, output_field_i = weight_and_output[-1]
            if output_instruction != ADD_OUTPUT_U:
                # Last entry should be output
                raise ValueError("calculate_derivative_uw: unexpected weight_and_output", weight_and_output)
            
            # Outputfield defines the new grad_out_i
            grad_out_i = output_field_i

            # Apply the derivatives on the U fields
            for k, (instruction, field, field_pos, field_i) in enumerate(instructions_fields):
                if field == field_entry:
                    # Take the derivative

                    # This field is replaced by grad_out with grad_out_i from above.
                    # Its current field_i defines the new field_i_0 for the new output
                    field_i_0 = field_i

                    # Add initial instructions
                    d_instructions += instructions_init

                    if instruction == MULTIPLY_U:
                        # (u_0 * u_1 * ... u_{k-1}))^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger
                        # = u_{k-1}^\dagger * ... u_1^\dagger * u_0^\dagger * grad_out * u_{n-1}^\dagger * ... * u_{k+1}
                        #
                        # Positions are:
                        # x_0, x_1, ... x_{k-1}, x(grad_out), x_{k+1}, ..., x_{n-1}
                        # where x(grad_out) is instead of the original link at x_k.
                        # Derivative gives 
                        for m in reversed(range(k)):
                            instruction_2, field_2, field_pos_2, field_i_2 = instructions_fields[m]
                            field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                            instruction_2_dagger = MULTIPLY_U_DAGGER if instruction_2 == MULTIPLY_U else MULTIPLY_U
                            d_instructions.append((instruction_2_dagger, field_2, field_pos_2, field_i_2))
                        
                        grad_out_pos = negative_position(field_pos, dims)
                        d_instructions.append((MULTIPLY_U, grad_out_entry, grad_out_pos, grad_out_i))
                        
                        for m in reversed(range(k + 1, len(instructions_fields))):
                            instruction_2, field_2, field_pos_2, field_i_2 = instructions_fields[m]
                            field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                            instruction_2_dagger = MULTIPLY_U_DAGGER if instruction_2 == MULTIPLY_U else MULTIPLY_U
                            d_instructions.append((instruction_2_dagger, field_2, field_pos_2, field_i_2))

                    elif instruction == MULTIPLY_U_DAGGER:
                        # ((u_0 * u_1 * ... u_{k-1})^\dagger * grad_out * (u_{k+1} * ... * u_{n-1})^\dagger)^\dagger
                        # = u_{k+1} * ... * u_{n-1} * grad_out^\dagger * u_0 * u_1 * ... u_{k-1}
                        for m in range(k + 1, len(instructions_fields)):
                            instruction_2, field_2, field_pos_2, field_i_2 = instructions_fields[m]
                            field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                            d_instructions.append((instruction_2, field_2, field_pos_2, field_i_2))
                        
                        grad_out_pos = negative_position(field_pos, dims)
                        d_instructions.append((MULTIPLY_U_DAGGER, grad_out_entry, grad_out_pos, grad_out_i))
                        
                        for m in range(k):
                            instruction_2, field_2, field_pos_2, field_i_2 = instructions_fields[m]
                            field_pos_2 = subtract_positions(field_pos_2, field_pos, dims)
                            d_instructions.append((instruction_2, field_2, field_pos_2, field_i_2))

                    else:
                        raise ValueError("calculate_derivative_uw: unexpected instruction", (instruction, field, field_pos, field_i))
                    
                    # Copy MULTIPLY_WEIGHT command if it exists
                    if weight_and_output[0][0] == MULTIPLY_WEIGHT:
                        d_instructions.append(weight_and_output[0])

                    # Copy ADD_OUTPUT_U command, but replace last entry output_field_i by field_i_0
                    d_instructions.append((ADD_OUTPUT_U, output_field, output_field_pos, field_i_0))

                    # Copy STOP_EXECUTION command from last entry:
                    if instructions_rest[-1][0] == STOP_EXECUTION:
                        d_instructions.append(instructions_rest[-1])
                    else:
                        raise ValueError("calculate_derivative_uw: unexpected STOP instruction", instructions_rest[-1])


    return d_instructions

### TODO: The derivative with respect to weights has to be worked out properly

# def calculate_derivative_weight(instructions, field_entry, grad_out_entry):
#     """Calculate derivative with respect to weight.

#     For an entry of the form
#     u_0 * u_1 * ... * u_{n-1} * weight[field_1, i_1]

#     the backpropagation gradient is given by
#     tr(grad_out[j] * (u_0 * ... * u_{n-1})^\dagger)
#     = tr(grad_out[j] * u_{n-1}^\dagger * ... * u_1^\dagger * u_0^\dagger)
    
#     ++++++++)

#     Original expression is of the form:
#     U_temp = U * U * U * U; U_temp * weight[field_1, i_1], U_temp * weight[field_2, i_2], U_temp * weight[field_3, i_3], ...

#     -> For example, if field_1 and field_3 agree with field_entry, but field_2 is different (field_1 = field_3 = field_entry != field_2)
#        then field_2 entry is dropped, and the others are changed to grad_out_entry:
    
#     The derivative expression becomes (with field_2 dropped):
#     U_temp = U * U * U * U; U_temp * weight[gradout_entry, i_1], U_temp * weight[gradout_entry, i_3, ...

#     Typical instruction block of instruction_list:
#     - RESET_VALUES
#     - MULTIPLY_U
#     - MULTIPLY_U
#     - MULTIPLY_U_DAGGER
#     - ...
#     - STORE_TEMPORARY
    
#     - RETRIEVE_TEMPORARY
#     - MULTIPLY_WEIGHT --> Simply replace every occurence of this by grad_out if it exists, otherwise drop from instruction block
#     - ADD_OUTPUT_U

#     - RETRIEVE_TEMPORARY
#     - MULTIPLY_WEIGHT
#     - ADD_OUTPUT_U
#     - 
#     - STOP_EXECUTION

#     WARNING: We assume without checking that every weight only appears once in each instruction block.
#     """
#     # Calculate derivative for function
#     d_instructions = []

#     skip_instructions = False

#     for instruction_line in instructions:
#         instruction, field, field_pos, field_i = instruction_line
#         if instruction == MULTIPLY_WEIGHT:
#             if field == field_entry:
#                 # Replace this weight by the grad_out value:
#                 field = grad_out_entry
#             else:
#                 skip_instructions = True
        
#         if skip_instructions:        
#             if instruction == ADD_OUTPUT_U:
#                 pass # keep skipping
#             elif instruction == RETRIEVE_TEMPORARY:
#                 # Ok, can continue with next weight entry, but skip this entry
#                 skip_instructions = False
#             elif instruction == STOP_EXECUTION:
#                 # End of thread, so add this
#                 skip_instructions = False
#                 d_instructions.append((instruction, field, field_pos, field_i))
#             else:
#                 # don't expect any other command to be skipped.
#                 raise ValueError('calculate_derivative_weight: unexpected instruction to be skipped:', instruction_line)
#         else:
#             d_instructions.append((instruction, field, field_pos, field_i))
    
#     return d_instructions


###############################################################################
## TOP LEVEL CLASSES

