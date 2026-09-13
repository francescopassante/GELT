# Implementation of L-CNN layers using a generic WilsonLine class
import torch
import lge_cnn.nn.layers as layers

class LWilsonLine(torch.nn.Module):
    """
        A module that computes products of Wilson lines from a link configuration
    """

    def __init__(self, dims, nc, line_list=None):
        super(LWilsonLine, self).__init__()
        self.dims = dims
        self.nc = nc
        self.line_list = None
        self.shifted_indices_list = None

        if line_list:
            self.init_line_list(line_list)

    def init_line_list(self, line_list):
        self.line_list = line_list
        self.shifted_indices_list = create_lines(line_list, self.dims)

    def get_number_of_channels(self):
        return len(self.line_list)

    def forward(self, u):
        # u layout: (batch_dim, lattice, direction, matrix structure (2), real/imag)

        wilson_lines = []
        for list in self.shifted_indices_list:
            first_element = True
            for (pos_shift, mu, dagger) in list:
                u_mu = u[:,:,mu]
                u_mu = shift_multiple(u_mu, pos_shift, self.dims)
                if dagger > 0:
                    u_mu = dagger_ri(u_mu)
                if first_element:
                    u_product = u_mu
                    first_element = False
                else:
                    u_product = matmul_ri(u_product, u_mu)
            wilson_lines.append(u_product)

        return torch.stack(wilson_lines, dim=2)

    def update_dims(self, dims):
        self.dims = dims   

class PlaqWL(LWilsonLine):
    """
        A module that computes plaquettes from a link configuration
    """

    def __init__(self, dims, nc, orientation=-1):
        super(PlaqWL, self).__init__(dims, nc)
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


class PlaqCloverWL(LWilsonLine):
    """
        A module that computes all clover plaquettes from a link configuration
    """
    def __init__(self, dims, nc):
        super(PlaqCloverWL, self).__init__(dims, nc)

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
    
class RectangleWL(LWilsonLine):
    """
        A module that computes all rectangles from a link configuration
    """
    def __init__(self, dims, nc):
        super(RectangleWL, self).__init__(dims, nc)

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

class ParallelogramWL(LWilsonLine):
    """
        A module that computes all parallelograms from a link configuration
    """
    def __init__(self, dims, nc):
        super(ParallelogramWL, self).__init__(dims, nc)
        
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

class ChairWL(LWilsonLine):
    """
        A module that computes all chairs from a link configuration
    """
    def __init__(self, dims, nc):
        super(ChairWL, self).__init__(dims, nc)
        
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

# dagger on real/imag objects
def dagger_ri(u):
    u = torch.view_as_complex(u)
    u = u.conj()
    u = torch.transpose(u, -1, -2)
    u = u.resolve_conj()
    u = torch.view_as_real(u)
    return u

# matmul for real/imag objects
def matmul_ri(u1, u2):
    # u = complex_einsum('bxij, bxjk', u1, u2)
    u1 = torch.view_as_complex(u1)
    u2 = torch.view_as_complex(u2)
    u = torch.matmul(u1, u2)
    u = torch.view_as_real(u)
    return u

def shift_multiple(u, pos_shift, dims):
    for i in range(len(dims)):
        p = pos_shift[i]
        if p != 0:
            # roll along lattice dimension i in opposite direction to obtain value at position x_shifted
            u = layers.shift(u, i, -p, dims)
    return u

def create_lines(line_list, dims):
    shifted_indices_list = []
    for line in line_list:
        shifted_indices = get_index_shifts(line, dims)
        shifted_indices_list.append(shifted_indices)
    return shifted_indices_list

def get_index_shifts(line, dims):
    p = (0, ) * len(dims) # starting position
    object_list = []
    for ii in line:
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
        object_list.append((p0, i, dagger))
        p = shift_position(p, i, oi, dims)

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