from abc import ABC, abstractmethod

import numpy as np


class GaugeGroup(ABC):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def multiply(U, V):
        pass

    @abstractmethod
    def inverse(U):
        pass


class Z2(GaugeGroup):
    def __init__(
        self,
    ):
        super().__init__("Z2")

    def multiply(self, *operators):
        prod = 1
        for op in operators:
            prod *= op
        return prod

    def inverse(self, U):
        return U


class Site:
    def __init__(self, position, features=None):
        self.features = features
        self.position = position


class Link:
    def __init__(self, U, position, direction, gaugegroup):
        self.operator = U
        self.position = position
        self.direction = direction
        self.gaugegroup = gaugegroup
        self.D = len(position)

    def inverse(self):
        """Returns the inverse link"""

        # Direction is encoded as: mu=0,...,d-1 -> 'positive' along mu; mu=d, ..., 2d-1 -> 'negative' along mu%d
        axis = self.direction % self.D
        dir_vec = np.zeros(self.D, dtype=np.int64)
        dir_vec[axis] = 1
        inverse_direction = (
            self.direction + self.D
            if self.direction < self.D
            else self.direction - self.D
        )

        return Link(
            U=self.gaugegroup.inverse(self.operator),
            position=np.array(self.position) + dir_vec,
            direction=inverse_direction,
            gaugegroup=self.gaugegroup,
        )

    def __mul__(self, other):
        return Link(
            U=self.gaugegroup.multiply(self.operator, other.operator),
            position=self.position,
            direction=None,
            gaugegroup=self.gaugegroup,
        )

    def __str__(self):
        return f"Link at position: {self.position}, direction: {self.direction} with value: {self.operator}"


class Plaquette:
    def __init__(self, P, position, dir1, dir2):
        self.P = P
        self.position = position
        self.dir1 = dir1
        self.dir2 = dir2

    @classmethod
    def from_links(cls, link1, link2, link3, link4, dir1, dir2):
        P = link1.gaugegroup.multiply(
            link1.operator, link2.operator, link3.operator, link4.operator
        )
        return cls(P, link1.position, dir1, dir2)


class Lattice:
    """Class to manipulate lattices"""

    def __init__(self, L, D=2, gaugegroup=None):
        if not isinstance(D, int):
            raise TypeError("D must be an integer")
        if not isinstance(L, int):
            raise TypeError("L must be an integer")
        assert D > 0 and L > 0
        if gaugegroup is None:
            gaugegroup = Z2()
        self.L = L
        self.D = D
        self.gaugegroup = gaugegroup

        def _initialize_sites(L):
            # Initialize lattice points with no site-features
            lattice_sites = np.empty(shape=(L,) * self.D, dtype=Site)
            for coord in np.ndindex(lattice_sites.shape):
                lattice_sites[*coord] = Site(position=coord)
            return lattice_sites

        self.lattice_sites = _initialize_sites(self.L)

    def initialize_random_links(self):
        """Initialize all links to be either +1 or -1 with 50% chance. (Haar)"""
        self.links = np.empty(
            shape=(self.L,) * self.D + (self.D,), dtype=Link
        )  # [L, L, ..., L, D]
        for coord in np.ndindex(self.lattice_sites.shape):
            for i in range(self.D):
                self.links[*coord, i] = Link(
                    U=(1 if np.random.random() < 0.5 else -1),
                    position=coord,
                    direction=i,
                    gaugegroup=self.gaugegroup,
                )

    def get_link(self, position, direction) -> Link:
        """Return link at specified position and direction"""
        assert len(position) == self.D
        assert sum([0 <= position[i] < self.L for i in range(self.D)]) == self.D
        assert 0 <= direction < self.D

        return self.links[*position, direction]

    def get_site(self, position):
        """Return site at specified position"""
        assert len(position) == self.D
        assert sum([0 <= position[i] < self.L for i in range(self.D)]) == self.D

        return self.lattice_sites[*position]

    def plaquette(self, position, mu, nu):
        """Return the plaquette P_munu at position position and directions mu, nu"""
        assert len(position) == self.D
        assert sum([0 <= position[i] < self.L for i in range(self.D)]) == self.D
        assert 0 <= mu < self.D
        assert 0 <= nu < self.D
        assert mu != nu

        position = np.array(position)

        # Build the basis vectors mu^hat, nu^hat
        mu_vec = np.zeros(self.D, dtype=np.int64)
        nu_vec = np.zeros(self.D, dtype=np.int64)
        mu_vec[mu] = 1
        nu_vec[nu] = 1
        return Plaquette.from_links(
            self.links[*position, mu],
            self.links[*((position + mu_vec) % self.L), nu],
            self.links[*((position + nu_vec) % self.L), mu].inverse(),
            self.links[*position, nu].inverse(),
            mu,
            nu,
        )

    def all_plaquettes(self, position):
        """Return all the plaquettes at position position"""
        assert len(position) == self.D
        assert sum([0 <= position[i] < self.L for i in range(self.D)]) == self.D

        plaquettes = np.empty(shape=(self.D, self.D), dtype=Plaquette)
        # only compute P_munu with nu > mu, the remaining ones are the hermitians
        for mu, nu in np.ndindex(plaquettes.shape):
            if nu > mu:
                plaquettes[mu, nu] = self.plaquette(position, mu, nu)
            elif nu < mu:
                plaquettes[mu, nu] = plaquettes[nu, mu]
            else:  # mu = nu, None
                plaquettes[mu, nu] = None

        return plaquettes
