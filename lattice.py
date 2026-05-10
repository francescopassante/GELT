from abc import ABC, abstractmethod

import numpy as np
import torch


class GaugeGroup(ABC):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def multiply(self, *operators):
        pass

    @abstractmethod
    def inverse(self, U):
        pass


class Z2(GaugeGroup):
    def __init__(self):
        super().__init__("Z2")

    def multiply(self, *operators):
        prod = 1
        for op in operators:
            prod *= op
        return prod

    def inverse(self, U):
        return U


class Site:
    def __init__(self, features=None):
        self.features = features


class Link:
    def __init__(self, U, direction, gaugegroup, D):
        self.operator = U
        self.direction = direction
        self.gaugegroup = gaugegroup
        self.D = D

    def inverse(self):
        """Returns the inverse link.

        Direction encoding: mu=0,...,D-1 forward along axis mu; mu=D,...,2D-1 backward along axis mu%D.
        """
        inverse_direction = (
            self.direction + self.D
            if self.direction < self.D
            else self.direction - self.D
        )
        return Link(
            U=self.gaugegroup.inverse(self.operator),
            direction=inverse_direction,
            gaugegroup=self.gaugegroup,
            D=self.D,
        )

    def __mul__(self, other):
        return Link(
            U=self.gaugegroup.multiply(self.operator, other.operator),
            direction=None,
            gaugegroup=self.gaugegroup,
            D=self.D,
        )

    def __str__(self):
        return f"Link direction: {self.direction}, value: {self.operator}"


class Plaquette:
    def __init__(self, P, position, dir1, dir2):
        self.P = P
        self.position = position
        self.dir1 = dir1
        self.dir2 = dir2

    @classmethod
    def from_links(cls, link1, link2, link3, link4, position, dir1, dir2):
        P = link1.gaugegroup.multiply(
            link1.operator, link2.operator, link3.operator, link4.operator
        )
        return cls(P, position, dir1, dir2)


class Lattice:
    """Class to manipulate lattices"""

    def __init__(self, L, D=2, gaugegroup=None):
        assert D > 0 and L > 0
        if gaugegroup is None:
            gaugegroup = Z2()
        self.L = L
        self.D = D
        self.gaugegroup = gaugegroup

        self.lattice_sites = np.empty(shape=(L,) * D, dtype=Site)
        for coord in np.ndindex(self.lattice_sites.shape):
            self.lattice_sites[*coord] = Site()

    def initialize_random_links(self):
        """Initialize all links to be either +1 or -1 with 50% chance. (Haar)"""
        self.links = np.empty(shape=(self.L,) * self.D + (self.D,), dtype=Link)
        for coord in np.ndindex(self.lattice_sites.shape):
            for i in range(self.D):
                self.links[*coord, i] = Link(
                    U=(1 if np.random.random() < 0.5 else -1),
                    direction=i,
                    gaugegroup=self.gaugegroup,
                    D=self.D,
                )
        return self

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

        mu_vec = np.zeros(self.D, dtype=np.int64)
        nu_vec = np.zeros(self.D, dtype=np.int64)
        mu_vec[mu] = 1
        nu_vec[nu] = 1
        return Plaquette.from_links(
            self.links[*position, mu],
            self.links[*((position + mu_vec) % self.L), nu],
            self.links[*((position + nu_vec) % self.L), mu].inverse(),
            self.links[*position, nu].inverse(),
            position,
            mu,
            nu,
        )

    def plaquette_tensor(self):
        """Return the (*L, D*(D-1)//2) tensor of unique plaquette values per site."""
        pairs = [(mu, nu) for mu in range(self.D) for nu in range(mu + 1, self.D)]
        tensor = torch.zeros((len(pairs),) + (self.L,) * self.D)
        for coord in np.ndindex(self.lattice_sites.shape):
            for k, (mu, nu) in enumerate(pairs):
                tensor[k, *coord] = self.plaquette(coord, mu, nu).P
        return tensor

    def link_tensor(self):
        """Return the (*L, D) tensor of D links fr each site"""
        tensor = torch.zeros((self.D,) + (self.L,) * self.D)
        for coord in np.ndindex(self.lattice_sites.shape):
            for mu in range(self.D):
                tensor[mu, *coord] = self.links[*coord, mu]
        return tensor

    def action(self):
        plaq = self.plaquette_tensor()
        n_plaq = self.L**self.D * self.D * (self.D - 1) // 2
        # equivalent to sum_p (1 - P_p)
        return n_plaq - torch.sum(plaq)
