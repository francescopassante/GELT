import matplotlib.pyplot as plt


def visualize_lattice(lattice, title=None):
    assert lattice.D == 2 and hasattr(lattice, "links")

    L = lattice.L
    fig, ax = plt.subplots(figsize=(max(4, L), max(4, L)))

    for x in range(L):
        for y in range(L):
            for direction, (dx, dy) in enumerate([(1, 0), (0, 1)]):
                val = lattice.links[x, y, direction].operator
                color = "tab:green" if val > 0 else "tab:red"
                ax.plot([x, x + dx], [y, y + dy], color=color, lw=2.5)

    for x in range(L):
        for y in range(L):
            ax.plot(x, y, "ko", ms=7, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, L)
    ax.set_ylim(-0.5, L)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


from lattice import Lattice

lat = Lattice(L=5)
lat.initialize_random_links()
visualize_lattice(lat)
