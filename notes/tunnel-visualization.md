# Neural-Network Geometry of the QCD Vacuum

## The Goal

You already have a neural network that predicts the topological charge

\[
Q \in \mathbb{Z}
\]

from raw lattice gauge configurations.

The cool idea is:

> Use the *internal representation* learned by the network to build a visual map of the QCD vacuum.

If this works, different topological sectors become separate regions ("islands") in latent space.

The final visualization looks like:
- clusters corresponding to different integer topological charges,
- smooth motion inside a sector,
- sudden jumps during tunneling events.

This turns abstract vacuum topology into something you can literally see.

---

# Core Idea

A neural network does not only output a prediction.

Internally it builds a compressed representation of the configuration:

\[
x \rightarrow z \rightarrow Q
\]

where:
- \(x\) = lattice gauge configuration,
- \(z\) = latent representation,
- \(Q\) = predicted topological charge.

The latent vector \(z\) may contain physically meaningful structure.

If topology is truly learned, configurations with the same \(Q\) should organize together in latent space.

---

# What You Need

## Inputs

You need:
- gauge configurations,
- true topological charges,
- your trained model.

Preferably:
- configurations generated sequentially in Monte Carlo time,
- so you can visualize tunneling trajectories.

---

# Step 1 — Extract Latent Vectors

Remove the final classification layer.

Instead of outputting:

\[
Q
\]

extract the last hidden layer:

\[
z \in \mathbb{R}^d
\]

where:
- \(d\) might be 32, 64, 128, etc.

For every configuration:
1. pass the configuration through the network,
2. store the latent vector,
3. store the true \(Q\),
4. store Monte Carlo time index.

Result:

| Configuration | Latent Vector | Q | MC Time |
|---|---|---|---|
| cfg_0001 | z1 | 0 | 1 |
| cfg_0002 | z2 | 0 | 2 |
| cfg_0003 | z3 | 1 | 3 |

---

# Step 2 — Dimensional Reduction

The latent vectors are high-dimensional.

Project them into 2D using:
- UMAP (recommended),
- or t-SNE.

UMAP is usually better because:
- it preserves global structure better,
- it handles trajectories nicely,
- it is faster.

Example:

```python
from umap import UMAP

embedding = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric='euclidean'
).fit_transform(latents)
```

This gives:

\[
z_i \in \mathbb{R}^{128}
\rightarrow
(x_i, y_i) \in \mathbb{R}^2
\]

---

# Step 3 — Build the Vacuum-Island Plot

Create a scatter plot:
- each point = one gauge configuration,
- color = topological charge \(Q\).

Example color scheme:
- blue = \(Q=0\),
- red = \(Q=1\),
- green = \(Q=-1\),
- etc.

If the model learned topology properly, you may see:
- disconnected clusters,
- curved manifolds,
- topological sector separation.

This is the "QCD vacuum geometry" plot.

Example plotting code:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

for q in sorted(set(Q_values)):
    mask = (Q_values == q)

    plt.scatter(
        embedding[mask,0],
        embedding[mask,1],
        label=f"Q={q}",
        s=10,
        alpha=0.7
    )

plt.legend()
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("Latent Geometry of the QCD Vacuum")

plt.show()
```

---

# Step 4 — Add Monte Carlo Trajectories

This is the coolest part.

Take configurations in chronological Monte Carlo order and connect them with lines.

Now the system becomes dynamical.

You may see:
- wandering motion inside one sector,
- sudden jumps between islands,
- tunneling events becoming visible.

This is literally vacuum tunneling visualized geometrically.

Example:

```python
plt.figure(figsize=(8,8))

plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=Q_values,
    s=10
)

plt.plot(
    embedding[:,0],
    embedding[:,1],
    alpha=0.3,
    linewidth=1
)

plt.title("Monte Carlo Trajectory Through Vacuum Sectors")

plt.show()
```

---

# Step 5 — Animate It

Create an animation over Monte Carlo time.

The point moves through latent space:
- drifting smoothly,
- then suddenly jumping sectors.

This creates a striking visual narrative:
- the vacuum explores one topological sector,
- tunnels,
- enters another.

Minimal example:

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8,8))

sc = ax.scatter([], [])

ax.set_xlim(embedding[:,0].min(), embedding[:,0].max())
ax.set_ylim(embedding[:,1].min(), embedding[:,1].max())

def update(frame):

    sc.set_offsets(embedding[:frame])

    sc.set_array(Q_values[:frame])

    return sc,

ani = FuncAnimation(
    fig,
    update,
    frames=len(embedding),
    interval=30
)

plt.show()
```

---

# What Would Be Amazing

## 1. Clean sector separation

Different \(Q\) values form distinct clusters automatically.

This means the network internally encoded topology geometrically.

---

## 2. Smooth tunneling bridges

Instead of abrupt random jumps, trajectories pass through narrow transition regions.

That suggests:
- the latent space learned physically meaningful interpolation paths,
- possibly related to instanton transitions.

---

## 3. Structure inside each island

Maybe:
- different instanton sizes,
- action density,
- temperature,
- confinement phase,

organize internally within each sector.

That would mean the network learned *more than topology*.

---

# Why This Is Cool

This visualization turns:
- abstract topology,
- gauge vacua,
- tunneling,

into something geometric and intuitive.

Instead of:
> "topological charge is an integer label"

you get:
> "the QCD vacuum is a landscape with connected regions and tunneling pathways."

That is visually and conceptually powerful even for people outside lattice QCD.

---

# Optional Extensions

## Saliency maps

Highlight which spacetime regions the network uses most strongly.

Compare with:
- instantons,
- topological density after gradient flow.

Potential result:
> the network rediscovered instantons by itself.

---

## Compare before/after gradient flow

Check whether:
- raw configurations already separate,
- or separation emerges only after smoothing.

If raw fields work:
> the network learned renormalized topology directly.

---

## Temperature dependence

Build separate latent maps for:
- low temperature,
- near \(T_c\),
- high temperature.

Maybe topology islands deform or disappear.

Could look extremely cool.

---

# Suggested Final Figure

A publication-quality figure could contain:

1. UMAP latent-space scatter plot,
2. colors by \(Q\),
3. Monte Carlo trajectories,
4. highlighted tunneling events,
5. optional instanton saliency overlays.

Possible title:

> "Emergent Geometry of the QCD Vacuum Learned by a Neural Network"

or

> "Neural Latent-Space Reconstruction of Topological Vacuum Structure"

---

# Minimal Tech Stack

- PyTorch or TensorFlow,
- UMAP,
- matplotlib,
- optionally Plotly for interactive 3D plots,
- optionally Blender or manim for cinematic animations.

---

# One-Sentence Pitch

> "I trained a neural network to predict topological charge, and its internal representation spontaneously organized the QCD vacuum into geometric topological sectors with visible tunneling trajectories."
