# GELT codebase summary - man-made

GELT is a gauge equivariant lattice transformer architecture for regression/classification tasks in lattice gauge theories.

the codebase has many independent parts:
- ML models (blocks_rope.py, blocks_bias.py, cnn_baseline.py)
- LGT utilities (lattice.py)
- Sampling algorithms (sampler.py)
- scripts (train, check invariance, ...)
- tests (AI-made)

### lattice.py
lattice.py is the file that contains LGT utilities, such as an abstract gaugegroup class, Z2, SU(N) concrete implementations, plaquette tensor generation, target generation like action, rectangular wilson loop, topological charge density.
It also stores the code to compute the parallel transport tensor of a lattice configuration. Done via dynamical programming, the parallel transport from x to n is simplified as U_mu(x) * transport from x+mu to n, where x+mu is closer to n than x.
Also implements gauge transformation on a link configuration given a gauge transformation Omega(x).

### sampler.py
sampler.py has all the sampling logic. there is a metropolis_sweep algorithm with the classic propose-accept-reject logic, the proposal is routed by a _PROPOSAL_FN dict. e.g. Z2 has its z2 proposal function (U -> -U), SU2 has its SU2 (U -> V@U with V in SU2, close to identity). We also have a staple sum function to compute the next action given a proposal by just considering its change wrt the previous action, much less computation, exact.

Then we have two (as of now) ensembles (the actual config-generating loop). haar_ensemble generates random links (wrapper of lattice.py random_links), mcmc_ensemble actually uses the MC sampling. The algorithm is routed by a _SWEEP_FN dict. As of now we just have metropolis so both Z2 and SU2 default to metropolis, and inside metropolis they have their own proposal func.

### data.py
data.py is the data generating logic. build_plaquette_datasets is the main function. It calls the specified sampler to generate the configs, then builds the plaquette tensors, the specified target and the transport tensor for each of the configs, then calls the split() function to split them in train/val/test. has the option to save to disk the three datasets

### blocks.py
blocks_rope.py and blocks_bias.py contain essentially the same GELT architecture but with different positional encodings.
blocks_rope uses RoPE while blocks_bias uses a convolutional bias. Let's go through blocks_rope.py because 99% of the code is the same.

GELT module is the full model. takes L, D, R, n_head, n_layers, d_qkv, d_model, gate, mlp_hidden, mlp_dropout, reduction and various init scales.

First: the input plaquettes [B, D(D-1)/2, *L, nc, nc] pass through a linear model that goes from D(D-1)/2 channels to d_model channels -> [B, d_model, *L, nc, nc]. 
At init, the d_model x D(D-1)/2 matrix is the identity in the first D(D-1)/2 x D(D-1)/2 block, so that 
(plaq_1, ..., plaq_n) -> (0, ..., 0, plaq_1, ..., plaq_n)

Then we apply the n_layers GEMHSA blocks.

Then we take the trace of the final layer channels, concatenate real and imaginary parts and pass them through a per-site MLP
[B, *L, 2 d_model]

Now let's look inside the GEMHSA layer:

- We take in a [B, d_model, *L, nc, nc] input, we augment appending daggers and identity -> [B, 2 d_model+1, *L, nc, nc].
- for each head we compute Q,Q_v,K,V projections, essentially d_qkv complex linear combinations of the channels for each head -> 4 objects of shape [B, h, d_qkv, *L, nc, nc]
- we transport the K and V from neighbors using T_(n->x)
- we compute the attention score as s_xn = ReTr(Q_i(x) @ R_deltax(K_i(n->x))) where R_deltax is the RoPE, then softmax for alpha_xn
- we compute value as sum_n alpha_xn Q_v(x).dagger() @ V(n->x)























