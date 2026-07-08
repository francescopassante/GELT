# GELT: a Gauge-Equivariant Lattice Transformer for Lattice Gauge Theory



## Abstract

Gauge equivariant convolutional neural networks have shown that enforcing exact lattice gauge symmetry in a neural network significantly improves regression accuracy on gauge invariant observables. However, their convolutional kernels stay fixed after training and cannot adapt to configuration-dependent features. We introduce GELT (Gauge Equivariant Lattice Transformer), a gauge equivariant attention-based encoder network for lattice gauge theories that exploits the content-dependent attention mechanism to focus on the most physically relevant input features.

GELT employs gauge invariant attention scores together with a gauge equivariant matrix bilinear value path. Keys and values are parallel transported from neighboring lattice sites within a L1-ball along shortest lattice paths. We inject a geometrical prior in the attention weights with rotary positional encoding (RoPE).

We test GELT on various regression targets involving physical observables and compare its performance against existing gauge-equivariant architectures. We also investigate the interpretability of the attention filters, studying whether attention localizes in topologically rich regions and whether the attention range correlates with physical correlation length.


