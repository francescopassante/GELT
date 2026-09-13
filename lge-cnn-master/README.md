# Lattice Gauge Equivariant Convolutional Neural Networks (L-CNNs) and fixed-point (FP) actions 

This is the code repository for 

> 📚 M. Favoni, A. Ipp, D. I. Müller and D. Schuh , *Lattice Gauge Equivariant Convolutional Neural Networks*, [Phys.Rev.Lett. 128 (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.032003), [arXiv:2012.12901](https://arxiv.org/abs/2012.12901).

For code specific to the original 2022 publication, please see the `prl_2022` branch.

The code has since been extended to model, train and simulate SU(3) fixed-point actions in collaboration with K. Holland and U. Wenger. The main papers are

> 📚 K. Holland, A. Ipp, D.I. Müller and U. Wenger, *Machine learning a fixed point action for SU(3) gauge theory with a gauge equivariant convolutional neural network*, [Phys.Rev.D 110 (2024)](https://doi.org/10.1103/PhysRevD.110.074502), [arXiv:2401.06481](https://arxiv.org/abs/2401.06481).

> 📚 K. Holland, A. Ipp, D.I. Müller and U. Wenger, *Machine-learned RG-improved gauge actions and classically perfect gradient flows*, under review (2025), [arXiv:2504.15870](https://arxiv.org/abs/2504.15870)

Please cite the appropriate works if you use or adapt this code.

## Code overview

The main components of this repository are

* `lge_cnn.nn` Models, layers and dataset classes for L-CNNs
* `lge_cnn.ym` A basic SU(2) and SU(3) Yang-Mills simulation code 
* `fpaction` Modelling, training and testing of fixed-point actions
* `hmc` Monte-Carlo simulation of fixed-point actions using Hybrid Monte Carlo (HMC)

## Conda environment

To run this code, please install and activate one of the conda environments defined in the yml files.

For example:
```shell
conda env create -f environment_fpaction2.yml
conda activate fpaction2
```

Depending on your system, drivers and hardware you might need some adjustments to the environments.

## Running simulations with FP actions

Pre-trained FP action models can be found in the `fpaction` folder.
* `fpaction/instanton_finetuned_model.ckpt`: This model has been used in *arXiv:2401.06481*.
* `fpaction/finetuned_model_v01.ckpt` An updated model used for HMC simulations and gradient flow for *arXiv:2504.15870*.

These models can be used in HMC simulations using the `hmc` package, specifically `hmc/hmc_cmd.py` and `hmc/hmc_cmd_v2.py`. The latter is recommeneded due to reliability and ease of use. 

Here is an example script for a basic Monte Carlo simulation including measurements of plaquettes, Polyakov loops and gradient flow observables. Checkpoint configurations are saved to easily pause and continue simulations.

```bash
python hmc_cmd_v2.py --output_path                    ../example_output/ \
                     --model_input_path               ../fpaction/finetuned_model_v01.ckpt \
                     --l                              4 \
                     --t                              4 \
                     --b                              2.50 \
                     --hmc_seed                       777 \
                     --hmc_tau                        3.0 \
                     --hmc_steps                      24 \
                     --hmc_num_thermalization         200 \
                     --num_measurements               1000 \
                     --hmc_keep_cfg_checkpoints       3 \
                     --measurement_skip               3 \
                     --keep_measurement_checkpoints   3 \
                     --measure_action                 \
                     --measure_plaquette              \
                     --measure_polyakov               \
                     --gf_flow_model                  same \
                     --gf_energy_model                same \
                     --gf_flow_time                   3.0 \
                     --gf_timestep                    0.01 \
                     --auto_continue                  \
```

Please run
```bash
python hmc_cmd_v2.py --help
```
for more options.

## Data analysis

This code also includes some basic data analysis tools such as bootstrap for gradient flow quanitites in `hmc/hmc.py`, e.g. `scale_setting_bootstrap()`.  

## Contributors
*(in alphabetical order)*

Liane Backfried, Matteo Favoni, Kieran Holland, Andreas Ipp, David I. Müller, Thomas Ranner, Daniel Schuh, Urs Wenger
