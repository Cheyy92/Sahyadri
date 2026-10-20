# Galaxy Halo Occupation & Posterior Building: VVF Fisher Forecast

## Project Overview

This computational cosmology pipeline prepares data vectors, evaluates numerical derivatives, and executes Fisher matrix forecasts to constrain cosmological parameters ($\Omega_m$, $\sigma_8$) and Halo Occupation Distribution (HOD) parameters ($\alpha$, $\kappa$, $\sigma_{\log M}$, $\log M_1$, $\log M_{cut}$).

The primary objective is to quantify the constraint power provided by the Voronoi Volume Function (VVF) percentiles when combined with standard two-point statistics, demonstrating how VVF breaks parameter degeneracies and shrinks posterior contours.

**Analyzed Data Vectors:**

* $w_p$: Projected two-point correlation function
* $n_g$: Galaxy number density
* VVF: Voronoi Volume Function percentiles

## Pipeline Architecture

1. **Covariance Matrix:** Computes Jackknife covariance ($C$) and precision matrices ($C^{-1}$) for VVF percentiles using spatial sub-volumes (leave-one-out realizations).
2. **Numerical Derivatives:** Evaluates 1D derivatives of the VVF with respect to the five HOD parameters via finite differences. Gradients are calculated using parameter step variations ($\pm 2.5\%$ and $\pm 5\%$). For deeper analytical context on these calculations, refer to the Derivatives of VVF file.
3. **Fisher Matrix:** Constructs the joint Jacobian matrix ($D$) to compute the Fisher matrix:

$$F = D^T C^{-1} D$$


4. **Visualizations:** Generates correlation matrix heatmaps and posterior corner plots to compare contour sizes and orientations across three target cases: $w_p + n_g$, $\text{VVF} + n_g$, and $w_p + n_g + \text{VVF}$.

## Directory Structure

```text
.
├── BGS/                      # Bright Galaxy Sample selection cuts
├── Derivatives/              # Finite difference derivatives of VVF
│   ├── derivatives/          # Calculated derivative tables and validation plots
│   ├── minus_2_5/            # Galaxy populations measurements for -2.5% HOD variations
│   ├── minus_5/              # Galaxy populations for -5.0% HOD variations
│   ├── plus_2_5/             # Galaxy populations for +2.5% HOD variations
│   |── plus_5/               # Galaxy populations for +5.0% HOD variations
│   ├── var_As_m2048          # Galaxy populations for As negative perturbation
│   ├── var_As_p2048          # Galaxy populations for As positive perturbation
│   ├── var_h_m2048           # Galaxy populations for h negative perturbation
│   ├── var_h_p2048           # Galaxy populations for h positive perturbation
│   ├── var_ns_m2048
│   ├── var_ns_p2048
│   ├── var_Ok_m2048
│   ├── var_Ok_p2048
│   ├── var_Om_m2048
│   ├── var_Om_p2048
│   ├── var_w_m2048
│   └── var_w_p2048
├── Fisher Forecast/          # Fisher matrix calculation engine & corner plotting
├── JackKnife/                # Jackknife sub-volume sampling & covariance estimation codes
│   ├── Figures/              # self explanatory
│   ├── Notebook/             # Exploration notebooks
│   ├── Papers_to_refer/      # Reference literature
│   ├── real_results/         # Covariance outputs (real space)
│   ├── rsd_results/          # Covariance outputs (redshift space)
│   └── scripts/              # Automated Jackknife scripts
├── modified_poisson/         # Stochastic galaxy assignment & scatter modeling
├── Ran_Fac/                  # Random factor convergence codes
├── sahyadri-sandbox/         # Sahyadri sandbox forked from https://github.com/a-paranjape/sahyadri-sandbox/
│   ├── code/                 # Integrated Rockstar halo finder
│   ├── config/               # Simulation box & job submission configs
│   ├── docs/                 # N-body simulation documentation (Gadget-4)
│   ├── Sahyadri_pegasus/     # Cluster submission routines for Pegasus HPC
│   └── scripts/              # Gadget-4 execution and post-processing scripts
├── SDSS_Sahyadri/            # SDSS galaxy sample mapping interfaces (TUTORIAL)
└── spectrophotogroup/        # Group galaxy processing & cluster_toolkit integration (Forked from Shadab's repo)

```



## Sahyadri Cosmological Variations
The Sahyadri simulation boxes perfectly map to the 6 cosmological parameters required for our joint Fisher matrix forecast. Here is the breakdown of the parameter variations you have available:
* var_Om_p/m: Matter density ($\Omega_m$)
* var_As_p/m: Amplitude of scalar perturbations ($A_s$ or $\sigma_8$)
* var_h_p/m: Hubble parameter ($h$)
* var_ns_p/m: Scalar spectral index ($n_s$)
* var_w_p/m: Dark energy equation of state ($w$)
* var_Ok_p/m: Spatial curvature ($\Omega_k$)
* default2048: Your fiducial cosmology baseline