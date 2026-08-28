# Scale Is Not Color: Attribute-Dependent Limits of Causal Subspace Analysis in Vision Transformers

## Overview

This repository studies how causal subspace methods reveal representations in Vision Transformers. We replace color with object scale and reproduce the original analysis pipeline across multiple ViT-B models and training regimes. Shape yields the expected causal signature, while scale does not, despite the models solving the scale-based relational task; this shows that a failed intervention can reflect representational format rather than missing information.

## Scope of This Work

This work replace the color-based experimental attribute with scale. For efficiency and convenience, some directory names, route names, command-line values, and related identifiers remain associated with `color`; however, the code operating on those paths and values uses **scale** in the current experiments. These names are retained for compatibility with the existing pipeline and should not be interpreted as indicating that color is the manipulated attribute here.

Some experiment configurations are stored in the `.local_wand_sweep.json` file. This file is used for convenience and allow the experiments to be run without requiring Weights & Biases (`wandb`). The original repository and paper remain the source of attribution for the inherited methodology and implementation.

## Reproducibility

The complete procedure required to reproduce the experiments reported in this work is documented in [`pipeline.md`](pipeline.md). It includes stimulus generation, model training, Distributed Alignment Search (DAS), linear-probing interventions, and result analysis.

### Data Generation
`data.py` is used to generate training data for discrimination and RMTS tasks. Once these are generated, run `data.py --create_das` to generate DAS counterfactual datasets. Add `--match_to_sample` to generate RMTS counterfactual datasets.

### Model Training
`train.py` is used to model training. Add `--auxiliary_loss` to induce disentangled object representations.

### Perceptual Stage 
`das.py` runs DAS over all layers of a specified model to find a specified subspace. 

### Relational Stage
`das.py` also returns novel representation analysis results.

`linear_probe_intervention.py` trains linear probes over RMTS models and runs the additive intervention on them.

The commands and parameter tables in [`pipeline.md`](pipeline.md) should be treated as the primary replication guide for this work.

### Analyses
Graphing files are found in `analysis`

## Dependencies

This repository requires the included local copy of Pyvene. Additionally, it uses the fork of TransformerLens specified by the original project: `https://github.com/mlepori1/TransformerLens.git`.

When creating the virtual environment, select CUDA and PyTorch versions compatible with the available GPU hardware.

---

This repository is a fork of the original project:

- Original repository: <https://github.com/alexatartaglini/relational-circuits>

The current work replicates the original experiments while varying **scale** instead of **color** as the object attribute under investigation.


