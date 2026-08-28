# Experimental Replication Pipeline

This document describes the complete sequence for stimulus generation, model training, Distributed Alignment Search (DAS), linear-probing interventions, and result analysis. Commands are written once as templates; replace angle-bracketed values using the option tables below.

## 1. Data and Stimulus Generation

### 1.1 Base objects (one-time setup)

```bash
python data.py --create_source --source NOISE_RGB --obj_size 32
```

This creates the 64 base tokens (16 shapes x 4 scales). Use `--source mts` for RMTS stimuli and omit it for discrimination stimuli.

### 1.2 Dataset commands

The dataset options are:

| Dataset | `--source` | `--match_to_sample` | `--compositional` | Patch/object size |
| --- | --- | --- | --- | --- |
| Discrimination, non-compositional | omitted | omitted | omitted | 16 / 32 |
| Discrimination, compositional | omitted | omitted | `32` | 16 / 32 |
| RMTS, non-compositional | `mts` | required | omitted | 16 / 32 |
| RMTS, compositional | `mts` | required | `32` | 16 / 32 |
| Any DINOv2 dataset | as above | as above | as above | 14 / 28 |

For training data, run the following once for each row needed above. `--compositional 32` is added only for the compositional rows.

```bash
python data.py [--source <source>] --patch_size <patch_size> --obj_size <obj_size> --n_train 6400 [--match_to_sample] [--compositional 32]
```

Generate the held-out IID test set with the same dataset options and `--create_held_out_test_set`:

```bash
python data.py [--source <source>] --patch_size <patch_size> --obj_size <obj_size> --n_train 6400 [--match_to_sample] --create_held_out_test_set
```

Generate counterfactual data for DAS with the same dataset options and `--create_das` (do not pass `--n_train`):

```bash
python data.py [--source <source>] --patch_size <patch_size> --obj_size <obj_size> [--match_to_sample] --create_das
```

Square brackets denote optional arguments. For RMTS, use `--source mts --match_to_sample`; for discrimination, omit both. Use `--patch_size 14 --obj_size 28` only for DINOv2.

## 2. Model Training

Use this single command for every model/task/loss combination:

```bash
python train.py -m <model> [--pretrained] --dataset_str <dataset> --patch_size <patch_size> --obj_size <obj_size> --num_epochs <epochs> --batch_size 128 --compositional <compositional> [--auxiliary_loss]
```

Replace values using these tables. `--pretrained` is used for every model except the scratch ViT. `--auxiliary_loss` is either present or absent.

### Models and input sizes

| Model | `-m` | `--pretrained` | Patch/object size |
| --- | --- | --- | --- |
| ImageNet ViT | `vit` | present | 16 / 32 |
| CLIP ViT | `clip_vit` | present | 16 / 32 |
| DINO ViT | `dino_vit` | present | 16 / 32 |
| MAE ViT | `mae_vit` | present | 16 / 32 |
| DINOv2 ViT | `dinov2_vit` | present | 14 / 28 |
| ViT from scratch | `vit` | absent | 16 / 32 |

### Dataset and epoch values

`<dataset>` is `NOISE_RGB` for discrimination and `mts` for RMTS. For all runs, `<compositional>` is `-1` (the compositional training-set variant is `32` when intentionally training on that split).

| Model | Discrimination: without / with auxiliary loss | RMTS: without / with auxiliary loss |
| --- | ---: | ---: |
| ImageNet ViT | 50 / 100 | 100 / 200 |
| CLIP ViT | 15 / 30 | 30 / 50 |
| DINO ViT | 20 / 50 | 30 / 100 |
| MAE ViT | 20 / 50 | 50 / 100 |
| DINOv2 ViT | 15 / 30 | 30 / 50 |
| ViT from scratch | 100 / 200 | 200 / 300 |

## 3. Distributed Alignment Search (DAS)

Run this command for each row in the checkpoint table and for every requested combination of analysis and control:

```bash
python das.py --pretrain <pretrain> --task <task> [-ds mts] --run_id <run_id> --analysis <analysis> --compositional -1 --control <control> --patch_size <patch_size> --obj_size <obj_size> --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true [--min_layer 0 --max_layer 12]
```

Use `-ds mts` only for RMTS. Use the layer range for discrimination runs; omit it for RMTS. The allowed varying parameters are:

| Parameter | Options |
| --- | --- |
| `<task>` | `discrimination`, `rmts` |
| `<analysis>` | `shape`, `color` |
| `<control>` | `none`, `wrong_object` |
| `<pretrain>` | `clip`, `dino`, `dinov2`, `imagenet`, `mae`, `scratch` |
| `<patch_size>`, `<obj_size>` | `14`, `28` for `dinov2`; `16`, `32` otherwise |

### Checkpoint table

The `run_id` must match the checkpoint trained with the same task, input size, and compositional setting. Each cell is the run ID for **with auxiliary loss / without auxiliary loss**.

## 4. Linear-Probing Intervention

Run the following command for each checkpoint and both intervention conditions:

```bash
python linear_probe_intervention.py --pretrain <pretrain> --run_id <run_id> --patch_size <patch_size> --obj_size <obj_size> --compositional -1 --control <control> --alpha <alpha> --datasize <datasize>
```

Allowed values are:

| Parameter | Options |
| --- | --- |
| `<pretrain>` | `clip`, `dino`, `dinov2`, `imagenet`, `mae`, `scratch` |
| `<control>` | `false` for the standard intervention, `true` for the control intervention |
| `<alpha>` | `1.0` in the replication runs |
| `<datasize>` | `2000` in the replication runs |
| `<patch_size>`, `<obj_size>` | `14`, `28` for `dinov2`; `16`, `32` otherwise |

Use the checkpoint table in Section 3. The auxiliary-loss status determines which run ID to select; run both `control=false` and `control=true` for each selected checkpoint.

## 5. Analysis and Result Generation

### Abstraction analysis

```bash
python analysis/analyze_abstraction.py --pretrain <pretrain> --task <task> [--patch_size 14 --obj_size 28]
```

`<pretrain>` is one of `clip`, `dinov2`, `imagenet`, `dino`, `mae`, or `scratch`; `<task>` is `discrimination` or `rmts`. Add the DINOv2 size arguments only for `dinov2`.

### DAS analysis

```bash
python analysis/analyze_das.py --pretrain <pretrain> [--patch_size 14 --obj_size 28]
```

Use the same six `<pretrain>` options and DINOv2 size rule as above.

### Linear-intervention analysis

```bash
python analysis/analyze_intervention.py --pretrain <pretrain> --alpha 1.0 [--patch_size 14]
```

Use the same six `<pretrain>` options. Add `--patch_size 14` only for `dinov2`.

## 6. Consistency Checklist

- Use `patch_size=14` and `obj_size=28` for DINOv2; use `16` and `32` for all other models.
- Use `dataset_str=NOISE_RGB` for discrimination and `dataset_str=mts` plus `--match_to_sample` for RMTS.
- Use `compositional=-1` for the main replication runs; use `32` only for compositional training or data-generation rows.
- Select the auxiliary-loss or no-auxiliary-loss run ID consistently across DAS and linear probing.
- Confirm that every run ID was trained with the same task, input size, and compositional setting as the evaluation command.