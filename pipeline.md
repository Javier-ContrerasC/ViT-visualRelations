# Pipeline para la recreación de resultados
## 1. Generación de datos
### Source objects (one-time setup)
```bash
python data.py --create_source --source NOISE_RGB --obj_size 32
```

### Discrimination dataset (NOISE_RGB)
```bash
# No-compositional (256-256-256)
python data.py --patch_size 16 --obj_size 32 --n_train 6400

# Compositional (32-32-224)
python data.py --patch_size 16 --obj_size 32 --n_train 6400 --compositional 32
```

### RMTS dataset (mts)
```bash
# No-compositional (256-256-256)
python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --match_to_sample

# Compositional (32-32-224)
python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --match_to_sample --compositional 32
```

### DAS counterfactual dataset
```bash
# Discrimination (shape + color, para compositional -1 y 32)
python data.py --patch_size 16 --obj_size 32 --create_das

# RMTS (shape + color, para compositional -1 y 32)
python data.py --source mts --patch_size 16 --obj_size 32 --create_das --match_to_sample
```

---

## 2. Entrenamiento (Model Training)

**Dónde está definido:**
- argparsers.py - Parámetros CLI
- train.py - Lógica de entrenamiento
- sweep.py - Lanzador de sweeps (WandB)

**Parámetros clave:**
- `--model_type`: vit, clip_vit, dino_vit, mae_vit, dinov2_vit
- `--pretrained`: flag para usar pretrained o from-scratch
- `--dataset_str`: NOISE_RGB o mts
- `--patch_size`: 16, 32 o 14 (depende del modelo)
- `--obj_size`: 32 o 28
- `--compositional`: -1 (no-compositional 256-256-256) o N (compositional específico)
- `--lr`, `--num_epochs`, `--batch_size`
- `--auxiliary_loss`: flag para incluir auxiliary loss (inducer subspaces)
- `--ood`: flag para evaluación OOD

**Importante - Salida del entrenamiento:**
- Checkpoint se guarda en: `models/{pretrain_type}/{dataset_str}_{obj_size}/{comp_str}_{wandb_run_id}.pth`
- El `wandb.run.id` se usa luego como `--run_id` en DAS e intervención
- Ver results.csv para recuperar run_ids

### Ejemplos de llamadas para Discrimination (b16)

```bash
# CLIP Discrimination (pretrained)
python train.py -m clip_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# ImageNet Discrimination (pretrained)
python train.py -m vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# DINO Discrimination (pretrained)
python train.py -m dino_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# MAE Discrimination (pretrained)
python train.py -m mae_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# From Scratch Discrimination
python train.py -m vit --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1
```

### Ejemplos de llamadas para RMTS (b16)

```bash
# CLIP RMTS
python train.py -m clip_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# ImageNet RMTS
python train.py -m vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1
```

### Ejemplos para DINOv2 (b14, obj_size=28)

```bash
# DINOv2 Discrimination
python train.py -m dinov2_vit --dataset_str NOISE_RGB --patch_size 14 --obj_size 28 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1

# DINOv2 RMTS
python train.py -m dinov2_vit --dataset_str mts --patch_size 14 --obj_size 28 \
  --lr 1e-6 --num_epochs 30 --batch_size 64 --compositional -1
```

**ADVERTENCIA - WandB local stub:**
- Este repo incluye un stub local de WandB (wandb.py)
- El stub genera UUIDs locales como run_id, no sincroniza con WandB real
- Si necesitas WandB real, desactiva el stub local
- El run_id se imprime en la salida; guárdalo para usarlo en Paso 5 y 6

---

## 3. DAS (Perceptual Stage + análisis relacional)

**Dónde está definido:**
- argparsers.py - Parser DAS
- das.py - Implementación DAS
- Scripts SLURM: disc_das.script, rmts_das.script
- Barridos ejecutados: run_discrimination_das_b16.sh, run_discrimination_das_b32.sh, run_rmts_das_b16.sh, run_rmts_das_b32.sh

**Qué hace:**
- Corre Distributed Alignment Search en cada capa del modelo
- Identifica subspacio lineal (rotación + máscara) que interviene counterfáctico
- Evalúa en train/val/test IID/OOD
- Pruebas de abstracción: ¿pueden vectores nuevos en subespacio manipular decisiones?

**Parámetros críticos:**
- `--pretrain`: clip, imagenet, dino, mae, dinov2_vit
- `--task`: discrimination o rmts
- `--run_id`: **OBLIGATORIO** - from results.csv
- `--analysis`: shape o color
- `--compositional`: -1 o 32
- `--patch_size`, `--obj_size`
- `--control`: none o wrong_object
- `--lr` (0.001), `--mask_lr` (0.01), `--num_epochs` (20)
- `--tie_weights`: true (intervenciones en patches compartidas)

**Salida esperada:**
- CSV: `logs/{pretrain}/{task}/{dataset}/DAS/{analysis}/{control_str}results.csv`
- Pesos: `logs/.../DAS/{analysis}/weights/{control_str}{layer}_weights.pth`

### Llamadas exactas para Discrimination b16

```bash
# CLIP shape (no control, no-compositional)
python das.py --pretrain clip --task discrimination --run_id 2jylw4un \
  --analysis shape --compositional -1 --control none \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true --min_layer 0 --max_layer 12

# CLIP color (wrong_object control, compositional 32)
python das.py --pretrain clip --task discrimination --run_id qzu5c6cy \
  --analysis color --compositional 32 --control wrong_object \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true --min_layer 0 --max_layer 12

# ImageNet shape (no control, no-compositional)
python das.py --pretrain imagenet --task discrimination --run_id 46bl9da9 \
  --analysis shape --compositional -1 --control none \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true
```

### Llamadas exactas para RMTS b16

```bash
# CLIP shape (no control, no-compositional)
python das.py --pretrain clip --task rmts -ds mts --run_id tavk2a8o \
  --analysis shape --compositional -1 --control none \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true

# ImageNet color (wrong_object control, compositional 32)
python das.py --pretrain imagenet --task rmts -ds mts --run_id g0vq12sz \
  --analysis color --compositional 32 --control wrong_object \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true
```

### Llamadas exactas para DINOv2 b14

```bash
# DINOv2 Discrimination shape
python das.py --pretrain dinov2_vit --task discrimination --run_id m7ngrtcn \
  --analysis shape --compositional -1 --control none \
  --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true

# DINOv2 RMTS color (wrong_object control, compositional 32)
python das.py --pretrain dinov2_vit --task rmts -ds mts --run_id udjgzn8d \
  --analysis color --compositional 32 --control wrong_object \
  --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 \
  --tie_weights true
```

**ADVERTENCIA - Ruta de checkpoints:**
- DAS espera ruta: `models/b{patch_size}/{task}/{pretrain}/{dataset}_{obj_size}/{comp_str}_{run_id}.pth`
- train.py guarda en: `models/{pretrain_type}/{dataset_str}_{obj_size}/{comp_str}_{run_id}.pth`
- Si entrenas nuevo, tendrás que mover el checkpoint al path que DAS espera, o editar das.py

---

## 4. Linear Probe Intervention (Relational Stage)

**Dónde está definido:**
- argparsers.py - Parser probe
- linear_probe_intervention.py - Intervención lineal
- Script SLURM: linear_intervention.script
- Barrido: run_linear_intervention.sh

**Qué hace:**
1. Carga modelo RMTS entrenado
2. Extrae embeddings por capa
3. Entrena probes lineales por capa para intermediate judgements (same/different)
4. Aplica intervención aditiva en residuales
5. Evalúa efecto con alpha scaling
6. Control: prueba dirección opuesta para validar especificidad

**Parámetros críticos:**
- `--pretrain`: RMTS model type (clip_vit, vit, dinov2_vit, etc.)
- `--run_id`: **OBLIGATORIO** - from trained RMTS model
- `--patch_size`, `--obj_size`
- `--compositional`: -1 o 32
- `--control`: false (normal) o true (control - dirección opuesta)
- `--alpha`: scaling factor para intervención (típico: 1.0)
- `--datasize`: ejemplos para probe/intervención
- `--lr`, `--num_epochs`, `--batch_size`

**Salida esperada:**
- CSV: `logs/{pretrain}/Linear_Intervention/alpha_{alpha}/b{patch_size}/trainsize_6400_{comp_str}/results.csv`
- Control: `logs/.../Linear_Intervention/.../control_results.csv`

### Llamadas exactas para DINOv2 b14 RMTS

```bash
# Normal intervention (control=false)
python linear_probe_intervention.py --pretrain dinov2_vit --run_id okppu3qm \
  --patch_size 14 --obj_size 28 --compositional -1 \
  --control false --alpha 1.0 --datasize 2000 \
  --lr 0.01 --num_epochs 10 --batch_size 64

# Control intervention (control=true, dirección opuesta)
python linear_probe_intervention.py --pretrain dinov2_vit --run_id udjgzn8d \
  --patch_size 14 --obj_size 28 --compositional 32 \
  --control true --alpha 1.0 --datasize 2000 \
  --lr 0.01 --num_epochs 10 --batch_size 64
```

### Llamadas exactas para CLIP/ImageNet b16 RMTS

```bash
# CLIP RMTS normal
python linear_probe_intervention.py --pretrain clip --run_id tavk2a8o \
  --patch_size 16 --obj_size 32 --compositional -1 \
  --control false --alpha 1.0 --datasize 2000

# ImageNet RMTS control
python linear_probe_intervention.py --pretrain imagenet --run_id l2cgg62f \
  --patch_size 16 --obj_size 32 --compositional -1 \
  --control true --alpha 1.0 --datasize 2000
```

---

## Resumen: Run IDs conocidos en el repo

| Pretrain | Task | b16 No-comp | b16 Comp-32 | b14 No-comp | b14 Comp-32 |
|----------|------|-----------|-----------|-----------|-----------|
| CLIP | Disc | 2jylw4un | qzu5c6cy | - | - |
| CLIP | RMTS | tavk2a8o | ld8hk72f | - | - |
| ImageNet | Disc | 46bl9da9 | sckdfa23 | - | - |
| ImageNet | RMTS | l2cgg62f | g0vq12sz | - | - |
| DINO | Disc | xkil74gw | emmfnswo | - | - |
| DINO | RMTS | qdhm9vog | eawzcswn | - | - |
| MAE | Disc | m2zlw53d | epjzjtuf | - | - |
| MAE | RMTS | 94ajmz1n | evpzdb93 | - | - |
| DINOv2 | Disc | - | - | m7ngrtcn | o0d45svp |
| DINOv2 | RMTS | - | - | okppu3qm | udjgzn8d |

---

## Advertencias y troubleshooting

### 1. Rutas de checkpoints incompatibles
- **Problema:** Entrenamiento guarda en una ruta, pero DAS/intervención busca en otra
- **Solución:** Después de entrenar, copia el checkpoint a la ruta que DAS espera, o edita das.py y linear_probe_intervention.py

### 2. WandB stub local
- **Problema:** No puedes replicar sweeps reales con sweep.py si el stub está activo
- **Solución:** Desactiva wandb.py (renombra o elimina el archivo) para usar WandB real, o llama train.py directamente con parámetros explícitos

### 3. Datos DAS counterfactual
- **Problema:** DAS no encuentra stimuli en `stimuli/das/...`
- **Solución:** Asegúrate de ejecutar `python data.py --create_das` ANTES de DAS

### 4. GPU memory
- **Problema:** Entrenamiento/DAS se queda sin memoria
- **Solución:** Reduce `--batch_size`, o aumenta `--num_gpus` si tienes múltiples GPUs

### 5. Run_id no encontrado
- **Problema:** El archivo de checkpoint no existe en la ruta esperada
- **Solución:** Verifica el nombre exacto en results.csv y ajusta la llamada