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

### Test iid
```bash
python data.py --patch_size 16 --obj_size 32 --n_train 6400 --create_held_out_test_set

python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --match_to_sample --create_held_out_test_set
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
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss

# ImageNet Discrimination (pretrained)
python train.py -m vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss

# DINO Discrimination (pretrained)
python train.py -m dino_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss

# MAE Discrimination (pretrained)
python train.py -m mae_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss

# From Scratch Discrimination
python train.py -m vit --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss
```

### Ejemplos de llamadas para RMTS (b16)

```bash
# CLIP RMTS
python train.py -m clip_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss --num_epochs 30

# ImageNet RMTS
python train.py -m vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 \
  --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss --num_epochs 30
```

### Ejemplos para DINOv2 (b14, obj_size=28)

```bash
# DINOv2 Discrimination
python train.py -m dinov2_vit --dataset_str NOISE_RGB --patch_size 14 --obj_size 28 \
  --num_epochs 30 --batch_size 64 --compositional -1

# DINOv2 RMTS
python train.py -m dinov2_vit --dataset_str mts --patch_size 14 --obj_size 28 \
  --num_epochs 30 --batch_size 64 --compositional -1
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

# Flujo a mantener y considerar

Perfecto. Aquí va la mini guía corta para no desalinear rutas entre data, train, DAS y linear probe.

**Regla de oro**
1. Mantén idénticos estos campos en todas las etapas: dataset/task, patch_size, obj_size, compositional.
2. El run_id de DAS/linear probe debe venir del entrenamiento hecho con esos mismos valores.
3. Si usas RMTS, usa siempre match_to_sample en generación de datos.

**Valores por escenario**
1. No-compositional:
- compositional = -1
- Equivale a 256-256-256
2. Compositional-32:
- compositional = 32
- Equivale a 32-32-224

**Pipeline consistente (Discrimination)**
1. Data:
- python data.py --patch_size 16 --obj_size 32 --n_train 6400 --compositional X
2. Train:
- python train.py --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --compositional X ...
3. DAS:
- python das.py --task discrimination --patch_size 16 --obj_size 32 --compositional X --run_id <run_id_correcto> ...

**Pipeline consistente (RMTS)**
1. Data:
- python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --match_to_sample --compositional X
2. Train:
- python train.py --dataset_str mts --patch_size 16 --obj_size 32 --compositional X ...
3. DAS:
- python das.py --task rmts -ds mts --patch_size 16 --obj_size 32 --compositional X --run_id <run_id_correcto> ...
4. Linear probe:
- python linear_probe_intervention.py --patch_size 16 --obj_size 32 --compositional X --run_id <run_id_correcto> ...

Donde X es -1 o 32, pero siempre el mismo en toda la cadena.

**Para DINOv2**
1. Usa patch_size 14 y obj_size 28 en todas las etapas.
2. No mezclar con checkpoints/datasets b16 (16/32).

**Chequeo rápido antes de correr DAS/Probe**
1. ¿El run_id viene de ese mismo dataset/task/compositional?
2. ¿patch_size y obj_size coinciden con el checkpoint?
3. ¿Para RMTS usaste match_to_sample en data?
4. ¿Existe el split que vas a evaluar (test_iid solo si lo generaste)?

Si quieres, te preparo un bloque de comandos “copiar y pegar” para dos rutas completas: RMTS no-compositional y RMTS compositional-32.



# Codigos iniciales post-fix

Preparación (una sola vez por sesión)
source conda.sh
conda activate relational-circuits

Generar source forma+escala (obligatorio antes de regenerar datasets)
python data.py --create_source --source NOISE_RGB --obj_size 32

Discrimination IID completo (train/val/test + test_iid)
python data.py --source NOISE_RGB --patch_size 16 --obj_size 32 --n_train 6400 --n_val 6400 --n_test 6400 --compositional -1
python data.py --source NOISE_RGB --patch_size 16 --obj_size 32 --n_train 6400 --compositional -1 --create_held_out_test_set

Entrenamiento Discrimination (ViT)
WANDB_MODE=online python train.py -m vit --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --n_train 6400 --n_val 6400 --n_test 6400 --compositional -1 --batch_size 64 --num_epochs 30

RMTS completo (train/val/test + test_iid)
python data.py --create_source --source mts --obj_size 32
python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --n_val 6400 --n_test 6400 --match_to_sample --compositional -1
python data.py --source mts --patch_size 16 --obj_size 32 --n_train 6400 --match_to_sample --compositional -1 --create_held_out_test_set


---

ENTRENAMIENTO DISCRIMINACION

python train.py -m vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 100 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 50 --batch_size 128 --compositional -1 && python train.py -m clip_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m clip_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 15 --batch_size 128 --compositional -1 && python train.py -m dino_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 20 --batch_size 128 --compositional -1 && python train.py -m dino_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 50 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m mae_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 20 --batch_size 128 --compositional -1 && python train.py -m mae_vit --pretrained --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 50 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m dinov2_vit --pretrained --dataset_str NOISE_RGB --patch_size 14 --obj_size 28 --num_epochs 15 --batch_size 128 --compositional -1 && python train.py -m dinov2_vit --pretrained --dataset_str NOISE_RGB --patch_size 14 --obj_size 28 --num_epochs 30 --batch_size 128 --compositional -1 --auxiliary_loss

python train.py -m vit --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 200 --batch_size 128 --compositional -1 --auxiliary_loss
python train.py -m vit --dataset_str NOISE_RGB --patch_size 16 --obj_size 32 --num_epochs 100 --batch_size 128 --compositional -1
---

ENTRENAMIENTO RTMS

python train.py -m vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 200 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 100 --batch_size 128 --compositional -1 && python train.py -m clip_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 50 --batch_size 128 --compositional -1 --auxiliary_loss && python train.py -m clip_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 30 --batch_size 128 --compositional -1 && python train.py -m dino_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 30 --batch_size 128 --compositional -1 && python train.py -m dino_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 100 --batch_size 128 --compositional -1 --auxiliary_loss &&

python train.py -m dinov2_vit --pretrained --dataset_str mts --patch_size 14 --obj_size 28 --num_epochs 30 --batch_size 128 --compositional -1 ; python train.py -m dinov2_vit --pretrained --dataset_str mts --patch_size 14 --obj_size 28 --num_epochs 50 --batch_size 128 --compositional -1 --auxiliary_loss ; python train.py -m mae_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 50 --batch_size 128 --compositional -1 ; python train.py -m mae_vit --pretrained --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 100 --batch_size 128 --compositional -1 --auxiliary_loss

python train.py -m vit --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 300 --batch_size 128 --compositional -1 --auxiliary_loss
python train.py -m vit --dataset_str mts --patch_size 16 --obj_size 32 --num_epochs 200 --batch_size 128 --compositional -1
---

DAS - LINEAR PROBE w/AUX LOSS

CLIP
python das.py --pretrain clip --task discrimination --run_id model_29_2e-06_6a197401-1b1e-43e3-ace6-dbb3793d8489 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain clip --task rmts -ds mts --run_id model_49_2e-06_46608962-f732-4cd6-8606-97c70dfcc8c1 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


CUDA_VISIBLE_DEVICES=0 python das.py --pretrain clip --task discrimination --run_id model_29_2e-06_6a197401-1b1e-43e3-ace6-dbb3793d8489 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=0 python das.py --pretrain clip --task rmts -ds mts --run_id model_49_2e-06_46608962-f732-4cd6-8606-97c70dfcc8c1 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

DINO
python das.py --pretrain dino --task discrimination --run_id model_49_2e-06_c5e1de8a-b613-4851-90c2-0ca92b5d9e7c --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain dino --task rmts -ds mts --run_id model_99_2e-06_b34b4d22-5ae2-4bbf-b05e-5b6fa1573065 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


CUDA_VISIBLE_DEVICES=0 python das.py --pretrain dino --task discrimination --run_id model_49_2e-06_c5e1de8a-b613-4851-90c2-0ca92b5d9e7c --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=0 python das.py --pretrain dino --task rmts -ds mts --run_id model_99_2e-06_b34b4d22-5ae2-4bbf-b05e-5b6fa1573065 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


DINOV2
python das.py --pretrain dinov2 --task discrimination --run_id model_29_2e-06_ef385459-b223-4f67-8973-43aeba0f411f --analysis shape --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain dinov2 --task rmts -ds mts --run_id model_49_2e-06_c2a67f8e-fdb1-476c-b98d-7a2a940137f4 --analysis shape --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=0 python das.py --pretrain dinov2 --task discrimination --run_id model_29_2e-06_ef385459-b223-4f67-8973-43aeba0f411f --analysis color --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain dinov2 --task rmts -ds mts --run_id model_49_2e-06_c2a67f8e-fdb1-476c-b98d-7a2a940137f4 --analysis color --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


IMAGENET
python das.py --pretrain imagenet --task discrimination --run_id model_99_2e-06_890cd6c4-e840-40ee-bae7-e797ecb1e45b --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain imagenet --task rmts -ds mts --run_id model_199_2e-06_5893d3ce-bf15-4e32-9888-f03342f672e9 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=1 python das.py --pretrain imagenet --task discrimination --run_id model_99_2e-06_890cd6c4-e840-40ee-bae7-e797ecb1e45b --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain imagenet --task rmts -ds mts --run_id model_199_2e-06_5893d3ce-bf15-4e32-9888-f03342f672e9 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


MAE
python das.py --pretrain mae --task discrimination --run_id model_49_2e-06_d627acb6-e8ad-4185-88f6-68eaba365063 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain mae --task rmts -ds mts --run_id model_99_2e-06_5ca4e4a9-0001-4367-8a24-0bb04cc9b1d3 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=1 python das.py --pretrain mae --task discrimination --run_id model_49_2e-06_d627acb6-e8ad-4185-88f6-68eaba365063 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain mae --task rmts -ds mts --run_id model_99_2e-06_5ca4e4a9-0001-4367-8a24-0bb04cc9b1d3 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


SCRATCH
python das.py --pretrain scratch --task discrimination --run_id model_199_2e-06_c516e30d-86b2-46a6-9e2e-9954bdf3c73a --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain scratch --task rmts -ds mts --run_id model_299_2e-06_200b7a79-fee6-44f8-8187-1fc7926e63ff --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=1 python das.py --pretrain scratch --task discrimination --run_id model_199_2e-06_c516e30d-86b2-46a6-9e2e-9954bdf3c73a --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain scratch --task rmts -ds mts --run_id model_299_2e-06_200b7a79-fee6-44f8-8187-1fc7926e63ff --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true


LINEAR PROBING

python linear_probe_intervention.py --pretrain clip --run_id model_49_2e-06_46608962-f732-4cd6-8606-97c70dfcc8c1 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain dino --run_id model_99_2e-06_b34b4d22-5ae2-4bbf-b05e-5b6fa1573065 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain dinov2 --run_id model_49_2e-06_c2a67f8e-fdb1-476c-b98d-7a2a940137f4 --patch_size 14 --obj_size 28 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain imagenet --run_id model_199_2e-06_5893d3ce-bf15-4e32-9888-f03342f672e9 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain mae --run_id model_99_2e-06_5ca4e4a9-0001-4367-8a24-0bb04cc9b1d3 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000

python linear_probe_intervention.py --pretrain scratch --run_id model_299_2e-06_57205943-6232-46fa-8d9d-6602e8bd4136 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000


---

DAS - LINEAR PROBE W/O AUX LOSS

CLIP
python das.py --pretrain clip --task discrimination --run_id model_14_2e-06_1660c0a2-4cac-49b8-959c-0dba66ed9fd3 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain clip --task rmts -ds mts --run_id model_29_2e-06_76f2255c-2028-4a52-aa37-fa5c5ca78f28 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=0 python das.py --pretrain clip --task discrimination --run_id model_14_2e-06_1660c0a2-4cac-49b8-959c-0dba66ed9fd3 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=0 python das.py --pretrain clip --task rmts -ds mts --run_id model_29_2e-06_76f2255c-2028-4a52-aa37-fa5c5ca78f28 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true 

DINO
python das.py --pretrain dino --task discrimination --run_id model_19_2e-06_1f304b43-ba91-4ac1-b435-e9b0acad60fd --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain dino --task rmts -ds mts --run_id model_29_2e-06_da1723c3-e3c7-4de9-994b-367e60eb14e7 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=0 python das.py --pretrain dino --task discrimination --run_id model_19_2e-06_1f304b43-ba91-4ac1-b435-e9b0acad60fd --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=0 python das.py --pretrain dino --task rmts -ds mts --run_id model_29_2e-06_da1723c3-e3c7-4de9-994b-367e60eb14e7 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

DINOV2
python das.py --pretrain dinov2 --task discrimination --run_id model_14_2e-06_7d7972c8-2736-4de9-8144-fae6b29bfd7a --analysis shape --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain dinov2 --task rmts -ds mts --run_id model_29_2e-06_5e9130b6-96c6-47d1-b9ff-04c65393663a --analysis shape --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=1 python das.py --pretrain dinov2 --task discrimination --run_id model_14_2e-06_7d7972c8-2736-4de9-8144-fae6b29bfd7a --analysis color --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain dinov2 --task rmts -ds mts --run_id model_29_2e-06_5e9130b6-96c6-47d1-b9ff-04c65393663a --analysis color --compositional -1 --control none --patch_size 14 --obj_size 28 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

IMAGENET
python das.py --pretrain imagenet --task discrimination --run_id model_49_2e-06_ca3da516-c1ff-42f0-81ad-0ebb1982ba78 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain imagenet --task rmts -ds mts --run_id model_99_2e-06_d912a591-5fce-4ad9-ae94-4fc021178a09 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=1 python das.py --pretrain imagenet --task discrimination --run_id model_49_2e-06_ca3da516-c1ff-42f0-81ad-0ebb1982ba78 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; CUDA_VISIBLE_DEVICES=1 python das.py --pretrain imagenet --task rmts -ds mts --run_id model_99_2e-06_d912a591-5fce-4ad9-ae94-4fc021178a09 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

MAE
python das.py --pretrain mae --task discrimination --run_id model_19_2e-06_a68971e3-bdf3-4fdc-b67f-ed4f96303fc0 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain mae --task rmts -ds mts --run_id model_49_2e-06_33c445ed-3d0b-463e-a9ac-f25f3d2fafab --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

python das.py --pretrain mae --task discrimination --run_id model_19_2e-06_a68971e3-bdf3-4fdc-b67f-ed4f96303fc0 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain mae --task rmts -ds mts --run_id model_49_2e-06_33c445ed-3d0b-463e-a9ac-f25f3d2fafab --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

SCRATCH
CUDA_VISIBLE_DEVICES=0 python das.py --pretrain scratch --task discrimination --run_id model_99_2e-06_7e84b406-e32c-4454-b997-3571207da5d8 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain scratch --task rmts -ds mts --run_id model_199_2e-06_8f7ba8cf-d48d-443d-ad37-732d9b7a9d03 --analysis shape --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

CUDA_VISIBLE_DEVICES=0 python das.py --pretrain scratch --task discrimination --run_id model_99_2e-06_7e84b406-e32c-4454-b997-3571207da5d8 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true --min_layer 0 --max_layer 12 ; python das.py --pretrain scratch --task rmts -ds mts --run_id model_199_2e-06_8f7ba8cf-d48d-443d-ad37-732d9b7a9d03 --analysis color --compositional -1 --control none --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20 --tie_weights true

LINEAR PROBE
python linear_probe_intervention.py --pretrain clip --run_id model_29_2e-06_76f2255c-2028-4a52-aa37-fa5c5ca78f28 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain dino --run_id model_29_2e-06_da1723c3-e3c7-4de9-994b-367e60eb14e7 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain dinov2 --run_id model_29_2e-06_5e9130b6-96c6-47d1-b9ff-04c65393663a --patch_size 14 --obj_size 28 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain imagenet --run_id model_99_2e-06_d912a591-5fce-4ad9-ae94-4fc021178a09 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain mae --run_id model_49_2e-06_33c445ed-3d0b-463e-a9ac-f25f3d2fafab --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000 ; python linear_probe_intervention.py --pretrain scratch --run_id model_199_2e-06_8f7ba8cf-d48d-443d-ad37-732d9b7a9d03 --patch_size 16 --obj_size 32 --compositional -1 --control false --alpha 1.0 --datasize 2000