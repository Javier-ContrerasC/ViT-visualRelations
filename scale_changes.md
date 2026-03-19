
## **Cambios necesarios para agregar Scale**

### **1. Generación de estímulos (data.py)**

**Actualmente:**
- Shapes: 16 formas en `stimuli/source/shapemasks/{0..15}.png`
- Colors: 16 colores base en `color_combos`
- Objects se crean a un tamaño fijo (`obj_size`)
- Filenames: `{shape}_{color}.png`

**Con Scale:**
- Necesitas múltiples tamaños de los mismos objetos
- Filenames: `{shape}_{color}_{scale}.png`

**Cambios en data.py:**

Agregar diccionario de escalas (similar a `color_combos`):
```python
# Agregar después de color_combos (alrededor línea 47)
scale_options = [0.5, 0.75, 1.0, 1.25, 1.5]  # 5 escalas diferentes
scale_to_int = {str(s): i for i, s in enumerate(scale_options)}
```

Modificar `create_source()` para iterar sobre escalas:
```python
def create_source(source, obj_size=32):
    # ... código existente ...
    
    for color_file in colors:
        color_name = f"mean{color_file[0]}_var{color_file[1]}"
        
        for shape_file in shape_masks:
            shape_name = shape_file.split("/")[-1][:-4]
            
            # NUEVO: Iterar sobre escalas
            for scale in scale_options:
                # Crear objeto en tamaño base
                mask = (Image.open(shape_file)
                        .convert("RGBA")
                        .resize((obj_size, obj_size), Image.NEAREST))
                
                # ... resto del código existente ...
                
                # NUEVO: Redimensionar según scale
                scaled_size = int(obj_size * scale)
                mask = mask.resize((scaled_size, scaled_size), Image.LANCZOS)
                color = color.resize((scaled_size, scaled_size), Image.LANCZOS)
                
                # Guardar con scale en nombre
                scale_name = f"scale{str(scale).replace('.', '_')}"
                base.convert("RGB").save(
                    f"{stim_dir}/{shape_name}_{color_name}_{scale_name}.png"
                )
```

---

### **2. Actualizar datadict (metadata)**

**Actualmente:** datadict.pkl almacena para cada imagen:
```python
{
    "s1": shape_1,      # 0-15
    "s2": shape_2,      # 0-15
    "c1": color_1,      # "mean233-30-99_var10"
    "c2": color_2,
    "pos1": [posiciones],
    "pos2": [posiciones],
    # RMTS también tiene display1-s, display2-s, display1-c, display2-c
}
```

**Con Scale:** Agregar campos:
```python
{
    "s1": shape_1,
    "s2": shape_2,
    "c1": color_1,
    "c2": color_2,
    "scale_1": 2,       # NUEVO: índice 0-4 según scale_options
    "scale_2": 1,
    "pos1": [...],
    "pos2": [...],
    # Para RMTS también:
    "display1-scale": scale_idx,
    "display2-scale": scale_idx,
}
```

**Cambios en función que genera pares (alrededor línea 1200):**
```python
def generate_pairs(...):
    # ... código existente que selecciona shapes y colors ...
    
    # NUEVO: También seleccionar escalas
    scale_1 = random.choice(range(len(scale_options)))
    scale_2 = random.choice(range(len(scale_options)))
    
    # Guardar en datadict
    im_dict["scale_1"] = scale_1
    im_dict["scale_2"] = scale_2
    return im_dict
```

---

### **3. Actualizar datasets (load_dataset y SameDifferentDataset)**

**En `load_dataset()` (alrededor línea 280):**
```python
def load_dataset(root_dir, subset=None, task="discrimination", size=-1):
    # ... código existente ...
    
    for im in im_paths:
        dict_key = os.path.join(*im.split("/")[-3:])
        
        if task == "rmts":
            im_dict = {
                # ... campos existentes ...
                "scale_1": data_dictionary[dict_key]["scale_1"],
                "scale_2": data_dictionary[dict_key]["scale_2"],
                "display_scale_1": data_dictionary[dict_key]["display1-scale"],
                "display_scale_2": data_dictionary[dict_key]["display2-scale"],
            }
        elif task == "discrimination":
            im_dict = {
                # ... campos existentes ...
                "scale_1": data_dictionary[dict_key]["scale_1"],
                "scale_2": data_dictionary[dict_key]["scale_2"],
            }
        
        ims[idx] = im_dict
```

**En `SameDifferentDataset.__getitem__()` (alrededor línea 643):**
```python
def __getitem__(self, idx):
    # ... código existente ...
    
    item["scale_1"] = int(self.im_dict[idx]["scale_1"])
    item["scale_2"] = int(self.im_dict[idx]["scale_2"])
    
    # Object identity ahora incluye 3 factores
    # object_id = color * num_shapes * num_scales + shape * num_scales + scale
    num_scales = len(scale_options)
    object_1 = (item["color_1"] * self.num_shapes * num_scales) + \
               (item["shape_1"] * num_scales) + item["scale_1"]
    object_2 = (item["color_2"] * self.num_shapes * num_scales) + \
               (item["shape_2"] * num_scales) + item["scale_2"]
    
    item["object_1"] = object_1
    item["object_2"] = object_2
    
    if self.task == "rmts":
        item["display_scale_1"] = int(self.im_dict[idx]["display_scale_1"])
        item["display_scale_2"] = int(self.im_dict[idx]["display_scale_2"])
```

---

### **4. Actualizar probes en utils.py**

**En `get_model_probes()` (alrededor línea 305):**
```python
def get_model_probes(
    model, 
    num_shapes, 
    num_colors, 
    num_scales,  # NUEVO parámetro
    num_classes, 
    probe_for, 
    split_embed=False, 
    device="cuda", 
    obj_size=32, 
    patch_size=32
):
    
    if probe_for == "auxiliary_loss":
        return (
            nn.Linear(probe_dim, num_shapes).to(device),
            nn.Linear(probe_dim, num_colors).to(device),
            nn.Linear(probe_dim, num_scales).to(device),  # NUEVO
        )
```

**En train.py (alrededor línea 1180):**
```python
probes = utils.get_model_probes(
    model,
    num_shapes=16,
    num_colors=16,
    num_scales=5,  # Nuevo parámetro
    num_classes=2,
    probe_for=probe_value,
    obj_size=obj_size,
    patch_size=patch_size,
    split_embed=args.probe_type == "shape-color",
)
```

**En train.py (alrededor línea 1360):**
```python
if args.auxiliary_loss:
    if args.probe_type == "shape-color":
        params = (
            list(model_params)
            + list(probes[0].parameters())
            + list(probes[1].parameters())
            + list(probes[2].parameters())  # Probe de scale
        )
```

---

### **5. DAS con Scale**

**Para ejecutar DAS con scale:**
```bash
python das.py --pretrain clip --task discrimination --run_id {run_id} \
  --analysis scale --compositional -1 --control none \
  --patch_size 16 --obj_size 32 --lr 0.001 --mask_lr 0.01 --num_epochs 20
```

No necesitas cambios en das.py si usas el parámetro `--analysis scale` directamente, porque ya maneja análisis dinámicos.

---

### **6. Linear Intervention con Scale**

**Para linear probe intervention con scale:**
```bash
python linear_probe_intervention.py --pretrain clip --run_id {rmts_run_id} \
  --patch_size 16 --obj_size 32 --compositional -1 \
  --control false --alpha 1.0
```

En linear_probe_intervention.py, la intervención lineal usará los probes de size automáticamente si los contiene.

---

### **7. Archivos a crear y modificar - Resumen**

| Archivo | Cambios |
|---------|---------|
| data.py | Agregar `scale_options`, `scale_to_int`. Modificar `create_source()`, `generate_pairs()`, `load_dataset()`, `SameDifferentDataset`. |
| utils.py | Agregar parámetro `num_scales` a `get_model_probes()`. |
| train.py | Pasar `num_scales=5` a `get_model_probes()`. Manejar 3 probes en auxiliary loss. |
| das.py | Sin cambios (ya soporta `--analysis` dinámico). |
| linear_probe_intervention.py | Sin cambios (automático si hay probes). |

---

### **8. Ejecución del nuevo pipeline con Scale**

```bash
# 1. Generar source objects con scale
python data.py --create_source --source NOISE_RGB --obj_size 32

# 2. Generar datasets (automáticamente incluyen scale)
python data.py --patch_size 16 --obj_size 32 --n_train 6400

# 3. Entrenar modelo (con auxiliary loss para induce scale subspace)
python train.py -m clip_vit --pretrained --dataset_str NOISE_RGB \
  --patch_size 16 --obj_size 32 --auxiliary_loss --num_epochs 30

# 4. Generar DAS counterfactual para scale
python data.py --patch_size 16 --obj_size 32 --create_das

# 5. Correr DAS con scale
python das.py --pretrain clip --task discrimination --run_id {run_id} \
  --analysis scale --compositional -1 --control none --patch_size 16 --obj_size 32
```