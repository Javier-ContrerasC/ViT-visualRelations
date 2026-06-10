# Entrenamiento

## Discrimination, no compositional

- ImageNet:
    - Epoch accuracy: 0.9853
    - Epoch shape accuracy: 0.9753
    - Epoch color accuracy: 0.9749
    - Val acc: 0.9885
    - Val ROC-AUC: 0.9933
- ImageNet: (sin loss)
    - Epoch accuracy: 0.9841
    - Val acc: 0.9794
    - Val ROC-AUC: 0.9833

- Clip: 
    - Epoch accuracy: 0.9858
    - Epoch shape accuracy: 0.9858
    - Epoch color accuracy: 0.9858 
    - Val acc: 0.9974
    - Val ROC-AUC: 0.9999
- Clip: (sin loss)
    - Epoch accuracy: 0.9858
    - Val acc: 0.9957
    - Val ROC-AUC: 0.9999

- Dino:
    - Epoch accuracy: 0.9858
    - Epoch shape accuracy: 0.9858
    - Epoch color accuracy: 0.9858
    - Val acc: 0.9852
    - Val ROC-AUC: 0.9977
- Dino: (sin loss)
    - Epoch accuracy: 0.9858
    - Val acc: 0.9763
    - Val ROC-AUC: 0.9927

- Mae:
    - Epoch accuracy: 0.9858
    - Epoch shape accuracy: 0.9828
    - Epoch color accuracy: 0.9731
    - Val acc: 0.9553
    - Val ROC-AUC: 0.9699
- Mae: (sin loss)
    - Epoch accuracy: 0.9858
    - Val acc: 0.9818
    - Val ROC-AUC: 0.9937

- Scratch: (con loss) (200 epochs)
    - Epoch accuracy: 0.9832
    - Epoch shape accuracy: 0.9793
    - Epoch color accuracy: 0.9663
    - Val acc: 0.9010
    - Val ROC-AUC: 0.9098
- Scratch: (sin loss) (100 epochs)
    - Epoch accuracy: 0.7316 
    - Val acc: 0.7436 
    - Val ROC-AUC: 0.4750 

- Dino v2:
    - Epoch accuracy: 0.9863
    - Epoch shape accuracy: 0.9863 
    - Epoch color accuracy: 0.9863 
    - Val acc: 0.9974 
    - Val ROC-AUC: 0.9999
- Dino v2 (sin loss):
    - Epoch accuracy: 0.9863
    - Val acc: 0.9976
    - Val ROC-AUC: 0.9996


## RTMS, no compositional

- ImageNet: (200 epochs) ajustado
    - Epoch accuracy: 0.9715
    - Epoch shape accuracy: 0.9914 
    - Epoch color accuracy: 0.9903
    - Val acc: 0.5988
    - Val ROC-AUC: 0.6430
- ImageNet: (100 epochs - sin losses)
    - Epoch accuracy: 0.9790
    - Val acc: 0.6172
    - Val ROC-AUC: 0.6456
    
- Clip: (50 epochs) ajustado
    - Epoch accuracy: 0.9915
    - Epoch shape accuracy: 0.9915
    - Epoch color accuracy: 0.9915
    - Val acc: 0.9207
    - Val ROC-AUC: 0.9590
- Clip: (30 epochs - sin losses)
    - Epoch accuracy: 0.9915
    - Val acc: 0.8925
    - Val ROC-AUC: 0.9366

- Dino: (100 epochs) ajustado
    - Epoch accuracy: 0.9915
    - Epoch shape accuracy: 0.9915
    - Epoch color accuracy: 0.9915
    - Val acc: 0.5116
    - Val ROC-AUC: 0.5116
- Dino: (30 epochs - sin losses)
    - Epoch accuracy: 0.9915
    - Val acc: 0.5156
    - Val ROC-AUC: 0.5198

- Mae: (100 epochs) - ajustado
    - Epoch accuracy: 0.6102
    - Epoch shape accuracy: 0.9909
    - Epoch color accuracy: 0.9910
    - Val acc: 0.4971
    - Val ROC-AUC: 0.4973
- Mae: (50 epochs - sin losses)
    - Epoch accuracy: 0.9897
    - Val acc: 0.5070
    - Val ROC-AUC: 0.5027

- Dino v2:
    - Epoch accuracy: 0.9915
    - Epoch shape accuracy: 0.9915
    - Epoch color accuracy: 0.9915
    - Val acc: 0.6234
    - Val ROC-AUC: 0.6577
- Dino v2 (sin losses):
    - Epoch accuracy: 0.9893
    - Val acc: 0.8841
    - Val ROC-AUC: 0.9510

- Scratch: (con loss)
    - Epoch accuracy: 0.6046
    - Epoch shape accuracy: 0.9457
    - Epoch color accuracy: 0.7652
    - Val acc: 0.4914
    - Val ROC-AUC: 0.4925
- Scratch: (sin loss) (200 epochs)
    - Epoch accuracy: 0.5830
    - Val acc: 0.5064
    - Val ROC-AUC: 0.5027

## Sobre --auxiliary_loss

Cuando activas --auxiliary_loss, esas dos métricas no miden la tarea principal same/different, sino el desempeño de las sondas auxiliares (probes) sobre las representaciones internas del ViT.

Epoch shape accuracy
Es la accuracy de la sonda de forma: qué tan bien puede predecir la clase de forma del objeto (argmax de shape_outs vs etiqueta real de shape). Se calcula en train.py:344 y se acumula por batch en train.py:556, luego se reporta por época en train.py:616.

Epoch color accuracy
Es la accuracy de la sonda de color: qué tan bien puede predecir la clase de color del objeto (argmax de color_outs vs etiqueta real de color). Se calcula en train.py:345, se acumula en train.py:557 y se imprime por época en train.py:617.

Además:

Ambas salen de la función auxiliar train.py:213, donde la sonda de shape usa la primera mitad del embedding y la de color la segunda mitad (train.py:339, train.py:340).
La accuracy principal del entrenamiento sigue siendo Epoch accuracy (same/different), en train.py:612. Las de shape/color son métricas auxiliares para ver si el modelo está separando bien esos atributos en el espacio latente.

### Sobre tokens

Se modificó la creación de tokens, ahora se realiza de forma dinámica:

La modificación exacta fue esta: antes, la lógica de patching tomaba la caja del objeto como si ocupara el cuadrado completo que se pegaba en la imagen. Ahora primero se calcula una caja efectiva más pequeña, centrada dentro de ese cuadrado, y recién después esa caja se convierte a tokens de ViT.

La secuencia nueva está en data.py:291 y data.py:1326. Ahí pasa esto:

Eso significa:

El objeto sigue pegándose dentro del mismo cuadrado base.
Pero la parte “efectiva” que cuenta para patches se achica o se mantiene según la escala.
Esa caja efectiva se centra con un margen simétrico.
Luego data.py:272 transforma esa caja en tokens de parche.
Con tus valores actuales, el tamaño que se pega sigue viniendo de data.py:1206 como un cuadrado de 24×24 píxeles, porque obj_size = 32 y se resta obj_size // buffer_factor = 8. Sobre eso, la escala aplica estas proporciones:

var8 → 0.70 → caja efectiva de 17×17
var10 → 0.80 → caja efectiva de 19×19
var12 → 0.90 → caja efectiva de 22×22
var14 → 1.00 → caja efectiva de 24×24
O sea: la lógica de patching no cambió; lo que cambió es el tamaño efectivo que entra al cálculo de cobertura de parches. Eso hace que una figura pequeña active menos parches, pero sin tocar la grilla ni el criterio de tokenización.

Si quieres, puedo darte también un ejemplo numérico de cómo una figura concreta de 24×24 termina ocupando 1, 2 o 4 patches según su escala y posición.

