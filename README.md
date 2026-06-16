# EX-IA-PadChest-GR

Repositorio limpio para descargar PadChest-GR y preparar el experimento:

- top-10 categorias mas frecuentes desde `category annotations`;
- bucket `Other` para el resto;
- baseline DenseNet121 con 11 salidas;
- Qwen congelado con tokens especiales cerrados, 10 MLPs visuales de categoria, loss de clasificacion, next-token y CLIP.

## Requisitos

```bash
python -m pip install -r requirements.txt
```

## 1. Descargar PadChest-GR

```bash
python download_padchest_gr.py --output data/PadChest-GR.zip
```

Si hace falta otra URL:

```bash
python download_padchest_gr.py \
  --url "https://b2drop.bsc.es/nextcloud/s/PadChest-GR/download" \
  --output data/PadChest-GR.zip
```

## 2. Preparar top-10 + Other

Cuando tengas extraido el archivo de `category annotations`, genera el manifiesto:

```bash
python -m src.padchest_gr.prepare_top10_manifest \
  --annotations data/raw/category_annotations.tsv \
  --out-tsv data/processed/categories_top10_other.tsv \
  --vocab-out data/processed/category_vocab.json \
  --image-id-column image_id \
  --image-path-column image_path \
  --category-column category
```

Si no pasas las columnas, el script intenta detectar nombres habituales.

Salidas:

```text
data/processed/categories_top10_other.tsv
data/processed/category_vocab.json
```

El TSV contiene:

```text
image_id
image_path
categories_top10
has_other
target_categories
target_category_ids
target_tokens
label_<top10_categoria>
label_other
```

La baseline usa 11 salidas: las 10 categorias frecuentes + `Other`.

## 3. Baseline DenseNet

```bash
python -m src.padchest_gr.train_densenet_baseline \
  --manifest-tsv data/processed/categories_top10_other.tsv \
  --vocab-json data/processed/category_vocab.json \
  --out-dir runs/densenet_top10_other \
  --epochs 20 \
  --batch-size 16 \
  --device cuda
```

El entrenamiento usa `BCEWithLogitsLoss`, porque las etiquetas son multilabel.
Guarda el mejor checkpoint en:

```text
runs/densenet_top10_other/best.pt
```

## 4. Qwen con tokens especiales cerrados

El vocabulario genera tokens del tipo:

```text
<finding_cardiomegaly>
<finding_pleural_effusion>
...
<finding_other>
```

La arquitectura Qwen usa:

- DenseNet para extraer features visuales;
- 10 MLPs de dos capas, uno por cada categoria top-10;
- un slot visual por categoria en el espacio de Qwen;
- Qwen congelado;
- embeddings de los tokens especiales entrenables solo en sus filas;
- una cabeza MLP de clasificacion sobre hidden states de los slots visuales;
- evaluacion sin leer tokens de respuesta.

Entrenamiento:

```bash
python -m src.padchest_gr.train_qwen_category_tokens \
  --manifest-tsv data/processed/categories_top10_other.tsv \
  --vocab-json data/processed/category_vocab.json \
  --densenet-checkpoint runs/densenet_top10_other/best.pt \
  --out-dir runs/qwen_category_tokens \
  --model-id Qwen/Qwen2.5-1.5B \
  --epochs 10 \
  --batch-size 2 \
  --device cuda \
  --freeze-densenet
```

Loss total:

```text
loss =
  lambda_cls  * BCEWithLogitsLoss(hidden_state_classifier, labels_11)
+ lambda_lm   * next_token_loss(answer_tokens)
+ lambda_clip * category_slot_to_token_alignment
```

Puedes ajustar pesos:

```bash
--lambda-cls 1.0 --lambda-lm 1.0 --lambda-clip 1.0
```

## 5. Category vs box annotations

Este pipeline usa `category annotations` para clasificacion multilabel.

Las `box annotations` son una tarea separada de localizacion/grounding. No se mezclan aqui para no asumir que toda categoria global tiene caja. El siguiente paso natural seria crear:

```text
data/processed/boxes.tsv
```

con:

```text
image_id
category
category_id
x_min
y_min
x_max
y_max
```

filtrado al mismo vocabulario top-10 + `Other`.

## 6. Preparar box annotations

Las etiquetas de caja pueden ser distintas de las categorias globales. Para esta rama se calculan las 10 labels de caja mas frecuentes y el resto pasa a `Other`.

```bash
python -m src.padchest_gr.prepare_box_manifest \
  --boxes data/raw/box_annotations.tsv \
  --out-tsv data/processed/boxes_top10_other.tsv \
  --vocab-out data/processed/box_vocab.json \
  --image-id-column image_id \
  --image-path-column image_path \
  --label-column label \
  --x-min-column x_min \
  --y-min-column y_min \
  --x-max-column x_max \
  --y-max-column y_max
```

Si las cajas estan en pixeles, anade columnas de tamano de imagen:

```bash
--image-width-column image_width --image-height-column image_height
```

El manifiesto genera coordenadas originales y una version discreta en grid `49x49`:

```text
grid_x_min
grid_y_min
grid_x_max
grid_y_max
```

## 7. GradCAM vs bounding boxes

Usa el checkpoint DenseNet de clasificacion y compara los mapas GradCAM contra las cajas que compartan el mismo `label_id` entre vocabulario de categorias y vocabulario de cajas.

```bash
python -m src.padchest_gr.compare_gradcam_boxes \
  --boxes-tsv data/processed/boxes_top10_other.tsv \
  --box-vocab-json data/processed/box_vocab.json \
  --category-vocab-json data/processed/category_vocab.json \
  --densenet-checkpoint runs/densenet_top10_other/best.pt \
  --out-tsv runs/gradcam_box_comparison.tsv \
  --device cuda
```

Metricas por imagen/label:

```text
gradcam_iou
pointing_hit
```

Esta comparativa es inicial: si las labels de caja y categoria no coinciden por nombre normalizado, no se fuerza un mapeo artificial.

## 8. Qwen para grounding 49x49

Esta rama usa labels de caja como tokens especiales cerrados:

```text
<finding_box_label_1>
...
<finding_box_label_10>
```

Cada label tiene un MLP de dos capas que proyecta su heatmap `49x49` al espacio hidden de Qwen. Desde los hidden states previos a la respuesta se predice:

- presencia multilabel de las 10 labels de caja;
- una matriz de calor `49x49` por label.

```bash
python -m src.padchest_gr.train_qwen_box_grounding \
  --boxes-tsv data/processed/boxes_top10_other.tsv \
  --box-vocab-json data/processed/box_vocab.json \
  --out-dir runs/qwen_box_grounding \
  --model-id Qwen/Qwen2.5-1.5B \
  --epochs 10 \
  --batch-size 2 \
  --device cuda
```

Loss total:

```text
loss =
  lambda_label   * BCEWithLogitsLoss(label_prediction, labels_10)
+ lambda_heatmap * BCEWithLogitsLoss(heatmap_prediction, heatmaps_10x49x49)
+ lambda_lm      * next_token_loss(box_label_tokens)
+ lambda_clip    * label_slot_to_token_alignment
```

La evaluacion no lee los tokens generados como clasificacion. Usa hidden states de los slots de label antes de la respuesta.
