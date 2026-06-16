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
