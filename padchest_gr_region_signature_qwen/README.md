# PadChest-GR Region Signature Qwen Prototype

Entrenable de punta a punta para:

`imagen RX -> tokens 14x14 -> Region Signature Tokens -> heatmaps/ROIs -> clasificacion localizada -> metricas y visualizaciones`

El prototipo prioriza resultados visuales y heads auxiliares. No predice bboxes ni labels como texto. En entrenamiento real, las imagenes se pasan por la rama visual de Qwen2.5-VL; los tokens visuales de Qwen se remuestrean a 14x14 y se fusionan con Region Signature Tokens para ROI y clasificacion.

## Estructura esperada

```text
data/
  images/
  annotations.json
```

Tambien acepta CSV. Cada anotacion puede estar en formato por imagen:

```json
{
  "image_path": "case.png",
  "split": "train",
  "labels": ["atelectasis"],
  "boxes": [[0.1, 0.2, 0.4, 0.6]],
  "box_labels": ["atelectasis"]
}
```

O por bbox, con columnas/campos como `image_path`, `split`, `x1`, `y1`, `x2`, `y2`, `box_label`.

Antes de entrenar, `train.py` calcula las 10 clases ROI mas frecuentes en train, filtra el resto, excluye imagenes sin ROIs top-10 y guarda `logs/top10_roi_labels.json`.

## Uso rapido

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml --dummy_data --epochs 1
python evaluate.py --config configs/default.yaml --dummy_data --checkpoint checkpoints/best.pt
python infer_single_image.py --config configs/default.yaml --image path/to/image.png --checkpoint checkpoints/best.pt
```

## Salidas

- `logs/metrics.csv`: loss, micro/macro-F1 global, recall ROI@K, IoU, precision/recall a IoU 0.25/0.5, Dice de heatmap y F1/accuracy ROI.
- `logs/batch_metrics.csv`: mini-evaluaciones random cada `monitoring.sample_every_n_batches`.
- `logs/top10_roi_labels.json`: clases seleccionadas y frecuencias.
- `outputs/epoch_XXX/`: imagen con GT verde/pred rojo, heatmap GT vs pred y JSON por ejemplo.
- `outputs/batch_samples/epoch_XXX_step_YYYYYY/`: 2-3 ejemplos random de validacion durante el entrenamiento.

## Integracion Qwen

El wrapper de Qwen carga `Qwen/Qwen2.5-VL-3B-Instruct` por defecto con `load_qwen_weights: true`, `require_qwen: true` y `load_in_4bit: true`. Si Qwen no puede cargarse, el entrenamiento real falla de forma explicita. El modo `--dummy_data` desactiva esa carga para tests rapidos sin descargar el modelo.

Flujo interno real:

1. `pil_image` -> `AutoProcessor` de Qwen.
2. `pixel_values` + `image_grid_thw` -> `qwen.visual`.
3. visual hidden states de Qwen -> remuestreo a `14x14`.
4. proyeccion a `hidden_dim`.
5. fusion con Region Signature Tokens.
6. heads auxiliares para heatmaps, ROI labels, bbox opcional y multilabel global.

TODO posterior: inyectar Region Signature Tokens dentro de la secuencia multimodal de Qwen antes de capas concretas del LLM, no solo consumir sus visual hidden states.
