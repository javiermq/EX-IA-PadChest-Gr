# EX-IA-PadChest-Gr

Script para descargar el dataset PadChest-GR desde el enlace publico de B2Drop/Nextcloud.

## Requisitos

- Python 3.9 o superior
- `requests`

Instala la dependencia con:

```bash
pip install requests
```

## Uso

Descarga el archivo con el nombre por defecto `PadChest-GR.zip`:

```bash
python download_padchest_gr.py
```

Tambien puedes indicar una ruta de salida:

```bash
python download_padchest_gr.py --output data/PadChest-GR.zip
```

O usar otra URL de descarga:

```bash
python download_padchest_gr.py --url "https://b2drop.bsc.es/nextcloud/s/PadChest-GR/download" --output PadChest-GR.zip
```

## Notas

- El script descarga el archivo por bloques, asi que no carga todo el dataset en memoria.
- Si la carpeta de salida no existe, se crea automaticamente.
- Si el servidor informa del tamano del archivo, se muestra el porcentaje de descarga.
