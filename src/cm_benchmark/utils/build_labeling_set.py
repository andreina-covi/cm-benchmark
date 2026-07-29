"""
A partir de calibration_sample.csv (salida de sample_for_calibration.py):
1. Genera un recorte (crop) por fila desde la imagen original (con un padding
   chico para dar contexto), automaticamente.
2. Arma un visor HTML autocontenido (imagenes embebidas en base64, sin
   servidor, sin dependencias) para etiquetar rapido con teclado:
     Y / 1  = distinguible
     N / 0  = no distinguible
     <- ->  = moverse sin etiquetar
   Al terminar, un boton descarga labels.json.

Correr esto localmente donde SI tienes las imagenes (path_to_images del json
de metadata de cada episodio).
"""

import base64
import io
import json
import os
import pandas as pd
from PIL import Image

PADDING_PX = 6
THUMB_MAX_SIDE = 220  # para que el HTML no pese demasiado con cientos de crops

# Features que se guardan en calibration_manifest.csv para que fit_thresholds.py
# no tenga que volver a leer el CSV crudo. Nombres tal cual salen de tu
# navigation-*.csv real (con guiones, no guion_bajo).
FEATURE_COLUMNS = [
    "obj-distance", "cmin", "rmin", "cmax", "rmax",
    "expected-bbox-area", "ang-width-deg", "ang-height-deg",
    "visible-pixels", "bbox-area", "min-side", "occupancy-ratio",
]


def make_crop_b64(image_path, cmin, rmin, cmax, rmax, W, H):
    im = Image.open(image_path).convert("RGB")
    c0 = max(0, int(cmin) - PADDING_PX)
    r0 = max(0, int(rmin) - PADDING_PX)
    c1 = min(W, int(cmax) + PADDING_PX)
    r1 = min(H, int(rmax) + PADDING_PX)
    crop = im.crop((c0, r0, c1, r1))

    scale = THUMB_MAX_SIDE / max(crop.width, crop.height, 1)
    if scale > 1:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                            Image.NEAREST)  # NEAREST para no "inventar" detalle en objetos chicos

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(nav_csv, images_dir, scene_id, episode_id=None,
          out_html="labeling_tool.html", out_manifest="calibration_manifest.csv",
          W=396, H=224, only_real_mask=True):
    """
    nav_csv:    navigation-house_XXXXXX.csv (el crudo, tal cual lo exporta collector.py)
    images_dir: carpeta images/ de ESE episodio (hermana de annotations/, ver DATA_COLLECTION.md)
    scene_id:   ej. "house_007514" -- no viene en el CSV, hay que pasarlo a mano
    episode_id: si lo omites, se usa el nombre del nav_csv como identificador del episodio
    """
    episode_id = episode_id or os.path.basename(nav_csv)
    df = pd.read_csv(nav_csv)

    if only_real_mask:
        # Tu pool de calibracion YA esta estratificado por el stride de coleccion.
        # No hace falta volver a muestrear -- solo quedarnos con las filas que
        # de verdad tuvieron mascara real (stride steps).
        df = df[df["visible-pixels"].notna()].copy()
        df = df[df["displaced"] == False]

    df = df.reset_index(drop=True)
    df["item_id"] = df.index.astype(str)
    df["scene_id"] = scene_id
    df["episode_id"] = episode_id

    items = []
    manifest_rows = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row["path"])
        try:
            b64 = make_crop_b64(image_path, row.cmin, row.rmin, row.cmax, row.rmax, W, H)
        except Exception as e:
            print(f"WARN: no pude recortar {row.get('obj-id')} en {image_path}: {e}")
            continue
        items.append({
            "item_id": row["item_id"],
            "obj_id": row.get("obj-id", ""),
            "episode_id": episode_id,
            "b64": b64,
        })
        manifest_rows.append({
            "item_id": row["item_id"],
            "obj_id": row.get("obj-id", ""),
            "scene_id": scene_id,
            "episode_id": episode_id,
            **{c: row.get(c) for c in FEATURE_COLUMNS},
        })

    html = _HTML_TEMPLATE.replace("__ITEMS_JSON__", json.dumps(items))
    with open(out_html, "w") as f:
        f.write(html)

    pd.DataFrame(manifest_rows).to_csv(out_manifest, index=False)
    print(f"Listo: {out_html} con {len(items)} items para etiquetar.")
    print(f"Manifest guardado en {out_manifest} (usalo junto con labels.json en fit_thresholds.py).")


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Labeling tool</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;text-align:center;padding:20px}
#crop{max-width:400px;image-rendering:pixelated;border:2px solid #555;background:#000}
#status{margin-top:10px;font-size:14px;color:#aaa}
button{font-size:16px;margin:6px;padding:8px 16px;cursor:pointer}
#count{font-size:20px;margin-bottom:10px}
</style></head>
<body>
<div id="count"></div>
<img id="crop"/>
<div>
  <button onclick="label(0)">Indistinguishable (N)</button>
  <button onclick="label(1)">Distinguishable (Y)</button>
  <button onclick="label(2)">Ambiguous (A)</button>
</div>
<div id="status"></div>
<div><button onclick="download()">Download labels.json</button></div>
<script>
const items = __ITEMS_JSON__;
let i = 0;
let labels = {};

function render(){
  if(i >= items.length){
    document.getElementById('count').innerText = 'Listo: ' + Object.keys(labels).length + ' / ' + items.length;
    document.getElementById('crop').src = '';
    return;
  }
  document.getElementById('count').innerText = (i+1) + ' / ' + items.length;
  document.getElementById('crop').src = 'data:image/png;base64,' + items[i].b64;
  document.getElementById('status').innerText = items[i].obj_id + ' (' + items[i].episode_id + ')';
}

function label(v){
  if(i >= items.length) return;
  labels[items[i].item_id] = {obj_id: items[i].obj_id, episode_id: items[i].episode_id, label: v};
  i++;
  render();
}

function download(){
  const blob = new Blob([JSON.stringify(labels, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels.json';
  a.click();
}

document.addEventListener('keydown', (e) => {
  if(e.key === 'a' || e.key === '2') label(2);
  if(e.key === 'y' || e.key === '1') label(1);
  if(e.key === 'n' || e.key === '0') label(0);
});

render();
</script>
</body></html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav_csv", help="navigation-house_XXXXXX.csv crudo")
    parser.add_argument("--images_dir", help="carpeta images/ de ese mismo episodio")
    parser.add_argument("--scene_id", help='ej. "house_007514"')
    parser.add_argument("--episode_id", default=None)
    args = parser.parse_args()
    build(args.nav_csv, args.images_dir, args.scene_id, episode_id=args.episode_id)