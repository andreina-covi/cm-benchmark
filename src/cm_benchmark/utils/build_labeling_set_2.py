import base64
import io
import json
import os
import pandas as pd
from PIL import Image

PADDING_PX = 6
THUMB_MAX_SIDE = 220  # to avoid the HTML being too heavy with hundreds of crops

# Features saved in calibration_manifest.csv so fit_thresholds.py
# doesn't have to read the raw CSV again. Names exactly as they come from your
# navigation-*.csv (with hyphens, not underscores).
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
                            Image.NEAREST)  # NEAREST to avoid "inventing" detail in small objects

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def full_image_b64(image_path):
    """Full scene image, without cropping, to give context of the timestep."""
    im = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(nav_csv, images_dir, scene_id, episode_id=None,
          out_html="labeling_tool.html", out_manifest="calibration_manifest.csv",
          W=396, H=224, only_real_mask=True):
    """
    nav_csv:    navigation-house_XXXXXX.csv (raw, as exported by collector.py)
    images_dir: images/ folder of that episode (sister of annotations/, see DATA_COLLECTION.md)
    scene_id:   e.g. "house_007514" -- not in the CSV, have to pass it manually
    episode_id: if omitted, use the name of the nav_csv as the episode id
    """
    episode_id = episode_id or os.path.basename(nav_csv)
    df = pd.read_csv(nav_csv)

    if only_real_mask:
        df = df[df["visible-pixels"].notna()].copy()

    df = df.reset_index(drop=True)
    df["item_id"] = df.index.astype(str)
    df["scene_id"] = scene_id
    df["episode_id"] = episode_id

    items = []
    manifest_rows = []
    images_cache = {}  # path -> b64, to avoid repeating the full scene for each object

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row["path"])
        try:
            b64 = make_crop_b64(image_path, row.cmin, row.rmin, row.cmax, row.rmax, W, H)
        except Exception as e:
            print(f"WARN: could not crop {row.get('obj-id')} in {image_path}: {e}")
            continue

        if row["path"] not in images_cache:
            try:
                images_cache[row["path"]] = full_image_b64(image_path)
            except Exception as e:
                print(f"WARN: could not load the full scene {image_path}: {e}")
                images_cache[row["path"]] = None

        items.append({
            "item_id": row["item_id"],
            "obj_id": row.get("obj-id", ""),
            "episode_id": episode_id,
            "timestep": row.get("timestep", ""),
            "b64": b64,
            "image_key": row["path"],
            # position of the object within the full scene, to draw the rectangle
            "bbox": {"cmin": row.cmin, "rmin": row.rmin, "cmax": row.cmax, "rmax": row.rmax,
                     "W": W, "H": H},
        })
        manifest_rows.append({
            "item_id": row["item_id"],
            "obj_id": row.get("obj-id", ""),
            "scene_id": scene_id,
            "episode_id": episode_id,
            **{c: row.get(c) for c in FEATURE_COLUMNS},
        })

    html = _HTML_TEMPLATE.replace("__ITEMS_JSON__", json.dumps(items))
    html = html.replace("__IMAGES_JSON__", json.dumps(images_cache))
    with open(out_html, "w") as f:
        f.write(html)

    pd.DataFrame(manifest_rows).to_csv(out_manifest, index=False)
    print(f"Ready: {out_html} with {len(items)} items to label.")
    print(f"Manifest saved in {out_manifest} (use it together with labels.json in fit_thresholds.py to calibrate the thresholds).")


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Labeling tool</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;text-align:center;padding:20px}
#crop{max-width:400px;image-rendering:pixelated;border:2px solid #555;background:#000}
#status{margin-top:10px;font-size:14px;color:#aaa}
button{font-size:16px;margin:6px;padding:8px 16px;cursor:pointer}
#count{font-size:20px;margin-bottom:10px}
#scene-wrap{position:relative;display:inline-block;margin-top:18px;border:1px solid #444}
#scene{display:block;width:500px;image-rendering:pixelated}
#bbox-overlay{position:absolute;border:2px solid #ff4d4d;box-shadow:0 0 0 1px rgba(0,0,0,.6);pointer-events:none}
#scene-label{font-size:12px;color:#888;margin-top:4px}
</style></head>
<body>
<div id="count"></div>
<div><strong>Object crop</strong></div>
<img id="crop"/>
<div>
  <button onclick="label(0)">Indistinguishable (N)</button>
  <button onclick="label(1)">Distinguishable (Y)</button>
  <button onclick="label(2)">Ambiguous (A)</button>
</div>
<div id="status"></div>
<div id="scene-wrap">
  <img id="scene"/>
  <div id="bbox-overlay"></div>
</div>
<div id="scene-label">Full scene (timestep) -- the red rectangle marks the object</div>
<div><button onclick="download()">Download labels.json</button></div>
<script>
const items = __ITEMS_JSON__;
const images = __IMAGES_JSON__;  // { path: b64_png }
let i = 0;
let labels = {};

function render(){
  if(i >= items.length){
    document.getElementById('count').innerText = 'Listo: ' + Object.keys(labels).length + ' / ' + items.length;
    document.getElementById('crop').src = '';
    document.getElementById('scene').src = '';
    document.getElementById('bbox-overlay').style.display = 'none';
    return;
  }
  const it = items[i];
  document.getElementById('count').innerText = (i+1) + ' / ' + items.length;
  document.getElementById('crop').src = 'data:image/png;base64,' + it.b64;
  document.getElementById('status').innerText = it.obj_id + '  (timestep ' + it.timestep + ', ' + it.episode_id + ')';

  const sceneB64 = images[it.image_key];
  const sceneImg = document.getElementById('scene');
  const overlay = document.getElementById('bbox-overlay');
  if(sceneB64){
    sceneImg.src = 'data:image/png;base64,' + sceneB64;
    // The overlay is positioned in % relative to the native size of the camera (bbox.W/H),
    // so it scales well even though the <img> is displayed larger via CSS.
    const b = it.bbox;
    overlay.style.display = 'block';
    overlay.style.left   = (100 * b.cmin / b.W) + '%';
    overlay.style.top    = (100 * b.rmin / b.H) + '%';
    overlay.style.width  = (100 * (b.cmax - b.cmin) / b.W) + '%';
    overlay.style.height = (100 * (b.rmax - b.rmin) / b.H) + '%';
  } else {
    sceneImg.src = '';
    overlay.style.display = 'none';
  }
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
    parser.add_argument("--images_dir", help="images/ folder of that same episode")
    parser.add_argument("--scene_id", help='e.g. "house_007514"')
    parser.add_argument("--episode_id", default=None)
    args = parser.parse_args()
    build(args.nav_csv, args.images_dir, args.scene_id, episode_id=args.episode_id)