# FLUX.2 Klein 4B GGUF — Staged Nodes + Character Turnaround for ComfyUI

Memory-staged ComfyUI custom nodes for running **FLUX.2 Klein 4B (distilled, 4-step)**
GGUF quantizations on **8 GB VRAM laptops**, plus a one-click **character turnaround
generator** (front / back / left / right @ 1024×1024) built to feed 3D reconstruction
pipelines such as **Pixal3D**, Trellis and Hunyuan3D.

![License](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-0.36.0%2B-green)
![VRAM](https://img.shields.io/badge/VRAM-8GB%2B-orange)

> This pack does **not** re-implement GGUF parsing or FLUX.2 math. It delegates
> quantized ops to [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and sampling
> math to current ComfyUI core nodes. It owns orchestration, validation, presets,
> caching, cleanup policy, performance modes and CPU-mode compatibility.

---

## ✨ Features

### FLUX.2 Klein GGUF Staged pack (`nodes.py`)
- **Stage-isolated memory management** — text encoder, VAE and diffusion model are
  never resident at the same time; targeted patcher release between stages.
- **4 performance modes**: `balanced` (8 GB default), `low_vram`, `high_speed`, `cpu_only`.
- **3 cleanup policies**: `auto`, `always_release`, `keep_loaded`.
- **Official Klein sampling path**: Euler + Flux2Scheduler, 4 steps, CFG 1.0 — with
  validation warnings when you deviate.
- **1–4 image ReferenceLatent stack** with per-reference megapixel budget and LRU cache.
- **Auto/tiled VAE ladder** with OOM fallbacks; official 1024×1024 + portrait/landscape
  canvas presets.
- **Prompt / reference / scheduler LRU caches** — repeat runs skip recomputation.
- **Memory report node** with per-stage timings.

### Character Turnaround pack (`character_turnaround_nodes.py`)
- **One reference image → four 1024×1024 views**: front, back, left, right.
- Front is either a **generated view** (default, same style/lighting as the others)
  or a **pixel-exact passthrough** of your reference.
- Identity via FLUX.2 `ReferenceLatent` on every view; one shared model load and one
  shared reference encode for all views.
- Per-view **OOM retry at reduced resolution**, upscaled back to target size.
- 2×2 contact-sheet output, ready for Pixal3D / texture baking workflows.

---

## 🖼️ Example output

<!-- Place a 2x2 sheet screenshot at docs/images/sheet_example.png -->
![Character sheet example](docs/images/sheet_example.png)

---

## 📦 Requirements & Complete Setup

### Step 0 — ComfyUI (skip if you already have it)
Any recent ComfyUI (tested on 0.36.0). Windows users can use the portable build or
ComfyUI-Easy-Install; Linux/macOS: `git clone https://github.com/comfyanonymous/ComfyUI`
then `pip install -r requirements.txt`.

### Step 1 — ComfyUI-GGUF (REQUIRED)

This pack does not parse GGUF files itself — it delegates all quantized loading to
ComfyUI-GGUF by city96.

**Option A — ComfyUI Manager:** Manager → *Custom Nodes Manager* → search
`ComfyUI-GGUF` → Install → restart ComfyUI [[22]].

**Option B — git (recommended):** the ComfyUI registry lists this extension as
git-install only [[24]], so cloning is the most reliable path:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
cd ComfyUI-GGUF
pip install -r requirements.txt
```
[[18]][[19]]

**Windows portable / ComfyUI-Easy-Install:** use the bundled Python instead of
system pip (run from inside the `ComfyUI-GGUF` folder, adjust `..` depth to your
install):

```bat
..\..\..\python_embeded\python.exe -s -m pip install -r requirements.txt
```
[[21]]

**Verify:** restart ComfyUI — the startup log prints a `ComfyUI-GGUF:` line and
nodes named `Unet Loader (GGUF)` / `CLIPLoaderGGUF` become available.

### Step 2 — Model files (~9 GB total)

| File | Size | Download from | Place in |
|---|---|---|---|
| `flux-2-klein-4b-Q8_0.gguf` | ~4.3 GB | [unsloth/FLUX.2-klein-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) [[2]] or [leejet/FLUX.2-klein-4B-GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF) [[4]] | `models/unet_gguf/` |
| `Qwen3-4B-Q8_0.gguf` | ~4.3 GB | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q8_0.gguf) [[13]] or [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) [[15]] | `models/clip_gguf/` |
| FLUX.2 VAE (`flux2-vae.safetensors`) | ~300 MB | [Comfy-Org/vae-text-encorder-for-flux-klein-4b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b) (ComfyUI-ready names) [[30]] or the original `vae/diffusion_pytorch_model.safetensors` from [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/vae/diffusion_pytorch_model.safetensors) [[26]] | `models/vae/` |

Create the folders if they don't exist. Command-line download (optional):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download unsloth/FLUX.2-klein-4B-GGUF flux-2-klein-4b-Q8_0.gguf --local-dir ComfyUI/models/unet_gguf
huggingface-cli download unsloth/Qwen3-4B-GGUF Qwen3-4B-Q8_0.gguf --local-dir ComfyUI/models/clip_gguf
```

**Filename rules enforced by the nodes:**
- Diffusion model must contain `klein` and be the **4B distilled** checkpoint.
  `Klein-Base` and 9B/12B+ files are rejected at load time.
- Text encoder must be **Qwen3-4B**. 8B/14B+ files are rejected.
- Lower-RAM machines can substitute `flux-2-klein-4b-Q6_K.gguf` [[6]] — quality
  drops slightly, RAM usage drops ~1 GB.

> Some Black Forest Labs repositories require accepting the license on the
> Hugging Face page before download. The unsloth GGUF mirrors above are direct
> downloads. Model weights are governed by their own licenses, not this repo's.

### Step 3 — This pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Mstafa-awad/flux-4b-gguf-comfyui-nodes_for_pixal3d_workflow.git
```

Restart ComfyUI. Nodes appear under **`Flux2 Klein GGUF Staged`** and
**`Character Turnaround`**.

### Step 4 — First-run check

1. Drag a JSON from `workflows/` onto the canvas.
2. Pick your three files in the dropdowns (they auto-filter to Klein/Qwen/FLUX2-VAE).
3. Run **FLUX2 Staged Memory Report** once — it confirms resolved mode, free
   GPU/RAM and that all three model files were found before you spend time
   generating.
   
---

## 🚀 Installation

### Option A — git clone (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Mstafa-awad/flux-4b-gguf-comfyui-nodes_for_pixal3d_workflow.git
```

Restart ComfyUI. Nodes appear under the categories
**`Flux2 Klein GGUF Staged`** and **`Character Turnaround`**.

### Option B — manual

1. Download this repository as ZIP (Code → Download ZIP).
2. Extract the folder into `ComfyUI/custom_nodes/`.
3. Restart ComfyUI.

### Load a workflow

Drag any JSON from `workflows/` onto the ComfyUI canvas, or use
Menu → Workflows → Open. Select your GGUF/VAE files in the dropdowns on first run.

---

## 🧩 Node reference

### Flux2 Klein GGUF Staged
| Node | What it does |
|---|---|
| FLUX2 Klein GGUF Loader (Staged) | Loads the Klein GGUF via ComfyUI-GGUF with patch-placement policy |
| Qwen3 4B Encode + Release (Staged) | Encodes prompt, caches it, releases the encoder per policy |
| FLUX2 Klein Canvas Presets | Official resolution presets + memory warnings |
| FLUX2 Klein 4-Step Sampler | Validated distilled path (Euler, 4 steps, CFG 1) with OOM retry |
| FLUX2 Reference Stack (1-4 Images) | Budget-fits and VAE-encodes references, applies ReferenceLatent |
| FLUX2 VAE Decode (Auto / Quality Safe) | Normal/tiled decode ladder with fallbacks |
| FLUX2 Staged Memory Report | GPU/RAM, tracked models, caches, stage timings |

### Character Turnaround
| Node | What it does |
|---|---|
| Character Turnaround (Front/Back/Left/Right) [8GB Staged] | 1 reference → 4 views + 2×2 sheet + info string |
| Character Turnaround Cache + Memory Report | Clears scheduler cache, reports memory |

#### Key turnaround widgets
| Widget | Default | Notes |
|---|---|---|
| `performance_mode` | `low_vram` | Use `balanced`/`high_speed` when you have headroom |
| `front_output_source` | `generated_front_view` | Switch to `reference_passthrough` for pixel-exact front |
| `front_fit_mode` | `fit_padding` | Keeps whole character on a white square |
| `steps` / `cfg` / `sampler` | 4 / 1.0 / euler | Klein distilled validated values — deviations warn in `info` |
| `reference_budget_megapixels` | 1.0 | Lower to 0.6 if the reference encode OOMs |
| `auto_reduce_on_oom` | on | Failed view retries at 896 → 768 px, upscaled back to 1024 |
| `swap_left_right` | off | Flip if your pipeline labels sides the other way |
| `prompt_template` | sheet template | Must contain `{view}` |

**Outputs:** `front`, `back`, `left`, `right` (each 1024×1024), `sheet_2x2`
(order: front | left / back | right), `info` (timings + diagnostics).

---

## ⚙️ Performance modes & cleanup

| Mode | Behaviour |
|---|---|
| `low_vram` | Release every stage, tiled VAE, gc after each stage. Slowest, safest. |
| `balanced` | 8 GB target: release CLIP after encode, VAE after reference, UNet after sampling. |
| `high_speed` | Keep models warm across nodes, normal VAE when free VRAM allows. |
| `cpu_only` | Emergency path; restart ComfyUI with `--cpu` for reliability. |

| Cleanup policy | Behaviour |
|---|---|
| `auto` | Follow the mode table above |
| `always_release` | Unload everything after every stage |
| `keep_loaded` | Never unload (fast repeats, needs headroom) |

Approximate timings on an 8 GB laptop GPU (1024×1024, 4 steps):
**~10–30 s per generated view** depending on mode; a full 4-view turnaround is
roughly 3–4× one view when generating the front, 3× when passing it through.

---

## 🧊 Tips for Pixal3D / 3D pipelines

- Keep `front_fit_mode = fit_padding`: full body, head to feet, white background —
  masks and silhouettes extract cleanly.
- Use the **same seed** for all views (default behaviour) so scale and pose stay
  consistent across the sheet.
- Prefer `generated_front_view` so all four panels share identical lighting and
  background — texture bakes hate mixed backgrounds.
- Feed `sheet_2x2` to tools that accept contact sheets, or the four single outputs
  to multi-view reconstructors.
- The **back view is inferred** — a single front photo contains no back information.
  Expect invented details (logos, patterns, hair back) there.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Required node 'UnetLoaderGGUFAdvanced' is missing` | Install/update ComfyUI-GGUF, restart |
| OOM during sampling | `low_vram` mode, or rely on `auto_reduce_on_oom` (on by default) |
| OOM during reference encode | Lower `reference_budget_megapixels` to 0.6 |
| Garbage / wrong colours | Wrong VAE selected — must be `flux2-vae.safetensors` |
| Output ignores my character | FLUX.2 identity comes from the reference latent; keep the reference clean, centred, full-body |
| Front looks like a re-render | That's `generated_front_view`; switch to `reference_passthrough` for exact pixels |
| Very slow on CPU | Expected; restart with `--cpu` only if you have no GPU at all |
| Widget changes seem ignored | Existing canvas nodes keep saved values — change them on the node or re-add it |

Run **FLUX2 Staged Memory Report** or **Character Turnaround Cache + Memory Report**
first when diagnosing — they print resolved mode, free GPU/RAM, tracked models and
the last stage timings.

---

## 🗂️ Repository contents

```
__init__.py                     combined node registration
nodes.py                        FLUX.2 Klein GGUF staged pack
character_turnaround_nodes.py   character turnaround pack
workflows/                      ready-to-load example workflows
docs/images/                    screenshots and example sheets
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and PRs welcome — include the
`info`/report string and your GPU/VRAM when reporting memory issues.

---

## 📜 License

**GPL-3.0-or-later** — Copyright (C) 2026 Mostafa Awad.
See [LICENSE](LICENSE). Model weights are governed by their own licenses
(Black Forest Labs FLUX.2, Qwen3).

## 🙏 Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) by city96
- FLUX.2 Klein by Black Forest Labs; Qwen3 by Alibaba Qwen team
- Pixal3D and the 3D-reconstruction community for the turnaround use-case