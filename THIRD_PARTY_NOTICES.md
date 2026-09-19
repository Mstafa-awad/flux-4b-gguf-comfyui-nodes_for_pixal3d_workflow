# Third-Party Notices

This repository contains original orchestration code and workflow files. It
does not contain ComfyUI, ComfyUI-GGUF, model implementation source copied from
those projects, or any model weights.

The project calls public runtime interfaces exposed by the following separately
installed projects:

| Project | Role | Upstream license / terms |
|---|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Host application and native FLUX.2 sampling nodes | [GNU GPL v3](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | GGUF model loading and quantized operations | [Apache License 2.0](https://github.com/city96/ComfyUI-GGUF/blob/main/LICENSE) |
| [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Diffusion model architecture and official weights | [Apache License 2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B#license) |
| [Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B) | Text encoder model | [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-4B) |
| [Unsloth GGUF conversions](https://huggingface.co/unsloth) | Optional quantized model downloads | Terms and provenance shown on each repository's model card |

## Model weights

Model files are not part of this repository or its GPL license. Users download
them directly from their respective publishers and must comply with the model
card, license, acceptable-use rules, and local law that apply to each file.

This repository targets the **4B** FLUX.2 Klein release. Do not assume that the
license for the separate 9B checkpoint is identical.

## Names and trademarks

ComfyUI, FLUX, Qwen, Unsloth, GGUF, and related project names are used only to
identify compatibility. All trademarks and logos belong to their respective
owners. No affiliation, sponsorship, or endorsement is claimed.

## Documentation screenshots

The screenshots in `docs/images/` were supplied by the project author to
document this node pack. They depict the separately licensed ComfyUI user
interface. Repository distribution of those screenshots is authorized for this
documentation; underlying third-party rights remain with their owners.

This notice is informational and is not legal advice.
