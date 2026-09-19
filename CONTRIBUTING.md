# Contributing

Thank you for helping improve the project.

## Before opening an issue

1. Update ComfyUI, ComfyUI-GGUF, and this repository.
2. Restart ComfyUI completely.
3. Reproduce the problem with one of the included workflows.
4. Include the full console traceback, GPU model, VRAM, system RAM, model
   filenames, image dimensions, and the workflow JSON.

Do not upload model weights, private prompts, personal images, API keys, or
other secrets.

## Pull requests

- Keep the staged Qwen → FLUX → VAE memory behavior intact.
- Preserve compatibility with current ComfyUI and ComfyUI-GGUF.
- Validate every changed workflow as JSON.
- Add a short entry to `CHANGELOG.md` for user-visible changes.
- Use an SPDX identifier in new source files.

By contributing, you certify that you have the right to submit the work and
agree that it is distributed under GPL-3.0-or-later. Do not submit code, images,
or other material copied from a source whose license is incompatible or
unknown.
