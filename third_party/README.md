# Third-party code

## `ttc/`: Test-time Counterattacks (TTC)

Copied from the official TTC repository by Songlong Xing, Zhengyu Zhao and Nicu Sebe:
[github.com/Sxing2/CLIP-Test-time-Counterattacks](https://github.com/Sxing2/CLIP-Test-time-Counterattacks)
("CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial
Robustness of CLIP", CVPR 2025). The upstream repository does not declare a license; all rights remain with
the original authors, the files are redistributed unchanged except as noted below, and they are excluded from
this repository's MIT license. Please cite their work.

SAVE-TTC uses these files for exactly what the published numbers used:

| File | Used for |
|---|---|
| `replace/` | CLIP ViT-B/32 with prompt-token support (modified OpenAI CLIP, MIT) and the torchvision-style dataset classes with `clip_prompts` |
| `utils.py` | `load_val_dataset`, `get_text_prompts_val`, `convert_models_to_fp32` |
| `attacks.py` | `attack_pgd` (PGD against undefended CLIP for the standard protocol) |
| `func.py` | `clip_img_preprocessing` |
| `models/prompters.py` | `TokenPrompter` (empty prompt token, `prompt_len=0`) |

Changes with respect to the original files (no numerical change):

- `attacks.py`: the `autoattack` import moved inside `attack_auto` so that the package is optional.
- `replace/clip.py`: `from pkg_resources import packaging` replaced by `import packaging.version`
  (`pkg_resources` is deprecated in recent setuptools).