# SideViewCows2026 – Dairy Cow Re-Identification Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21605650.svg)](https://doi.org/10.5281/zenodo.21605650)

An image dataset for individual re-identification of dairy cows, comprising
**80,260 images** with corresponding segmentation masks **of 110 individual animals**,
recorded in **three different settings**.

All animals are photographed **from the right-hand side**.

The herd is predominantly **Holstein Friesian**, with a smaller number of
Fleckvieh (Simmental), one Jersey, and one Brown Swiss animal. Coat patterns
are therefore mostly black-and-white. The dataset is not balanced across
breeds.

![Example images from all three subsets, one individual per row, with the
segmentation mask overlaid](preview.jpg)

*Each row shows the same animal in `parlor`, `barn` and `snapshots`; the
segmentation mask is overlaid in green with a red outline.*

## Contents

| Subset | Images | Individuals | Description |
|---|---|---|---|
| `parlor` | 54,393 | 110 | Video frames from a fixed camera at the milking parlor entrance. Consistent viewpoint and framing. |
| `barn` | 25,260 | 69 | Frames from handheld video recorded in the barn. Varying viewpoint and camera motion. |
| `snapshots` | 607 | 63 | Individual photographs in varied settings. Camera angle varies considerably more than in the other subsets; locations differ (indoors and outdoors), and some animals are lying down in their cubicles. |

The subsets are **nested**: the 63 individuals in `snapshots` are a subset of the
69 in `barn`, which are in turn a subset of the 110 in `parlor`. Identifiers are
consistent across all three subsets, so the same animal carries the same
`individual_id` everywhere.

This structure supports cross-domain evaluation. `parlor` provides a large
gallery under controlled conditions, while `barn` and `snapshots` offer query
images under progressively greater domain shift. The 63 shared individuals are
the ones for which images exist in every setting.

## Directory layout

```
.
├── manifest.csv
├── SHA256SUMS
├── README.md
├── preview.jpg
├── parlor/
│   ├── images/
│   │   ├── 128/
│   │   │   ├── 128_0001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── masks/
│       ├── 128/
│       │   ├── 128_0001.png
│       │   └── ...
│       └── ...
├── barn/
│   ├── images/
│   └── masks/
└── snapshots/
    ├── images/
    └── masks/
```

Each subset contains `images/` and `masks/`, subdivided into one directory per
individual. Files follow the pattern `<individual_id>_<nnnn>`, numbered
consecutively per individual and ordered chronologically. Image and mask share
the same filename stem, so the mask for
`parlor/images/128/128_0001.jpg` is `parlor/masks/128/128_0001.png`.

**Images** are JPEG. Pixels are at native recording resolution — no image has
been up- or downscaled — but the `parlor` frames are **cropped to a region of
interest** rather than showing the full camera frame. Image dimensions therefore
vary; the exact size of every image is given in `manifest.csv`.
**Masks** are PNG, binary (white foreground on black background), the same
dimensions as the corresponding image. The foreground is the silhouette of the
animal the image belongs to.

## manifest.csv

One row per image.

| Column | Description |
|---|---|
| `image_path` | Path to the image, relative to the dataset root |
| `mask_path` | Path to the corresponding mask |
| `individual_id` | Identifier of the animal, consistent across all subsets |
| `subset` | `parlor`, `barn` or `snapshots` |
| `frame_no` | Consecutive number of the image within that individual and subset |
| `time_offset_s` | Time of capture in seconds, relative to the earliest image in the dataset (see below) |
| `width`, `height` | Image dimensions in pixels |
| `sha256` | SHA-256 checksum of the image file |

### time_offset_s

Absolute capture dates have been removed. `time_offset_s` gives the elapsed
time in seconds relative to a single global reference point: the earliest
`parlor` frame, which is defined as 0. All intervals between recordings are
therefore preserved — both within and between subsets — while no absolute date
can be derived. For `barn`, the value is the start time of the source video and
so is identical for all frames of a recording.

## Privacy measures

- Regions in which people were detected are overwritten with solid magenta,
  RGB (255, 0, 255). Detection was carried out with a deliberately
  recall-oriented procedure and reviewed manually. Owing to JPEG compression,
  pixels at the edges of these regions may deviate slightly from the exact
  color value; a tolerance-based comparison locates them reliably.
- `individual_id` is a pseudonym and does not correspond to the herd numbers
  used on the farm.
- Filenames and directory names contain no capture dates or holding
  identifiers.

## Verifying integrity

`SHA256SUMS` covers every file in the dataset:

```bash
shasum -c SHA256SUMS
```

## Loading the data

```python
import pandas as pd
from pathlib import Path
from PIL import Image

root = Path(".")
df = pd.read_csv(root / "manifest.csv")

# All images of one individual, across every subset
subject = df[df.individual_id == df.individual_id.iloc[0]]

# Gallery/query split across domains
gallery = df[df.subset == "parlor"]
query   = df[df.subset == "snapshots"]

row  = df.iloc[0]
img  = Image.open(root / row.image_path)
mask = Image.open(root / row.mask_path)
```

## Notes on use

Frames from the same recording are strongly correlated. When constructing
evaluation splits, group by individual and recording rather than sampling
frames at random, or performance figures will be optimistic.

## License

Released under the Creative Commons Attribution 4.0 International license
(**CC BY 4.0**). You may share and adapt the material for any purpose, including
commercially, provided you give appropriate credit.
See <https://creativecommons.org/licenses/by/4.0/>.

## Contact and maintenance

Maintained by Sebastian Möller, Osnabrück University of Applied Sciences —
[s.moeller@hs-osnabrueck.de](mailto:s.moeller@hs-osnabrueck.de), ORCID [0009-0008-5364-6506](https://orcid.org/0009-0008-5364-6506).

Questions about the dataset, corrections, and reports of errors are welcome.
If you find an image in which a person is still identifiable, please report it
to the address above; such images will be corrected or removed in a revised
version.

## Citation

If you use this dataset, please cite it directly:

> Möller, S. (2026). SideViewCows2026 – Dairy Cow Re-Identification Dataset
> [Dataset]. Zenodo. <https://doi.org/10.5281/zenodo.21605650>

```bibtex
@dataset{Moeller2026SideViewCows,
  author    = {M{\"o}ller, Sebastian},
  title     = {{SideViewCows2026 -- Dairy Cow Re-Identification Dataset}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21605650},
  url       = {https://doi.org/10.5281/zenodo.21605650}
}
```

The dataset was generated based on the pipeline described in:

> Möller, S., Hölscher, M., & Morisse, K. (2025). Automatisierte Erzeugung eines
> Trainingsdatensatzes zur bildbasierten Tieridentifikation mittels KI. In
> *Lecture Notes in Informatics: Proceedings*, Vol. 2025, No. 358 (pp. 351–356).
> Gesellschaft für Informatik e.V. (GI).
> <https://dl.gi.de/handle/20.500.12116/45701>

```bibtex
@inproceedings{MoellerHoelscherMorisse2025,
  author       = {M{\"o}ller, Sebastian and H{\"o}lscher, Matthias and Morisse, Karsten},
  title        = {Automatisierte Erzeugung eines Trainingsdatensatzes zur
                  bildbasierten Tieridentifikation mittels KI},
  booktitle    = {Lecture Notes in Informatics : Proceedings},
  volume       = {2025},
  number       = {358},
  organization = {Gesellschaft f{\"u}r Informatik e.V. (GI)},
  isbn         = {978-3-88579-802-6},
  issn         = {1617-5468},
  pages        = {351--356},
  year         = {2025},
  url          = {https://dl.gi.de/handle/20.500.12116/45701},
  language     = {de}
}
```