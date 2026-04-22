# SEM examples

These examples convert a raw SEM micrograph into an RVE mesh.

Two routes are supported:

1. **`sem_to_synthetic.yaml`** — segment the SEM image, fit each fibre to a
   circle (or ellipse) and rebuild it as a clean synthetic RVE.
   Use when the micrograph contains roughly circular fibre cross-sections and
   you want a smooth mesh.

2. **`import.yaml`** — threshold the SEM image directly into a binary fibre
   mask and import the contours as polygonal fibres.
   Use when you want to preserve the raw fibre outlines.

3. **`masked_import.yaml`** — same as `import.yaml` but reads a cleaned
   `preprocessed_mask.png` produced by `tools/generate_test_image_preview.py`.

## Required input

You must provide your own SEM micrograph at:

```
examples/2d/sem/sem_sample.png
```

Any greyscale image readable by `skimage.io.imread` works (`.png`, `.tif`,
`.webp`, …). Update the `image_path` field if you store it elsewhere.

## Pixel size

`pixel_size` is the physical edge length (in metres or whatever unit you are
working in) of one pixel. Set it to your SEM scale-bar calibration.
