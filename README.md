# Multilingual Sign Language Recognition

Real-time static sign recognition for **Indian Sign Language (ISL)** and **American Sign Language (ASL)**
from a webcam. Hand landmarks (MediaPipe) feed a per-language k-NN classifier, so a language is just a
dataset plus a model file — switchable live with one keypress.

Runs at ~30 FPS on CPU. No GPU, no training infrastructure.

## Why landmarks instead of pixels

A raw-pixel CNN has to learn lighting, skin tone, background and camera framing before it learns
handshape. MediaPipe discards all of that first and returns 21 3-D points per hand, so the classifier
only ever sees geometry. In practice this is what makes a model trained on one source survive a
different camera — and, as the results below show, it is still not enough to survive a different signer.

**Feature vector (126-D).** Two fixed hand slots (Left, Right) × 21 landmarks × (x, y, z). Each hand is
translation-normalised (wrist at origin) and scale-normalised (divided by its own max landmark
distance), so distance from the camera and position in frame do not matter. An absent hand is zeros,
which also encodes one-handed vs two-handed signs — the distinction that separates ISL from ASL
fingerspelling.

Two corrections that materially affect accuracy:

- **Aspect correction.** MediaPipe normalises `x` by image *width* and `y` by *height*, so the same hand
  in a square training image and a 4:3 webcam frame differ by a 4:3 stretch. `x` is scaled by `w/h`.
- **Out-of-distribution rejection.** k-NN always returns its nearest class, however far away the sample
  is. Each model stores a rejection radius derived from its own k-th-neighbour spread; beyond it the
  prediction is reported as `?` rather than a confident wrong letter.

**Augmentation** (training only): each recorded sample is mirrored (swap hand slots, negate `x`) so either
hand works, and rotated ±12° four times, because a held pose produces near-identical frames and a live
sign rotated a few degrees otherwise falls outside its own class.

## Results

| Model | Source | Classes | Samples | Accuracy |
|---|---|---|---|---|
| `isl` | own recordings | 17 | 1,020 → 10,200 aug | **12/17 live** (see note) |
| `isldata` | [Kaggle ISL](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) | 35 | 13,396 | **0.993** held-out |
| `asl` | [HF asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) | 26 | 10,613 | **0.799** held-out, **5/6 live** |

Reproduce any row with `./sign evaluate <lang>`.

Held-out figures use a **contiguous** split that keeps mirrored pairs and adjacent video frames together.
Random k-fold on this data reports 0.99+ because consecutive frames of a held pose are near-duplicates
and leak across folds — that number is not meaningful and is not reported here.

Live figures were measured by performing each sign fresh at the camera with `./sign livetest`. The `isl`
live figure predates mirror/rotation augmentation and was not re-measured after; the `asl` figure is
post-augmentation.

### The interesting part: augmentation is what makes a dataset model transfer

Trained naively, `isldata` scores 0.993 held-out and is useless on a new signer. Measured on the
author's hand signing A:

- distance between two samples of the same class, within the dataset: **0.184**
- distance from the live hand to the nearest training point: **1.734** (~9x)
- prediction: **D**, at confidence 1.00

k-NN confidence is worthless here: it reports the nearest class, not whether that class is plausible.
This is the usual reason a model with a headline accuracy fails in front of a camera.

The cause is pose variance, not the representation. A dataset of held poses, and recordings of held
poses, both produce near-identical samples (class radius ~0.05), so a live hand rotated a few degrees
falls outside every class. Adding mirrored and +/-12 deg rotated copies at training time widens each
class to cover variation a signer actually produces. On rotated held-out poses this lifts accuracy from
**0.761 to 0.972**, and it changes live behaviour qualitatively:

| live test (new signer, dataset-trained model) | result | median distance |
|---|---|---|
| `isldata`, before augmentation | A predicted as **D** | 1.734 |
| `asl`, after augmentation | **5/6** letters correct | 0.19 - 0.39 |
| `isldata` digits, after augmentation | 2/5 correct, but matched at close range | 0.25 - 0.73 |

The remaining `isldata` errors are not transfer failures. They were 1 -> I, 2 -> V, 3 -> 8, and those
classes overlap *within the source dataset*: 1 vs I have a centroid gap of 0.80 against class radii of
1.65 and 1.98, so the gap is smaller than the spread. The digits are the closest cluster in the whole
35-class set (3-4 at 0.40). Those labels are not separable from the source data, with or without a
new signer.

**Conclusion: a dataset-trained landmark model does transfer to a new signer, provided training covers
pose variation the dataset itself does not contain.** Reported cross-validation accuracy says nothing
about either property.

## Usage

```bash
pip install -r requirements.txt
curl -sSfL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

./sign train asl                         # regenerates models/asl.pkl (52 MB, not in repo)

./sign probe                             # camera + detector check, no window
./sign collect isl A B C -n 60 --delay 8 # hands-free: countdown, then auto-record
./sign train isl
./sign livetest isl A B C                # accuracy on fresh performances (live, at the camera)
./sign evaluate asl                      # held-out accuracy on a stored dataset
./sign run isl asl                       # live demo; 'l' switches language
```

`collect` is countdown-driven rather than keypress-gated because two-handed signs leave no hand free for
the keyboard. `forget <lang> <labels>` drops labels so they can be re-recorded. `ingest <lang> <folder>`
converts a folder of class-named image subfolders into landmark features. `--cam N` maps to `/dev/videoN`;
blank devices are skipped automatically.

## Limitations

- **Static poses only.** Signs distinguished by motion are out of scope, including ASL **J** and **Z**.
  Upgrade path: an LSTM/GRU over a 30-frame landmark window — the feature extractor already emits what
  it would consume.
- **K and V collide** in the ISL model (centroid gap 0.26 against a class radius of 0.03) — as performed,
  they are the same handshape.
- **Two-handed signs need both hands in frame.** If one leaves the frame the vector matches a one-handed
  sign instead. This caused every W→H error observed.
- **Digits 1-9 in the Kaggle ISL set are not reliably separable** from each other or from their letter
  lookalikes (1/I, 2/V, 3/8) - an ambiguity in the source labels, not the model.

## Tests

```bash
./sign selftest                     # feature invariances, mirroring, jitter, rejection, I/O
.venv/bin/python test_run.py        # drives the live loop with a fake camera, no display
```

## Licence

MIT
