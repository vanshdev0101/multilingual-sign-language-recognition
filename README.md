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

Live figures were measured by performing each sign fresh at the camera with `./sign livetest`, after
mirror/rotation augmentation. The 17 `isl` letters last measured 12/17 live before augmentation was
added and were not re-tested after; the 9 digits measured 9/9 after.

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

The remaining `isldata` errors (1 -> I, 2 -> V, 3 -> 8) are a performance gap, not label ambiguity:
inside the dataset the digits are perfectly separable, **1081/1081 held-out, every digit at 1.000**.
The prediction was confirmed directly - re-recording the same nine digits from the author's own hands
and retraining gives **9/9 live at distances of 0.14-0.34**. The handshapes simply differed from that
dataset signer's.

Centroid distance suggests ambiguity where there is none (1 vs I gap 0.80) and is the wrong metric
here: ingest mirrors every image, so each class holds two mirrored clusters and its apparent radius is
inflated. k-NN handles multi-modal classes; centroid separation does not describe it.

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
- **Transfer is per-handshape, not per-model.** A dataset model recognises signs you perform the way
  its signers did. Where your handshape differs, re-record that label; the pipeline mixes sources fine.

## Tests

```bash
./sign selftest                     # feature invariances, mirroring, jitter, rejection, I/O
.venv/bin/python test_run.py        # drives the live loop with a fake camera, no display
```

## Licence

MIT

## Web demo

`web/` is a static page that runs the whole pipeline in the browser — MediaPipe's JS hand
tracker plus the exported k-NN model (`model.bin`, `model.json`). No server, no upload: the
video never leaves the machine.

```bash
cd web && python3 -m http.server 8000   # then open http://localhost:8000
```

It also lets a visitor **teach it their own hand**. Hands differ in size and angle, and a model
trained on one signer misreads another — the finding above, turned into a feature. Recording a
sign in the browser stores those samples in `localStorage` and matches them ahead of the shipped
ones, so the page adapts to whoever is using it without retraining anything.
