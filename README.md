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
| `asl` | [HF asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) | 26 | 10,613 | **0.799** held-out |

Reproduce any row with `./sign evaluate <lang>`.

Held-out figures use a **contiguous** split that keeps mirrored pairs and adjacent video frames together.
Random k-fold on this data reports 0.99+ because consecutive frames of a held pose are near-duplicates
and leak across folds — that number is not meaningful and is not reported here.

The live figure was measured by performing each sign fresh at the camera, before mirror/rotation
augmentation was added. Augmentation raised accuracy on simulated rotated poses from **0.761 to 0.972**;
the live test was not repeated after.

### The interesting part: landmarks don't survive a new signer

`isldata` scores 0.989 held-out and is useless on the author's hands. Measured directly:

- distance between two samples of the same class, within the dataset: **0.184**
- distance from a live hand signing A to the nearest training point: **1.734** (~9×)
- prediction: **D**, at confidence 1.00

k-NN confidence is meaningless here — it reports the nearest class, not whether that class is plausible.
The same effect reappears within a single signer's own data: `isl` scores 0.999 cross-validated and
0.71 live. **Public-dataset accuracy does not predict live accuracy**, which is why the demo model is
trained on recordings from the person demoing it. This is the project's actual finding.

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
- **Dataset-trained models do not transfer to new signers** (see above).

## Tests

```bash
./sign selftest                     # feature invariances, mirroring, jitter, rejection, I/O
.venv/bin/python test_run.py        # drives the live loop with a fake camera, no display
```

## Licence

MIT
