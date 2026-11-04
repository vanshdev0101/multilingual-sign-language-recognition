#!/usr/bin/env python3
"""Multilingual static sign detector: MediaPipe hand landmarks -> KNN, one model per language.

  python sign.py probe                   # headless camera + landmarker check
  python sign.py collect isl A [-n 60]   # hold the pose, SPACE starts/pauses recording
  python sign.py train isl
  python sign.py run isl asl             # live demo; 'l' cycles language, 'c' clears text, 'q' quits
  python sign.py selftest

ponytail: static poses only (no motion signs). Upgrade path = swap KNN for a small
LSTM/GRU over a 30-frame landmark window; feature extraction below stays as-is.
"""
import sys, time, pickle, argparse
import collections
from collections import deque, Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / "data", ROOT / "models"
TASK = MODELS / "hand_landmarker.task"
SLOTS = ("Left", "Right")          # fixed slot order so the vector is deterministic
DIM = len(SLOTS) * 21 * 3          # 126


def features(hands):
    """hands: list of (handedness, (21,3) landmarks) -> (126,) vector.

    Each hand is made translation-invariant (wrist at origin) and scale-invariant
    (divided by its own max landmark distance), so distance from the camera and
    position in frame don't matter. Absent hand = zeros, which also encodes
    one-handed vs two-handed signs (ASL alphabet vs ISL alphabet).
    """
    vec = np.zeros((len(SLOTS), 21, 3), dtype=np.float32)
    for handedness, lm in hands:
        if handedness not in SLOTS:
            continue
        lm = np.asarray(lm, dtype=np.float32).reshape(21, 3)
        lm = lm - lm[0]
        scale = np.linalg.norm(lm, axis=1).max()
        vec[SLOTS.index(handedness)] = lm / scale if scale > 1e-6 else lm
    return vec.reshape(-1)


def mirror_features(v):
    """Same sign performed with the other hand: swap the two hand slots, negate x.

    Landmarks are already wrist-centred, so mirroring the image negates x about the
    wrist and exchanges the Left/Right slots. Training on both makes a one-handed sign
    hand-agnostic -- otherwise signing with the other hand lands in the empty slot and
    the distance explodes (measured: d=3.56 for a class whose own radius is 0.05).
    """
    m = np.asarray(v, np.float32).reshape(len(SLOTS), 21, 3).copy()
    m[:, :, 0] *= -1.0
    return m[::-1].reshape(-1)


def jitter_features(v, rng, max_deg=12.0, noise=0.012):
    """One plausible re-performance of the same sign: small in-plane rotation + noise.

    Recorded bursts are near-identical (radius ~0.05) because a sign is held still,
    so a live pose rotated a few degrees falls outside the class and matches a
    neighbour instead. Rotating about the image axis is the variation a signer
    actually produces, so generating it widens each class without more recording.
    Empty hand slots stay empty -- noise there would invent a second hand.
    """
    m = np.asarray(v, np.float32).reshape(len(SLOTS), 21, 3).copy()
    occupied = np.abs(m).sum(axis=(1, 2)) > 1e-6
    th = np.radians(rng.uniform(-max_deg, max_deg))
    c, sn = np.cos(th), np.sin(th)
    x, yy = m[:, :, 0].copy(), m[:, :, 1].copy()
    m[:, :, 0], m[:, :, 1] = x * c - yy * sn, x * sn + yy * c
    m[occupied] += rng.normal(0, noise, m[occupied].shape).astype(np.float32)
    m[~occupied] = 0.0
    return m.reshape(-1)


# ---------- storage ----------

def dataset_path(lang):
    return DATA / f"{lang}.npz"


def load_dataset(lang):
    p = dataset_path(lang)
    if not p.exists():
        return np.empty((0, DIM), np.float32), []
    z = np.load(p, allow_pickle=False)
    return z["X"], [str(v) for v in z["y"]]


def save_dataset(lang, X, y):
    DATA.mkdir(exist_ok=True)
    np.savez(dataset_path(lang), X=np.asarray(X, np.float32), y=np.asarray(y, dtype="U32"))


# ---------- mediapipe tasks (imported lazily: selftest/train need no camera stack) ----------

def open_hands(max_hands=2, still=False):
    from mediapipe.tasks.python import vision, BaseOptions
    if not TASK.exists():
        sys.exit(f"missing {TASK}\ncurl -sSfL -o {TASK} https://storage.googleapis.com/"
                 "mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK)),
        running_mode=vision.RunningMode.IMAGE if still else vision.RunningMode.VIDEO,
        num_hands=max_hands, min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))


def detect(landmarker, frame_bgr, ts_ms=None):
    import cv2, mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect(img) if ts_ms is None else landmarker.detect_for_video(img, ts_ms)


def read_hands(res, shape=None):
    """shape: the (h, w, ...) of the source image, so x can be aspect-corrected.

    MediaPipe normalises x by image WIDTH and y by HEIGHT, so the same physical hand
    yields different landmarks in a square training image than in a 4:3 webcam frame
    -- the shape gets stretched horizontally. Multiplying x by w/h puts both in the
    same square coordinate space. Without this, training on square crops and running
    on a 4:3 camera compares distorted shapes.
    """
    aspect = (shape[1] / shape[0]) if shape is not None else 1.0
    out = []
    for i, lms in enumerate(getattr(res, "hand_landmarks", None) or []):
        handed = getattr(res, "handedness", None) or []
        label = handed[i][0].category_name if i < len(handed) and handed[i] else "Right"
        out.append((label, np.array([[p.x * aspect, p.y, p.z * aspect] for p in lms], np.float32)))
    return out


def draw(frame, res):
    import cv2
    from mediapipe.tasks.python.vision import HandLandmarksConnections
    h, w = frame.shape[:2]
    conns = HandLandmarksConnections.HAND_CONNECTIONS
    for lms in getattr(res, "hand_landmarks", None) or []:
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for c in conns:
            a, b = (c.start, c.end) if hasattr(c, "start") else (c[0], c[1])
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)


def looks_blank(frame, min_std=8.0):
    """True for an all-black / covered-lens frame.

    A device can open and hand back frames while the sensor shows nothing (lens
    cover, disabled camera): the built-in webcam here reads mean 12 / std 2, a real
    image reads mean 119 / std 72. Contrast, not brightness, is what separates them.
    ponytail: fixed threshold; if a genuinely dark room trips it, pass --cam N to override.
    """
    return float(np.asarray(frame).std()) < min_std


def camera(index=None, warmup=6):
    """Open the first index that yields a frame with actual content.

    With the V4L2 backend index N == /dev/videoN, so --cam 2 is the Razer Kiyo and
    --cam 0 the built-in webcam. Some nodes are metadata-only and never open; others
    open but stay black. An explicit --cam is honoured even if it looks blank.
    """
    import cv2
    explicit = index is not None
    for i in ([index] if explicit else range(6)):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            frame = None
            for _ in range(warmup):          # let auto-exposure settle
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is not None:
                blank = looks_blank(frame)
                if explicit or not blank:
                    print(f"camera: index {i}" + ("  [warning: frames look blank -- lens cover?]" if blank else ""))
                    return cap
                print(f"camera: skipping index {i}, frames are blank (std={frame.std():.1f})")
        cap.release()
    sys.exit("no working camera found (tried 0-5). Close other apps using it, or pass --cam N")


# ---------- commands ----------

def cmd_probe(frames=30, cam=None):
    """Headless: prove camera + landmarker + feature extraction work end to end."""
    cap, hands, seen = camera(cam), open_hands(), 0
    t0 = time.monotonic()
    try:
        for i in range(frames):
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); return
            hs = read_hands(detect(hands, frame, int((time.monotonic() - t0) * 1000) + i), frame.shape)
            if hs:
                seen += 1
                v = features(hs)
                print(f"frame {i}: {[h for h, _ in hs]} dim={v.shape[0]} nonzero={int((v != 0).sum())}")
    finally:
        cap.release(); hands.close()
    print(f"probe: {frames} frames, hands detected in {seen}"
          + ("" if seen else "  (no hand in view -- show a hand and rerun)"))


def cmd_collect(lang, labels, n, cam=None, delay=5):
    """Record several labels hands-free: a countdown per label, then auto-record n frames.

    Two-handed signs mean you cannot reach the keyboard, so nothing is keypress-gated:
    get into position during the countdown and it records on its own.
    """
    import cv2
    hands, cap = open_hands(), camera(cam)
    X, y = load_dataset(lang)
    X, y = list(X), list(y)
    t0, i, saved, quit_all = time.monotonic(), 0, 0, False
    print(f"[{lang}] {len(labels)} labels: {', '.join(labels)} -- {delay}s to get into each pose")
    print("hands-free: just hold the sign.   s skip   q quit (keeps what is saved)")
    for idx, label in enumerate(labels):
        new_samples = []
        ready_at = time.monotonic() + delay
        while len(new_samples) < n and not quit_all:
            ok, frame = cap.read()
            if not ok:
                quit_all = True; break
            frame = cv2.flip(frame, 1)
            res = detect(hands, frame, int((time.monotonic() - t0) * 1000) + i); i += 1
            hs = read_hands(res, frame.shape)
            draw(frame, res)
            left = ready_at - time.monotonic()
            if left > 0:
                cv2.putText(frame, str(int(left) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 165, 255), 6)
                cv2.putText(frame, f"get ready: {label}", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                if hs:
                    new_samples.append(features(hs))
                cv2.putText(frame, f"{lang}:{label}  REC {len(new_samples)}/{n}", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"sign {idx + 1}/{len(labels)}   next: {labels[idx + 1] if idx + 1 < len(labels) else '-'}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if not hs and left <= 0:
                cv2.putText(frame, "no hand detected", (10, 92),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("collect", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                break
            elif k == ord('q'):
                quit_all = True
        if new_samples:
            X.extend(new_samples); y.extend([label] * len(new_samples)); saved += len(new_samples)
            print(f"  {label}: {len(new_samples)} samples")
        if quit_all:
            break
    cap.release(); cv2.destroyAllWindows(); hands.close()
    if saved:
        save_dataset(lang, np.asarray(X, np.float32), y)
        print(f"saved {saved} samples -> {dataset_path(lang)}")
    else:
        print("nothing recorded")


def cmd_ingest(lang, folder, per_class=200, mirror=True, replace=False):
    """Turn a folder of class-named image subfolders (Kaggle-style) into landmark features.

    Each image is also fed in mirrored, so the model is handedness-agnostic: dataset
    stills are not mirrored while the live demo flips the webcam, and without this the
    hand lands in the opposite slot at run time and every prediction is wrong.
    """
    import cv2
    root = Path(folder)
    classes = sorted([d for d in root.iterdir() if d.is_dir()])
    if not classes:
        sys.exit(f"no class subfolders in {root}")
    hands = open_hands(still=True)
    X, y = ([], []) if replace else (lambda a, b: (list(a), list(b)))(*load_dataset(lang))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_imgs = total_hits = 0
    for d in classes:
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in exts])[:per_class]
        hits = 0
        for f in imgs:
            img = cv2.imread(str(f))
            if img is None:
                continue
            total_imgs += 1
            for variant in ((img, cv2.flip(img, 1)) if mirror else (img,)):
                hs = read_hands(detect(hands, variant), variant.shape)
                if hs:
                    X.append(features(hs)); y.append(d.name); hits += 1
        total_hits += hits
        print(f"  {d.name}: {len(imgs)} images -> {hits} samples")
    hands.close()
    if not X:
        sys.exit("no hands detected in any image -- wrong folder, or images are not hand photos")
    save_dataset(lang, np.asarray(X, np.float32), y)
    rate = (total_hits / (total_imgs * (2 if mirror else 1)) * 100) if total_imgs else 0
    print(f"{lang}: {len(X)} samples from {total_imgs} images ({rate:.0f}% detection) -> {dataset_path(lang)}")


def cmd_forget(lang, labels):
    """Drop labels from a dataset so they can be re-recorded cleanly (collect appends)."""
    X, y = load_dataset(lang)
    if len(X) == 0:
        sys.exit(f"no data for {lang}")
    keep = [i for i, v in enumerate(y) if v not in labels]
    dropped = len(y) - len(keep)
    if not dropped:
        sys.exit(f"none of {labels} found in {lang}; have {sorted(set(y))}")
    save_dataset(lang, X[keep], [y[i] for i in keep])
    print(f"dropped {dropped} samples ({', '.join(labels)}); {len(keep)} remain")


def cmd_train(lang, mirror=True, augment=4):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    X, y = load_dataset(lang)
    if mirror and len(X):
        X = np.vstack([X, np.array([mirror_features(v) for v in X], np.float32)])
        y = list(y) + list(y)
    if augment and len(X):
        rng = np.random.default_rng(0)
        extra = np.array([jitter_features(v, rng) for v in X for _ in range(augment)], np.float32)
        X = np.vstack([X, extra])
        y = list(y) + [lab for lab in y for _ in range(augment)]
    counts = Counter(y)
    if len(counts) < 2:
        sys.exit(f"need >=2 labels for {lang}, have {dict(counts)}")
    k = max(1, min(5, min(counts.values())))
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X, y)
    # Rejection radius: how far a live hand may sit from the training data before we
    # call it unknown. KNN always returns its nearest class, so without this an
    # out-of-distribution pose is reported confidently and wrongly. Measured from the
    # kth-neighbour distance (robust to mirrored near-duplicates), with a floor so a
    # dataset recorded very still does not produce an impossibly tight radius.
    # ponytail: heuristic constants; run --max-dist overrides, and run shows the live distance.
    ref = float(np.percentile(clf.kneighbors(X, n_neighbors=k)[0][:, -1], 99))
    max_dist = max(4 * ref, 0.6)
    MODELS.mkdir(exist_ok=True)
    with open(MODELS / f"{lang}.pkl", "wb") as f:
        pickle.dump({"clf": clf, "max_dist": max_dist}, f)
    print(f"  rejection radius: {max_dist:.2f} (typical kth-neighbour spread {ref:.3f})")
    folds = min(5, min(counts.values()))
    acc = cross_val_score(clf, X, y, cv=folds).mean() if folds >= 2 else float("nan")
    print(f"{lang}: {len(X)} samples, {len(counts)} labels {dict(counts)}, k={k}, cv acc={acc:.3f}")


def load_model(lang):
    """Returns (clf, max_dist). Accepts models pickled before the radius existed."""
    obj = pickle.load(open(MODELS / f"{lang}.pkl", "rb"))
    return (obj["clf"], obj["max_dist"]) if isinstance(obj, dict) else (obj, None)


def predict_one(clf, max_dist, vec):
    """(label, confidence, distance); label is None when the pose is too far from training."""
    dist = float(clf.kneighbors([vec], n_neighbors=1)[0][0][0])
    prob = clf.predict_proba([vec])[0]
    j = int(prob.argmax())
    if max_dist is not None and dist > max_dist:
        return None, float(prob[j]), dist
    return str(clf.classes_[j]), float(prob[j]), dist


def cmd_run(langs, cam=None, max_dist=None, thresh=0.6, hold=12):
    import cv2
    models = {}
    for lang in langs:
        p = MODELS / f"{lang}.pkl"
        if p.exists():
            clf, md = load_model(lang)
            models[lang] = (clf, md if max_dist is None else max_dist)
        else:
            print(f"skip {lang}: no model, run train first")
    if not models:
        sys.exit("no trained models")
    langs, cur = list(models), 0
    hands, cap = open_hands(), camera(cam)
    recent, text, last, t0, i = deque(maxlen=hold), "", None, time.monotonic(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = detect(hands, frame, int((time.monotonic() - t0) * 1000) + i); i += 1
        hs = read_hands(res, frame.shape)
        draw(frame, res)
        lang = langs[cur]
        clf, md = models[lang]
        pred, conf, dist, label = "-", 0.0, float("nan"), None
        if hs:
            label, conf, dist = predict_one(clf, md, features(hs))
            pred = label if label else "?"
        recent.append(label if (label and conf >= thresh) else None)
        # commit a sign only once it has been stable for the whole window
        stable = recent[0] if len(recent) == recent.maxlen and len(set(recent)) == 1 else None
        if stable and stable != last:
            text = (text + (" " if stable == "SPACE" else stable))[-40:]
            last = stable
        if stable is None:
            last = None
        ok_pred = label is not None and conf >= thresh
        cv2.putText(frame, f"[{lang}] {pred} {conf:.2f}  d={dist:.2f}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0) if ok_pred else (150, 150, 150), 2)
        cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 180, 255), 2)
        cv2.imshow("sign", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('l'):
            cur = (cur + 1) % len(langs); recent.clear(); last = None
        elif k == ord('c'):
            text = ""
    cap.release(); cv2.destroyAllWindows(); hands.close()


def cmd_livetest(lang, labels, cam=None, delay=4, hold=3):
    """Prompt for each sign in turn and report what the model actually predicts live.

    This is the honest accuracy number: trained on recordings, measured on a fresh
    performance of the same signs. Cross-validation over near-identical frames is not.
    """
    import cv2
    clf, md = load_model(lang)
    hands, cap = open_hands(), camera(cam)
    t0, i, results = time.monotonic(), 0, []
    for label in labels:
        votes, dists = collections.Counter(), []
        ready_at = time.monotonic() + delay
        stop_at = ready_at + hold
        while time.monotonic() < stop_at:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = detect(hands, frame, int((time.monotonic() - t0) * 1000) + i); i += 1
            hs = read_hands(res, frame.shape)
            draw(frame, res)
            left = ready_at - time.monotonic()
            if left > 0:
                cv2.putText(frame, str(int(left) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 165, 255), 6)
                cv2.putText(frame, f"show: {label}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                if hs:
                    pred, conf, d = predict_one(clf, md, features(hs))
                    votes[pred or "?"] += 1; dists.append(d)
                else:
                    votes["nohand"] += 1
                cv2.putText(frame, f"testing {label}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("livetest", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                stop_at = 0
        got = votes.most_common(1)[0][0] if votes else "nohand"
        share = votes[got] / max(1, sum(votes.values()))
        results.append((label, got, share, float(np.median(dists)) if dists else float("nan")))
        print(f"  show {label}: predicted {got} ({share*100:.0f}% of frames, median d={results[-1][3]:.2f})")
    cap.release(); cv2.destroyAllWindows(); hands.close()
    correct = sum(1 for l, g, _, _ in results if l == g)
    print(f"live accuracy: {correct}/{len(results)} = {correct/max(1,len(results)):.2f}")


def cmd_selftest():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(21, 3)).astype(np.float32)

    a = features([("Right", base)])
    b = features([("Right", base * 3.0 + np.array([0.4, -0.2, 0.1], np.float32))])
    assert np.allclose(a, b, atol=1e-5), "features must be translation/scale invariant"

    other = rng.normal(size=(21, 3)).astype(np.float32)
    assert np.allclose(features([("Left", base), ("Right", other)]),
                       features([("Right", other), ("Left", base)])), "slot order must be stable"

    one = features([("Right", base)])
    assert one.shape == (DIM,)
    assert not one[: 21 * 3].any() and one[21 * 3:].any(), "absent hand must be zeros"

    from sklearn.neighbors import KNeighborsClassifier
    X, y = [], []
    for cls, shift in (("A", 0.0), ("B", 5.0)):
        for _ in range(10):
            X.append(features([("Right", base + shift + rng.normal(scale=0.01, size=(21, 3)))]))
            y.append(cls)
    clf = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
    assert clf.predict([X[0]])[0] == "A" and clf.predict([X[-1]])[0] == "B", "round trip failed"

    # dataset round trip
    import tempfile
    global DATA
    keep = DATA
    try:
        DATA = Path(tempfile.mkdtemp())
        save_dataset("t", X, y)
        X2, y2 = load_dataset("t")
        assert X2.shape == (20, DIM) and y2 == y, "dataset round trip failed"
    finally:
        DATA = keep
    # blank-frame detection: black frame rejected, noisy frame accepted
    assert looks_blank(np.zeros((480, 640, 3), np.uint8)), "black frame must read as blank"
    assert looks_blank(np.full((480, 640, 3), 12, np.uint8)), "flat dark frame must read as blank"
    assert not looks_blank(rng.integers(0, 255, (480, 640, 3)).astype(np.uint8)), "real image must not read as blank"

    # aspect correction: a 4:3 frame must stretch x by 4/3 relative to a square one
    class _P:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    class _Res:
        hand_landmarks = [[_P(0.5, 0.5, 0.1)] * 21]
        handedness = [[type("C", (), {"category_name": "Right"})()]]
    sq = read_hands(_Res(), (480, 480, 3))[0][1]
    wide = read_hands(_Res(), (480, 640, 3))[0][1]
    assert np.isclose(wide[0][0] / sq[0][0], 640 / 480), "x must be scaled by w/h"
    assert np.isclose(wide[0][1], sq[0][1]), "y must not be rescaled"

    # rejection: a pose far from all training data must come back as unknown
    lab, _, d_near = predict_one(clf, 1.0, X[0])
    assert lab == "A", "an in-distribution sample must classify"
    far = np.full(DIM, 50.0, np.float32)
    lab_far, _, d_far = predict_one(clf, 1.0, far)
    assert lab_far is None, "an out-of-distribution pose must be rejected"
    assert d_far > d_near, "distance must grow with distance from training data"

    # mirroring: involutive, and moves a one-handed sign to the other slot
    v1 = features([("Right", base)])
    mv = mirror_features(v1)
    assert np.allclose(mirror_features(mv), v1, atol=1e-6), "mirror must be its own inverse"
    assert mv[: 21 * 3].any() and not mv[21 * 3:].any(), "mirror must swap hand slots"
    two = features([("Left", base), ("Right", other)])
    assert np.allclose(mirror_features(mirror_features(two)), two, atol=1e-6)

    # jitter: widens a class without inventing a second hand
    rng2 = np.random.default_rng(3)
    one = features([("Right", base)])
    j = jitter_features(one, rng2)
    assert not j[: 21 * 3].any(), "jitter must not fill an empty hand slot"
    assert j[21 * 3:].any() and not np.allclose(j, one), "jitter must change the occupied hand"
    assert 0.0 < float(np.linalg.norm(j - one)) < 1.5, "jitter must stay a small perturbation"

    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("lang"); c.add_argument("label", nargs="+")
    c.add_argument("-n", type=int, default=60); c.add_argument("--cam", type=int)
    c.add_argument("--delay", type=int, default=5, help="seconds to get into each pose")
    g = sub.add_parser("ingest"); g.add_argument("lang"); g.add_argument("folder")
    g.add_argument("--per-class", type=int, default=200); g.add_argument("--no-mirror", action="store_true")
    g.add_argument("--replace", action="store_true")
    fg = sub.add_parser("forget"); fg.add_argument("lang"); fg.add_argument("labels", nargs="+")
    tr = sub.add_parser("train"); tr.add_argument("lang")
    tr.add_argument("--no-mirror", action="store_true", help="do not add other-hand copies")
    tr.add_argument("--augment", type=int, default=4, help="rotated copies per sample (0 = off)")
    r = sub.add_parser("run"); r.add_argument("langs", nargs="+"); r.add_argument("--cam", type=int)
    r.add_argument("--max-dist", type=float, help="reject poses farther than this (default: from training)")
    pr = sub.add_parser("probe"); pr.add_argument("-n", type=int, default=30); pr.add_argument("--cam", type=int)
    lt = sub.add_parser("livetest"); lt.add_argument("lang"); lt.add_argument("labels", nargs="+")
    lt.add_argument("--cam", type=int); lt.add_argument("--delay", type=int, default=4)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "collect":
        cmd_collect(a.lang, a.label, a.n, a.cam, a.delay)
    elif a.cmd == "ingest":
        cmd_ingest(a.lang, a.folder, a.per_class, not a.no_mirror, a.replace)
    elif a.cmd == "forget":
        cmd_forget(a.lang, a.labels)
    elif a.cmd == "train":
        cmd_train(a.lang, not a.no_mirror, a.augment)
    elif a.cmd == "run":
        cmd_run(a.langs, a.cam, a.max_dist)
    elif a.cmd == "livetest":
        cmd_livetest(a.lang, a.labels, a.cam, a.delay)
    elif a.cmd == "probe":
        cmd_probe(a.n, a.cam)
    else:
        cmd_selftest()


if __name__ == "__main__":
    main()
