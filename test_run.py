"""Drive cmd_run's real loop with a fake camera and a stubbed GUI."""
import sys, types, numpy as np, cv2 as realcv2
sys.path.insert(0, "/home/vanshc/signlang")

frames = []
img = realcv2.imread("/home/vanshc/signlang/testdata/hand.jpg")
frames.append(realcv2.resize(img, (640, 480)))                 # two hands
frames.append(np.zeros((480, 640, 3), np.uint8))               # no hands

class FakeCap:
    def __init__(self): self.i = 0
    def read(self):
        if self.i >= 40: return False, None
        f = frames[self.i % len(frames)]; self.i += 1
        return True, f.copy()
    def release(self): pass

# stub only the GUI bits; everything else is the real cv2
stub = types.ModuleType("cv2")
stub.__dict__.update(realcv2.__dict__)
stub.imshow = lambda *a, **k: None
stub.waitKey = lambda *a, **k: 255
stub.destroyAllWindows = lambda: None
sys.modules["cv2"] = stub

import sign
sign.camera = lambda *a, **k: FakeCap()
sign.cmd_run(["isl"])
print("run loop smoke ok")
