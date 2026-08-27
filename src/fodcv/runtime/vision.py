"""Camera -> Hailo-8 -> tracked, triaged targets. The seam the robot talks to.

    from fodcv.runtime.vision import Vision

    with Vision(hef=".../best.hef") as vision:
        while patrolling:
            if vision.age > 0.5:
                halt("vision stalled")
            for t in vision.latest():
                if t.state == "CAUTION":
                    slow_down()
                if t.state == "CONFIRM" and t.action == "PICK":
                    approach(t.centroid, vision.frame_size)
            drive_step()

Capture runs on its own daemon thread, so the robot's loop is never blocked by a
20 ms frame grab and a multi-second retrieval routine does not leave a backlog of
stale frames behind it. `latest()` is the newest frame's targets or the previous
ones; `age` says how stale that is, and the robot -- not this module -- decides
what an unacceptable age is.

The tracker runs *in* the thread. Feeding it at poll rate instead would make
EMA_ALPHA and MAX_MISSES mean something different depending on how busy the robot
happened to be.

Boxes are pixels in the rotated frame. No ground-plane projection here: that needs
a mount height and tilt this package cannot know.

Import-time dependencies are cv2 + numpy + stdlib only. picamera2, libcamera and
hailo_platform are imported inside the methods that need them, because none of the
three installs on a Mac and every pure helper below is unit-tested there. On the Pi
this must run on the *system* python3.11 -- apt's picamera2 is built against it and
a 3.12 venv is a different C ABI.
"""

import json
import math
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from fodcv.runtime import policy

PAD = 114  # ultralytics' letterbox fill; the model was calibrated behind it
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# FOD-A's median object spans 11.1% of its 300x300 training frame. Every "is this
# thing big enough for the model to see" question in this file reduces to it.
TRAIN_SCALE = 0.111

STAGES = ("capture", "preprocess", "infer", "postprocess", "total")
WARMUP = 5  # frames excluded from the timing stats

#: What the robot loop consumes. `state` is confidence over time, `action` is what
#: the class means -- see fodcv.runtime.policy for why they are two fields.
Target = namedtuple("Target", "id state action cls conf box centroid misses")


# --- pure helpers, all unit-tested off the Pi ------------------------------------

def class_names(hef_path):
    """Class list for a .hef, from the run.json in its artifact dir. Hardcoded,
    it mislabels every box the moment a run changes taxonomy, silently."""
    run_json = Path(hef_path).resolve().parent.parent / "run.json"
    assert run_json.exists(), (
        f"no run.json at {run_json} -- a .hef must sit in artifacts/<run>/<export>/, "
        "beside the manifest that says what its class ids mean")
    names = json.loads(run_json.read_text())["classes"]
    names = [names[str(i)] for i in range(len(names))]
    # run.json is written from fodcv-migrate's --dataset, the .hef's class count
    # from the weights. Wrong --dataset and they disagree with no symptom: the
    # ids past the end of `names` are never in `_shown`, so every box of a class
    # the taxonomy grew is dropped before it is ever drawn.
    nms_config = Path(hef_path).with_name("nms_config.json")
    if nms_config.exists():
        decoded = json.loads(nms_config.read_text())["classes"]
        assert decoded == len(names), (
            f"{run_json} names {len(names)} classes {names} but the .hef decodes "
            f"{decoded} -- re-run fodcv-migrate --dataset for the taxonomy this "
            "model was trained on")
    return names


def hef_imgsz(hef_path, fallback=480):
    """The .hef's input size, off the nms_config.json the hailo export writes
    beside it.

    A property of the build, not something a caller should carry: fodcv.matrix
    defaults exports to 640 and hailo-compile.sh passes 480, so the two disagree
    by default and the mismatch has no symptom worth the name -- the letterbox
    just hands the chip the wrong square.
    """
    config = Path(hef_path).with_name("nms_config.json")
    if not config.exists():
        return fallback
    dims = json.loads(config.read_text())["image_dims"]
    assert dims[0] == dims[1], f"{config} declares a non-square input {dims}; letterbox assumes square"
    return int(dims[0])


def focus_arg(value):
    """--focus: 'auto', or a subject distance in metres.

    Raises ValueError, which argparse turns into a usage error by itself -- so this
    stays usable as an argparse `type=` without importing argparse here.
    """
    if value == "auto":
        return None
    distance = float(value)
    if distance <= 0:
        raise ValueError("focus must be positive metres, or 'auto'")
    return distance


def scale_constant(hfov_deg):
    """k, where `length = k * distance * zoom` is the object length that reaches
    the model at FOD-A's training scale.

    Frame width at distance d is 2*d*tan(hfov/2), and the object must cover
    TRAIN_SCALE of it. k(66 deg) = 0.144, i.e. roughly `distance = 7 * length`.
    """
    return TRAIN_SCALE * 2 * math.tan(math.radians(hfov_deg) / 2)


def fmt_m(distance):
    return "inf" if math.isinf(distance) else f"{distance:.2f} m"


def real_px(sensor_width, zoom):
    """Sensor pixels spanning the crop -- the ceiling on genuine detail."""
    return sensor_width * zoom


def need_px(frame_width, zoom):
    """Frame pixels an object must span to reach the model at training scale.

    Cropping magnifies, so zoom divides: at zoom 0.5 an object covers twice the
    frame pixels it did at full field of view.
    """
    return round(TRAIN_SCALE * frame_width * zoom)


def letterbox(frame, size):
    """Resize preserving aspect, pad to square. Returns the image and the inverse."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (scale, top, left)


def to_frame_coords(box, inverse, size, shape):
    """Normalised letterbox [y0,x0,y1,x1] -> pixel [x0,y0,x1,y1] in the original frame."""
    scale, top, left = inverse
    y0, x0, y1, x1 = (v * size for v in box[:4])
    x0, x1 = (x0 - left) / scale, (x1 - left) / scale
    y0, y1 = (y0 - top) / scale, (y1 - top) / scale
    h, w = shape[:2]
    return (int(np.clip(x0, 0, w)), int(np.clip(y0, 0, h)),
            int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h)))


def decode(per_class, shown, conf, inverse, size, shape):
    """The .hef's per-class output -> [(box_px, class_id, score)].

    `per_class` is `list(result.values())[0][0]`: one array per class id, each row
    a normalised [y0, x0, y1, x1, score]. NMS already ran on the chip, so there is
    nothing to suppress here -- only to filter and un-letterbox.
    """
    boxes = []
    for class_id, rows in enumerate(per_class):
        if class_id not in shown:
            continue
        for row in np.asarray(rows):
            if row[4] >= conf:
                boxes.append((to_frame_coords(row, inverse, size, shape),
                              class_id, float(row[4])))
    return boxes


def to_targets(tracks):
    """Live tracks -> what the robot reads. Missed tracks are dropped: a track
    coasting on MAX_MISSES has no box this frame, and a stale box is worse than
    none when the thing being decided is where to put a magnet."""
    return [
        Target(id=t.id, state=t.state(), action=t.action(), cls=t.cls,
               conf=t.ema_conf, box=getattr(t, "box", None),
               centroid=t.centroid, misses=t.misses)
        for t in tracks if t.misses == 0
    ]


# --- the seam --------------------------------------------------------------------

class Vision:
    """Open the camera and the accelerator, run detection on a thread, publish targets.

    Every knob is a constructor argument with the same default the CLI used, so a
    robot that passes nothing gets the configuration RESULT.md §6 measured.
    """

    def __init__(self, hef, imgsz=None, conf=0.25, classes=None,
                 width=1280, height=720, zoom=1.0, rotate=0, sensor_width=2304,
                 focus=None, shutter=0, settle=1.0):
        self.hef_path = str(hef)
        # None asks the .hef. An explicit size is still honoured -- a .hef with no
        # nms_config.json beside it has nothing to ask -- but never silently.
        declared = hef_imgsz(self.hef_path)
        assert imgsz is None or imgsz == declared, (
            f"--imgsz {imgsz} but {hef} declares a {declared}x{declared} input -- "
            "the letterbox would feed the chip the wrong square")
        self.imgsz = declared if imgsz is None else imgsz
        self.conf = conf
        self.width, self.height = width, height
        self.zoom = zoom
        self.rotate = rotate
        self.sensor_width = sensor_width
        self.focus = focus
        self.shutter = shutter
        self.settle = settle

        self.classes = class_names(self.hef_path)
        wanted = list(self.classes) if classes is None else list(classes)
        unknown = sorted(set(wanted) - set(self.classes))
        assert not unknown, f"classes {unknown} not in this model's classes: {self.classes}"
        # A reporting filter, not a model change: NMS is compiled into the .hef, so
        # the chip scores every class and costs the same either way.
        self._shown = {self.classes.index(name) for name in wanted}

        self.stats = {name: [] for name in STAGES}
        #: Highest score the chip emitted per class on the last frame, before
        #: `conf` and before `--classes`. "Found nothing" and "the filter hid it"
        #: are the same `0 det` on a HUD; this is what tells them apart.
        self.top_scores = [0.0] * len(self.classes)
        self.frame_id = 0
        self._frame = None
        self._targets = []
        self._stamp = 0.0
        self._error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._picam = None
        self._full_crop = None
        self.input_name = None  # the .hef's input vstream, named once the device is up

    # -- lifecycle ----------------------------------------------------------------

    def __enter__(self):
        self._open_camera()
        self._thread = threading.Thread(target=self._loop, name="fodcv-vision", daemon=True)
        self._thread.start()
        # Wait for the Hailo device to come up rather than returning a Vision whose
        # first latest() would raise. 30 s is device configure, not inference.
        if not self._ready.wait(30) and self._error is None:
            self.__exit__(None, None, None)
            raise TimeoutError("Hailo device did not become ready within 30 s")
        self._raise_if_failed()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._picam is not None:
            self._picam.stop()
            self._picam = None
        return False

    # -- what the robot reads -----------------------------------------------------

    def latest(self):
        """Triaged targets from the most recent frame. Re-raises a thread failure.

        A dead camera must not read as "no debris, keep patrolling", so the error
        surfaces here rather than in a log nobody is watching.
        """
        self._raise_if_failed()
        with self._lock:
            return list(self._targets)

    def snapshot(self):
        """(frame, targets) for a preview or a saved JPEG. The frame is not copied --
        draw on a copy of it."""
        self._raise_if_failed()
        with self._lock:
            return self._frame, list(self._targets)

    @property
    def age(self):
        """Seconds since the last completed frame. inf before the first one."""
        return math.inf if not self._stamp else time.monotonic() - self._stamp

    @property
    def fps(self):
        recent = self.stats["total"][-30:]
        return 1000 / (sum(recent) / len(recent)) if recent else 0.0

    @property
    def frame_size(self):
        """(width, height) of the frame boxes are measured in, after rotation."""
        return (self.height, self.width) if self.rotate in (90, 270) else (self.width, self.height)

    def _raise_if_failed(self):
        if self._error is not None:
            raise self._error

    # -- live tuning, all safe from the consumer's thread --------------------------

    def set_conf(self, conf):
        self.conf = max(0.0, min(1.0, conf))

    def set_rotate(self, degrees):
        assert degrees in (0, 90, 180, 270)
        self.rotate = degrees

    def set_classes(self, names):
        self._shown = {self.classes.index(n) for n in names}

    def toggle_class(self, name):
        if name in self.classes:
            self._shown ^= {self.classes.index(name)}

    def shown_classes(self):
        return [n for i, n in enumerate(self.classes) if i in self._shown]

    def set_zoom(self, factor):
        """Sensor crop factor. 1.0 is the full field of view; smaller crops in,
        which is how a small object buys back the pixels the letterbox takes away."""
        self.zoom = max(0.15, min(1.0, factor))
        x, y, w, h = self._full_crop
        cw, ch = int(w * self.zoom), int(h * self.zoom)
        self._picam.set_controls({"ScalerCrop": (x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)})

    def refocus(self):
        """One-shot autofocus sweep, then hold. Continuous AF hunts on a flat floor
        with one small object, and a hunting lens looks exactly like the defocus bug."""
        from libcamera import controls

        self._picam.set_controls({"AfMode": controls.AfModeEnum.Auto})
        self._picam.autofocus_cycle()

    def focus_state(self):
        """Measured subject distance in metres, and the focus figure of merit.

        LensPosition is dioptres, so its reciprocal is where the lens is actually
        focused -- an unknown working distance measured rather than guessed. 0.0
        means infinity. Costs one frame wait.
        """
        metadata = self._picam.capture_metadata()
        lens = metadata.get("LensPosition") or 0.0
        return (1.0 / lens if lens else math.inf), metadata.get("FocusFoM", 0)

    # -- internals ----------------------------------------------------------------

    def _open_camera(self):
        from libcamera import controls
        from picamera2 import Picamera2

        self._picam = Picamera2()
        # picamera2's "RGB888" is B,G,R in memory -- OpenCV order, what we want for
        # drawing and imwrite. The model wants true RGB, so convert per frame.
        # Pin the raw mode: left alone, picamera2 answers a 1280x720 request with a
        # binned readout, and a zoom 0.4 crop of that is upscaled interpolation the
        # model sees too.
        sensor = (self.sensor_width, round(self.sensor_width * 2592 / 4608))
        self._picam.configure(self._picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}, raw={"size": sensor}))
        self._picam.start()

        # Must be set explicitly: libcamera defaults to Manual at 1.0 m, which leaves
        # a fastener at arm's length defocused of exactly the fine detail a 50 px
        # object is made of. AfRange is Full, not Macro, because the working distance
        # is the thing being measured here.
        if self.focus is None:
            self._picam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                                      "AfRange": controls.AfRangeEnum.Full})
        else:
            self._picam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                      "LensPosition": 1.0 / self.focus})
        # Manual *exposure time* only: gain stays on auto, so capping the shutter
        # against motion blur costs brightness the AGC then puts back.
        if self.shutter:
            self._picam.set_controls({"ExposureTimeMode": controls.ExposureTimeModeEnum.Manual,
                                      "ExposureTime": self.shutter})

        self._full_crop = self._picam.camera_controls["ScalerCrop"][1]  # [1] is the whole sensor
        if self.zoom != 1.0:
            self.set_zoom(self.zoom)
        time.sleep(self.settle)  # AGC/AWB settle; without it the first frames are dark

    def _loop(self):
        """Capture -> infer -> track -> publish, until stopped.

        The Hailo contexts are created and torn down here rather than in __enter__ so
        the device is owned start to finish by the one thread that uses it.
        """
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )

            hef = HEF(self.hef_path)
            self.input_name = hef.get_input_vstream_infos()[0].name
            with VDevice() as device:
                params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
                network_group = device.configure(hef, params)[0]
                with network_group.activate(network_group.create_params()):
                    ivs = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
                    ovs = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
                    with InferVStreams(network_group, ivs, ovs) as pipeline:
                        self._ready.set()
                        tracks = []
                        while not self._stop.is_set():
                            tracks = self._step(pipeline, tracks)
        except BaseException as exc:  # surfaced from latest(), never swallowed
            self._error = exc
            self._ready.set()

    def _step(self, pipeline, tracks):
        t0 = time.perf_counter()
        bgr = self._picam.capture_array()
        # Rotate ahead of the letterbox, so preview and model agree. Rotating only
        # the preview would show upright boxes drawn from coordinates the model
        # derived from a sideways frame.
        rotate = self.rotate
        if rotate:
            bgr = cv2.rotate(bgr, ROTATIONS[rotate])
        t1 = time.perf_counter()

        square, inverse = letterbox(bgr, self.imgsz)
        batch = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)[None]
        t2 = time.perf_counter()

        result = pipeline.infer({self.input_name: np.ascontiguousarray(batch)})
        t3 = time.perf_counter()

        per_class = list(result.values())[0][0]
        self.top_scores = [float(np.asarray(rows)[:, 4].max()) if len(rows) else 0.0
                           for rows in per_class]
        boxes = decode(per_class, self._shown, self.conf, inverse, self.imgsz, bgr.shape)
        detections = [(((x0 + x1) / 2, (y0 + y1) / 2), score, self.classes[class_id])
                      for (x0, y0, x1, y1), class_id, score in boxes]
        tracks = policy.match_tracks(tracks, detections)
        # The tracker deals in centroids; the robot wants the box too, so carry it
        # across on the track. Matching is by centroid, so index by that.
        # Both sides destructured: `Target.box` is four pixels, and storing a whole
        # decode row here instead is a crash four frames away in the caller's
        # drawing code, with nothing pointing back at this line.
        by_centroid = {centroid: box
                       for (centroid, _, _), (box, _, _) in zip(detections, boxes)}
        for track in tracks:
            track.box = by_centroid.get(track.centroid, getattr(track, "box", None))
        t4 = time.perf_counter()

        with self._lock:
            self._frame = bgr
            self._targets = to_targets(tracks)
            self._stamp = time.monotonic()
        self.frame_id += 1

        if self.frame_id > WARMUP:  # warmup frames are not representative
            for name, ms in zip(STAGES, ((t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (t4 - t0))):
                self.stats[name].append(ms * 1000)
        return tracks
