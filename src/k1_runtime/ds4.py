"""DS4 on the laptop for push_ik_web.py: sticks, buttons with press and hold, light bar. Wired or Bluetooth.

Light bar is the hid-playstation multicolor LED (<led>/multi_intensity, <led>/brightness). It and evdev need permissions:
see docs/k1_phasespace_field_runbook.md, section 2. Without it the pad still works and the page says the light is off.
"""
from __future__ import annotations

import glob
import threading
import time

HOLD_S = 1.0
#: light bar colours, also shown on the page
COLOURS = {"not ready": (40, 40, 40), "ready": (0, 0, 255), "settle": (255, 200, 0), "policy": (0, 255, 0), "mocap hold": (255, 60, 0),
           "robot a": (0, 255, 255), "robot b": (255, 0, 255), "robot both": (255, 255, 255),
           "arms a": (0, 255, 255), "arms b": (255, 0, 255), "arms both": (255, 255, 255),
           "stopped": (255, 0, 0), "refused": (255, 0, 0)}
BLINKS = ("arms a", "arms b", "arms both", "refused")


class Pad:
  """on_button(name, held): held=False on press, then held=True once if it stays down HOLD_S."""

  def __init__(self, on_button):
    self.on_button = on_button
    self.zero()
    self.connected = False
    self.led, self.light_ok, self.shown = None, False, None
    threading.Thread(target=self._holds, daemon=True).start()

  def zero(self):
    self.lx, self.ly, self.rx, self.ry = 0.0, 0.0, 0.0, 0.0
    self.l1, self.r1, self.l2, self.r2 = False, False, False, False
    self.down = {}

  def button(self, name, is_down):
    if is_down:
      self.down[name] = time.monotonic()
      self.on_button(name, False)
    else:
      self.down.pop(name, None)

  def _holds(self):
    while True:
      for name, t in list(self.down.items()):
        if t is not None and time.monotonic() - t >= HOLD_S:
          self.down[name] = None  # fires once per press
          self.on_button(name, True)
      time.sleep(0.02)

  def start_evdev(self):
    threading.Thread(target=self._read_forever, daemon=True).start()

  def _read_forever(self):
    import evdev
    e = evdev.ecodes
    names = {e.BTN_EAST: "circle", e.BTN_NORTH: "triangle", e.BTN_SELECT: "share", e.BTN_START: "options"}
    while True:
      dev = next((d for d in map(evdev.InputDevice, evdev.list_devices())
                  if e.EV_ABS in d.capabilities() and e.BTN_SOUTH in d.capabilities().get(e.EV_KEY, [])), None)
      if dev is None:
        time.sleep(0.5)
        continue
      self._find_led(dev.path.rsplit("/", 1)[-1])
      self.connected = True
      print("pad:", dev.name, "| light bar", "ok" if self.light_ok else "not writable", flush=True)
      try:
        for ev in dev.read_loop():
          if ev.type == e.EV_ABS and ev.code in (e.ABS_X, e.ABS_Y, e.ABS_RX, e.ABS_RY):
            info = dev.absinfo(ev.code)
            v = 2.0 * (ev.value - info.min) / (info.max - info.min) - 1.0
            setattr(self, {e.ABS_X: "lx", e.ABS_Y: "ly", e.ABS_RX: "rx", e.ABS_RY: "ry"}[ev.code], v)
          elif ev.type == e.EV_KEY and ev.value != 2:
            if ev.code == e.BTN_TL2:
              self.l2 = ev.value == 1
            elif ev.code == e.BTN_TR2:
              self.r2 = ev.value == 1
            elif ev.code == e.BTN_TL:
              self.l1 = ev.value == 1
              self.button("l1", self.l1)
            elif ev.code == e.BTN_TR:
              self.r1 = ev.value == 1
              self.button("r1", self.r1)
            elif ev.code in names:
              self.button(names[ev.code], ev.value == 1)
      except OSError:  # unplugged, Bluetooth dropped
        pass
      self.connected = False
      self.zero()
      self.on_button("lost", False)

  def _find_led(self, event):
    found = glob.glob(f"/sys/class/input/{event}/device/device/leds/*:rgb:indicator")
    split = glob.glob(f"/sys/class/input/{event}/device/device/leds/*:red")  # kernels with separate red/green/blue LEDs
    self.use_led(found[0] if found else split[0][:-len(":red")] if split else None)

  @staticmethod
  def _led_files(path):
    if glob.glob(f"{path}:red"):
      return [f"{path}:{c}/brightness" for c in ("red", "green", "blue")]
    return [f"{path}/multi_intensity", f"{path}/brightness"]

  def use_led(self, path):
    self.led, self.light_ok, self.shown = path, False, None
    if path:
      try:
        for f in self._led_files(path):
          open(f, "a").close()
        self.light_ok = True
      except OSError:
        pass

  def show(self, name):
    rgb = (0, 0, 0) if name in BLINKS and int(time.monotonic() * 4) % 2 else COLOURS[name]
    if not self.light_ok or rgb == self.shown:
      return
    try:
      files = self._led_files(self.led)
      if len(files) == 3:
        for path, value in zip(files, rgb):
          with open(path, "w") as f:
            f.write(str(value))
      else:
        with open(files[0], "w") as f:
          f.write(" ".join(map(str, rgb)))
        with open(files[1], "w") as f:
          f.write("255")
      self.shown = rgb
    except OSError:
      self.light_ok = False
