---
name: sha2017-badge
description: SHA2017 badge (ESP32, 296x128 e-ink, MicroPython) hardware facts, the badge.team firmware API, and how to drive the display over the USB serial link. Use when working with the SHA2017 badge or any badge.team ESP32 badge, writing or installing MicroPython apps for it, driving its e-ink display, or streaming text/data to it over serial from a host.
---

# SHA2017 badge

ESP32-WROOM32 (16MB flash), 2.9" e-ink at 296x128 1-bit, eight MPR121 capacitive
buttons, six SK6812 LEDs, SD slot, 1000mAh LiPo, CP2102 USB serial at 115200
baud. MicroPython. No display backlight and no way to make e-ink fast.

Everything here was read from the firmware and PCB sources (see
[SOURCES.md](SOURCES.md)); none of it is measured on hardware. Timings marked
"computed" are arithmetic from driver constants, so treat them as an upper bound
on what the panel can do, not as a benchmark.

## Which firmware is on the badge

Two exist and their APIs do not overlap. Check first, over serial:

```python
import display          # badge.team ESP32-platform-firmware (current)
import ugfx, badge      # original 2017 SHA2017 firmware
```

`display` is the one to target. The original firmware's `ugfx`/`badge` modules
are superseded; if the badge has been in a drawer since 2017 it probably still
runs them, and updating means either OTA over WiFi from the launcher or a
serial reflash. See [FIRMWARE.md](FIRMWARE.md).

## Quick start

```bash
screen /dev/tty.usbserial-XXXX 115200      # macOS; /dev/ttyUSB0 on Linux
```

The badge shows a menu on serial, not a bare prompt. Pick "Python shell" to get
a REPL. Then:

```python
import display
display.drawFill(0xFFFFFF)
display.drawText(4, 4, "hello", 0x000000, "roboto_regular18")
display.flush(display.FLAG_LUT_FASTEST)
```

## Three things that bite

1. **Opening the serial port reboots the badge.** The CP2102's DTR and RTS drive
   the ESP32's reset and GPIO0 through two transistors (schematic sheet 1). Any
   host tool that asserts them on open restarts whatever is running. Clear both
   before opening the port.
2. **Every flush repaints the whole panel by default.** Partial refresh is
   implemented but `minimal_update_height` starts at the full 296, so a
   one-word update costs a full-screen refresh until you call
   `eink.setMinimalUpdateHeight(n)`.
3. **Switching apps reboots.** `system.start(name)` writes the app name to RTC
   memory and deep-sleeps for 1ms. There is no in-place app switch, and RAM does
   not survive.

## Reference

- [HARDWARE.md](HARDWARE.md): pinout, display controller, power, expansion header.
- [FIRMWARE.md](FIRMWARE.md): the `display`/`eink` API, fonts, refresh cost, app
  layout, installing and autostarting an app.
- [SERIAL.md](SERIAL.md): the serial link, reading a stream inside a running app,
  buffer sizes, Ctrl-C, and what breaks a long-running feed.
- [SOURCES.md](SOURCES.md): repos and files each claim came from.
