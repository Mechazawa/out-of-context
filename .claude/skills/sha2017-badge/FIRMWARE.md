# Firmware and the MicroPython API

## The two firmwares

**badge.team ESP32-platform-firmware** (`badgeteam/ESP32-platform-firmware`) is
the current one and the target for new work. Modules: `display`, `eink`,
`buttons`, `neopixel`, `system`, `machine`, `virtualtimers`, `easydraw`, `term`,
`wifi`, `woezel`, `voltages`, `orientation`, `rtc`.

**The original 2017 firmware** (`SHA2017-badge/Firmware`) used `ugfx` and `badge`
instead: `badge.init()`, `ugfx.init()`, `ugfx.string(x, y, text, font, colour)`,
`ugfx.flush()`, `ugfx.set_lut(...)`. Its repos stopped in 2018. Everything below
describes the badge.team firmware.

Tell them apart by trying `import display` over serial.

## Updating

The last build is 24021401, named "End of an era" (February 2024). The OTA
server is still serving it:

```
https://ota.badge.team/version/sha2017.txt   -> {"build":24021401, "name":"End of an era"}
https://ota.badge.team/sha2017.bin
```

So the easy update path works: join WiFi from the launcher, then "Update
firmware". The Hatchery app repository at `hatchery.badge.team` returns HTTP 500,
so installing apps over the air is not an option and files have to go over
serial.

Building from source needs the vendored toolchain in `toolchain/`, a python2 in
`$PATH` for the ESP-IDF v3 era build scripts, and
`cp firmware/configs/sha2017_defconfig firmware/sdkconfig` before `./build.sh`
and `./flash.sh`. Only worth it for firmware changes; use OTA otherwise.

## Drawing

`display` is the framebuffer module (registered from `modframebuffer.c`). It
needs no init; the firmware brings the panel up at boot.

```python
import display

display.drawFill(0xFFFFFF)
display.drawText(x, y, "text", 0x000000, "roboto_regular18")
display.drawLine(x0, y0, x1, y1, color)
display.drawRect(x, y, w, h, filled, color)
display.drawPng(x, y, "/path.png")
w = display.getTextWidth("text", "roboto_regular18")
h = display.getTextHeight("text", "roboto_regular18")
display.flush(display.FLAG_LUT_FASTEST)
```

`drawText` takes `([window,] x, y, text, color, font, xScale, yScale)` with
everything after `text` optional. `y` is the top of the glyph box.

Colours are `0xRRGGBB` and get downsampled by the driver. The framebuffer holds
8 bits per pixel, so intermediate greys exist in RAM but only reach the glass
through the greyscale flush.

Coordinates are 296 wide by 128 high in the default landscape orientation.
`display.orientation(90)` and friends rotate.

### Fonts

`org18`, `org01_8`, `fairlight8`, `fairlight12`, `dejavusans20`,
`permanentmarker22`, `permanentmarker36`, `roboto_black22`,
`roboto_blackitalic24`, `roboto_regular12`, `roboto_regular18`,
`roboto_regular22`, `weather42`, `pixelade13`, `7x5`, `ocra16`, `ocra22`, and
the `exo2_*` family. `display.listFonts()` returns the live list.

`7x5` is the small fixed bitmap font the firmware uses for terminal-style
output; `xScale`/`yScale` scale it cleanly.

## Flush cost

`display.flush(flags)` does nothing if nothing is dirty, unless `FLAG_FORCE`.

| flag | value | LUT frames |
|---|---|---|
| `FLAG_FORCE` | 1 | refresh even when clean |
| `FLAG_FULL` | 2 | force a full-panel refresh |
| `FLAG_LUT_GREYSCALE` | 4 | greyscale hack |
| `FLAG_LUT_NORMAL` | 8 | 19 |
| `FLAG_LUT_FAST` | 16 | 10 |
| `FLAG_LUT_FASTEST` | 32 | 7 |
| (no flag) | 0 | 78, the inverting full refresh |

The panel is clocked at 62µs per line with 26 dummy lines per gate
(`driver_eink.c`), so one LUT frame over a band of H rows costs
`(H + 26) * 62µs`. Computed from those constants:

| LUT | full panel (296) | 32-row band |
|---|---|---|
| full | 1.56 s | 0.19 s |
| normal | 0.38 s | 0.07 s |
| fast | 0.20 s | 0.04 s |
| fastest | 0.14 s | 0.03 s |

The driver's own comments say the fast LUT needs two passes and the fastest
needs four to fully drive a pixel, so a single fast flush leaves weaker contrast
and visible ghosting. Repeated flushes over the same area darken it. The full
LUT inverts the panel on the way (the black flash) and is the only one that
fully clears ghosting.

### Partial refresh

Partial refresh works on **x ranges of the landscape framebuffer**, because the
flush hands the dirty rectangle's `x0`/`x1` to the panel as its `y_start`/`y_end`
(the panel is mounted rotated 90 degrees). A refreshed band is therefore a
vertical strip spanning all 128 rows. Refreshing a single line of text is not
possible; refreshing the horizontal span a word occupies is.

By default this buys nothing, because `minimal_update_height` starts at the full
296 and every dirty rectangle is widened to it. Opt in:

```python
import eink
eink.setMinimalUpdateHeight(32)   # bands smaller than 32 get padded to 32
```

The driver author's comment explains the default: too small a band and the
pixels do not swing far enough to be legible. 32 is a starting point to test on
glass, not a known-good value.

`eink` also has `deep_sleep()`, `wakeup()`, `busy()`, `busy_wait()` and
`write(buffer, flags)` for pushing raw bitmaps.

### Greyscale

`display.flush(display.FLAG_LUT_GREYSCALE)` renders the 8bpp buffer as greys by
flashing the panel black and then layering partial updates. The DEPG0290B1 is
capped at 5 layers in the driver because more causes ghosting. It is a full
panel operation and slow. Not usable for a streaming display.

## Apps

An app is a folder with `__init__.py`, optionally `metadata.json`:

```json
{"name": "Stream", "category": "system", "hidden": false}
```

The launcher scans every folder on `sys.path`, which includes `/apps` on flash
and the SD card if one is mounted.

```python
import system
system.start("myapp")     # writes the name to RTC memory and reboots into it
system.launcher()
system.shell()            # boot to a bare REPL
```

Switching apps always reboots: `system.start` calls `machine.deepsleep(1)`. RAM
does not survive, so state has to go through NVS, the filesystem, or RTC memory.

To make an app the one that runs at power-on:

```python
import machine
machine.nvs_setstr("system", "default_app", "myapp")
```

`boot.py` reads that when RTC memory is empty. Holding START during boot loads
`dashboard.recovery` instead, which is the way back out of a broken autostart
app.

Getting files onto the badge: the SHA2017 build has fsoverbus disabled, so use
the MicroPython raw REPL over serial with `ampy` or `mpfshell` (both named in
the badge docs). `mpremote` expects a newer MicroPython than this LoBo-derived
fork, so verify before relying on it.

## Power management and watchdogs

Sleep is opt-in per app. `tasks.powermanagement` only sleeps the badge if the
app imports and enables it, which the launcher and home screen do and a custom
app need not. The idle timeout lives in NVS at `system/sleep` (default 20000ms),
and `system/usb_stay_awake` keeps it awake while USB power is present.

The task watchdog runs at 5 seconds with panic disabled, so a long blocking loop
logs a warning rather than resetting the badge. The interrupt watchdog is 300ms
and does reset.
