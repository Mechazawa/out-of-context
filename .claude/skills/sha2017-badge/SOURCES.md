# Where these claims come from

Read from source on 2026-07-30. Nothing in this skill was verified on hardware.

## Repositories

| repo | commit | used for |
|---|---|---|
| [badgeteam/ESP32-platform-firmware](https://github.com/badgeteam/ESP32-platform-firmware) | `010d13b` (2024-02-14) | everything about the current firmware |
| [SHA2017-badge/PCB](https://github.com/SHA2017-badge/PCB) | `d4ee9e2` (2017-12-08) | schematic, auto-reset circuit, power |
| [SHA2017-badge/Firmware](https://github.com/SHA2017-badge/Firmware) | `e9c4b2a` (2018-08-19) | the original 2017 firmware and its eink driver |
| [SHA2017-badge/micropython-esp32](https://github.com/SHA2017-badge/micropython-esp32) | `e5a4715` (2019-02-21) | original MicroPython port |
| [SHA2017-badge/ugfx](https://github.com/SHA2017-badge/ugfx) | `3981070` (2018-02-18) | original graphics library |

## Files that matter

In `badgeteam/ESP32-platform-firmware`:

- `firmware/configs/sha2017_defconfig`: every pin number, the display type,
  console baud, RX buffer size, watchdog settings, OTA and Hatchery hostnames.
- `firmware/components/driver_display_eink/driver_eink.c`: refresh timing
  constants (62µs per line, 26 dummy lines), `minimal_update_height`, the
  partial-update path.
- `firmware/components/driver_display_eink/driver_eink_lut.c`: the four LUTs and
  their frame counts.
- `firmware/components/driver_framebuffer/include/driver_framebuffer_devices.h`:
  the line that maps the dirty rectangle's x range onto the panel's y range.
- `firmware/components/driver_framebuffer/driver_framebuffer_text.cpp`: font names.
- `firmware/components/micropython/esp32/modframebuffer.c`: the `display` module.
- `firmware/components/micropython/esp32/modeink.c`: the `eink` module.
- `firmware/components/micropython/esp32/modmachine.c`: `stdin_get`,
  `stdout_put`, `stdin_disable`, the NVS functions.
- `firmware/components/micropython/esp32/uart.c` and `mphalport.c`: the UART ISR,
  the Ctrl-C check, the stdin ring buffer.
- `firmware/components/micropython/lib/utils/sys_stdio_mphal.c`: `sys.stdin`
  semantics (blocking, CR translation, no ioctl).
- `firmware/python_modules/sha2017/`: `boot.py`, `system.py`, `easydraw.py`,
  `_mpr121mapping.py`, `dashboard/launcher.py`.

In `SHA2017-badge/PCB`: `sha2017_rev1_0_1_schematic.pdf`, sheet 1 for the CP2102
and display, sheet 2 for LEDs, power and the IO header.

## Live services, checked 2026-07-30

- `https://ota.badge.team/version/sha2017.txt` returns
  `{"build":24021401, "name":"End of an era"}`. Firmware OTA works.
- `https://ota.badge.team/sha2017.bin` serves the image (range request answered
  with 206).
- `https://hatchery.badge.team/` returns HTTP 500. The app repository is down,
  so `woezel` and the on-badge installer cannot fetch anything.
- `https://badge.team/docs/` is up and is the best remaining documentation.
- `https://wiki.badge.team/` serves a certificate for unrelated domains and
  cannot be reached over TLS.

## Documentation used

- [badge.team docs, SHA2017 hardware](https://badge.team/docs/badges/sha2017/hardware/)
- [badge.team docs, display module](https://badge.team/docs/badgepython/api-reference/display/)
- [SHA2017 wiki, badge documentation](https://wiki.sha2017.org/w/Projects:Badge/Documentation)
- [SHA2017 wiki, MicroPython](https://wiki.sha2017.org/w/Projects:Badge/MicroPython)

## Numbers that are computed, not measured

Every refresh time in [FIRMWARE.md](FIRMWARE.md) is arithmetic over the driver's
constants: `frames * (rows + 26) * 62µs`, where the frame count is the sum of the
`length` fields in the chosen LUT. Real refreshes add SPI transfer (about 9.5KB
per update at 20MHz), the 8bpp to 1bpp conversion the CPU does per flush, and
whatever the panel's BUSY line adds. Treat them as a floor.
