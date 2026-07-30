# SHA2017 badge hardware

Board revision 1.0.1 is the one that shipped to attendees. A blue PCB about
95x85mm with the battery on the back.

## Pin assignments

Read from `firmware/configs/sha2017_defconfig` in the badge.team firmware, which
is what the running code actually uses.

| function | pin |
|---|---|
| VSPI CLK | GPIO18 |
| VSPI MOSI | GPIO5 |
| VSPI MISO | not connected (`-1`) |
| e-ink RESET | GPIO23 |
| e-ink BUSY | GPIO22 |
| e-ink CS | GPIO19 |
| e-ink D/C | GPIO21 |
| I2C SDA | GPIO26 |
| I2C SCL | GPIO27 |
| MPR121 interrupt | GPIO25 |
| SK6812 data | GPIO32 |
| UART0 (CP2102) | GPIO1 TX, GPIO3 RX |

The e-ink shares VSPI with the SD card and runs at 20MHz. MISO is unused because
the panel is write-only.

## Display

DKE DEPG0290B1, 2.9", 128x296 native portrait, 1 bit per pixel on the glass. The
firmware presents it as a **296x128 landscape framebuffer** and keeps 8 bits per
pixel in RAM (296*128 = 37888 bytes), which is what makes the greyscale trick
possible. `GDEH029A1` is a pin-compatible alternative selected by populating R26
on the board; the shipped badges are DEPG0290B1.

The driver writes the full 4736-byte bitplane on every update, plus a second
4736-byte copy of the previous image for the DEPG0290B1's differential mode. At
20MHz that is a few milliseconds, so SPI is never the bottleneck. The panel
drive time is. See [FIRMWARE.md](FIRMWARE.md) for the numbers.

## Buttons

An NXP MPR121 at I2C address 0x5A does double duty as capacitive touch
controller and GPIO expander. Its interrupt line is GPIO25.

Electrodes 0-7 map to A, B, START, SELECT, DOWN, RIGHT, UP, LEFT
(`_mpr121mapping.py`). There is no FLASH button on this badge, so code written
for other badge.team boards that expects `BTN_FLASH` will not find it.

Four MPR121 GPIOs drive the vibration motor, the LED power MOSFET, and SD card
power. The firmware config points both the SK6812 power gate and the SD card
power gate at MPR121 pin 10.

## LEDs

Six SK6812 RGBW in the 5050 package above the display, chained, data from
GPIO32. They are powered through a MOSFET gated by the MPR121, so they draw
nothing when the firmware has not enabled them. Driven from MicroPython with the
`neopixel` module: 4 bytes per LED, 24 bytes total.

## Power

- 1000mAh LiPo on a JST-PH connector, with over/under-voltage and overcurrent protection.
- TP4056 charger, 500mA by default, 1A if the jumper is closed. NTC on the pack.
- AP2114H regulator for 3.3V.
- An ideal-diode MOSFET picks USB or battery.
- Divider networks feed `VUSB_SENSE` and `VBAT_SENSE` into the ESP32 ADC, exposed
  as the `voltages` module in MicroPython.

For an installation running continuously on USB power, the battery is not needed
but the charger will keep topping it up if one is fitted.

## USB serial and the auto-reset circuit

A Silicon Labs CP2102 provides USB-UART on UART0 at 115200 baud. Its DTR and RTS
lines run through 10k resistors into two MMBT2222A transistors (Q3, Q4 on
schematic sheet 1) that drive the ESP32's reset and GPIO0. This is the standard
esptool auto-reset arrangement, and it means **any host program that asserts DTR
or RTS on open will reboot the badge**. See [SERIAL.md](SERIAL.md).

macOS needs the Silicon Labs VCP driver on older releases; recent macOS carries
a CP210x driver in-kernel and the port shows up as `/dev/tty.usbserial-*`. The
older SHA2017 docs name `/dev/tty.SLAB_USBtoUART`, which is the Silicon Labs
driver's name for the same thing.

## Expansion header

A 6-pin header (X3) beside the SD card carries switched 3.3V (the SD card power
rail), the I2C bus, and the SK6812 data-out and power lines, so a scarf or add-on
can extend the LED chain and hang I2C devices off the badge. The schematic PDF
in the PCB repo has the pin order; read it there rather than guessing, because
the labels are too small to resolve in a rendered page image.

## SD card

MicroSD slot on switched power, sharing VSPI with the display. The firmware
mounts it and adds it to `sys.path`, so apps can live on the card.
