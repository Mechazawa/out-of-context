# Driving the badge over serial

The link is a CP2102 on ESP32 UART0, 115200 8N1, no hardware flow control. RTS
and CTS on that chip are wired to the ESP32's reset and GPIO0, not to the UART's
flow control pins, so there is no way to make the badge say "stop sending".

## Do not let the host reset the badge

Asserting DTR or RTS reboots the ESP32 through the auto-reset transistors. Most
serial libraries assert both when they open a port. Clear them first.

pyserial:

```python
import serial
port = serial.Serial()
port.port = "/dev/ttyUSB0"
port.baudrate = 115200
port.dtr = False
port.rts = False
port.open()
```

Rust `serialport`:

```rust
let mut port = serialport::new("/dev/ttyUSB0", 115_200).open()?;
port.write_data_terminal_ready(false)?;
port.write_request_to_send(false)?;
```

Whether the lines settle before the ESP32 notices is platform-dependent, so
expect one reboot on connect and design the host to tolerate it rather than
assuming it never happens.

## Reading a stream inside a running app

`machine.stdin_get(size, timeout_ms)` is the primitive to use. It reads up to
`size` bytes, gives up after `timeout_ms`, returns what it got as a `str`, or
`None` if nothing arrived. While it runs it puts UART0 in raw mode, so 0x03 is
delivered as data instead of raising `KeyboardInterrupt`, and CR is not
translated.

```python
import machine, display

line = ""
while True:
    chunk = machine.stdin_get(64, 200)
    if chunk is None:
        continue
    for ch in chunk:
        if ch == "\n":
            render(line)      # draw + flush
            line = ""
        else:
            line += ch
```

The alternative, `sys.stdin.read(n)`, is worse here on three counts: it blocks
until it has exactly `n` bytes with no timeout, it rewrites `\r` to `\n`, and
Ctrl-C raises `KeyboardInterrupt` unless the app calls `micropython.kbd_intr(-1)`
first. `sys.stdin` also has no `ioctl`, so `uselect.poll()` cannot watch it and
there is no non-blocking read. If you do use it, send LF line endings only,
because CRLF arrives as two newlines.

`machine.stdin_disable(pattern)` makes the firmware drop everything on stdin
until it sees a pattern of up to 15 characters. Useful if the host needs a
resync marker after a badge reboot.

`_thread` exists (GIL, three threads max) if a reader thread suits the design
better than a blocking main loop.

## Pacing

The stdin ring buffer is 1080 bytes (`CONFIG_MICROPY_RX_BUFFER_SIZE`), roughly
94ms of wire time at 115200. The badge stops draining it while a flush is in
progress, and a flush blocks for anywhere from about 30ms (small band, fastest
LUT) to 1.5s (full panel, full LUT). Overflow is silent.

At a reading pace of one or two words per second, the host sends around a dozen
bytes per second and 1080 bytes is over a minute of headroom, so nothing needs
doing. Anything bursty needs application-level acknowledgement: the badge prints
a byte after each flush, the host waits for it before sending more. There is no
transport-level backpressure to fall back on.

## Noise on the line

Before an app is running the badge emits ROM bootloader output and firmware log
lines, and after a crash it emits a traceback. A host that parses badge output
should wait for a ready marker the app prints itself rather than assuming the
first bytes it sees are meaningful.

In the other direction, anything the app `print`s lands in the same stream the
host is reading. Prefix app output with a marker, or keep the app silent.

Reboots do not drop the USB device: the CP2102 is powered from USB
independently of the ESP32, so the host's file descriptor stays valid across a
badge restart. The badge is simply deaf for a couple of seconds and then replays
its boot noise.

## Getting a REPL by hand

Connecting a terminal lands on the firmware's serial menu, not a prompt. Choose
"Python shell" from it. From an already-running Python context,
`system.shell()` reboots into a bare REPL, and `machine.RTC().write_string("shell")`
followed by a reset does the same thing.

For pasting more than a line or two, use the REPL's paste mode (Ctrl-E, paste,
Ctrl-D) so auto-indent does not mangle the block.

## Choosing a channel

UART0 through the USB connector is the only sensible data path. The expansion
header carries the I2C bus that the MPR121 needs for the buttons, so
repurposing those pins as a second UART costs the badge its inputs. WiFi is
available in the firmware if a network is acceptable.
