# akunah-open-umpire

Automatic table tennis scoreboard for a Raspberry Pi (or any PC). It watches the table with a camera (a USB camera or a phone over WiFi), listens for bounces with a microphone if one is available, and keeps the score on a web page anyone can open.

Ideas are borrowed from [Helaly96/ImageProcessing-PongPing](https://github.com/Helaly96/ImageProcessing-PongPing) and `pingpong-score-tracker`, but the code is written from scratch because PongPing has no license.

## Quick start

```bash
# 1. Is video (and audio) arriving, and at what real frame rate?
umpire probe -s http://PHONE_IP:8080/video --audio http://PHONE_IP:8080/audio.wav

# 2. Find the table and save its geometry. Check table_preview.jpg afterwards.
umpire table -s http://PHONE_IP:8080/video

# 3. Referee a match. Open http://HOST_IP:8000 in any browser.
umpire run -s http://PHONE_IP:8080/video --audio http://PHONE_IP:8080/audio.wav \
           --bounces audio --web :8000 --names Ana Beto
```

With Poetry on a PC, prefix each command with `poetry run`.

## Installation

**PC / WSL**
```bash
poetry install --all-extras      # includes opencv-python-headless
poetry run pytest
```

**Raspberry Pi** (install OpenCV from apt, not pip: the pip wheels break the xcb plugin there)
```bash
sudo apt install python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e . --no-deps
```

## Setting up the camera

### Placement

This matters more than any setting:

- **Side of the table, level with the net**, so the net is near the middle of the image.
- **Raised and tilted down**: about 40–60 cm above the table top, looking down at 30–45°. The table top must appear as a clear four-sided shape. A camera level with the table only sees a thin line, automatic detection fails and the bounce side becomes unreliable.
- **The whole table in frame**, with some margin past both ends. Use the phone's wide lens (0.5x) if needed.
- **Fixed**: a tripod or a mount. If the camera moves, run `umpire table` again.

### Option A: phone over WiFi with *IP Webcam* (easiest)

Nothing to install on the Pi, and it also works from WSL.

1. Install **IP Webcam** (Pavel Khlebovich) from the Play Store.
2. Settings:

   | Setting | Value | Why |
   |---|---|---|
   | Video resolution | 1280x720 (960x540 if the Pi struggles) | More pixels don't help find the ball and cost CPU |
   | Quality | 60–70 | Less data, steadier WiFi, faster decoding |
   | FPS limit | 60 | Don't let the app cap it; over WiFi you'll usually get ~30 |
   | Focus mode | **Auto, manually triggered** | Tap once to focus on the table; it then stays locked. Continuous modes ("smooth", "aggressive", usually "default") refocus when players move |
   | Scene mode | Default or Auto | HDR lowers fps, adds lag and ghosts the ball; face priority adjusts for players, not the table |
   | Exposure | Locked, short if possible | Less motion blur on the ball, no brightness jumps when someone walks past |
   | Video stabilization | Off | It shifts the image and fools the detection |
   | Night vision / HDR | Off | Lower fps |
   | Audio mode | On | Needed for `--bounces audio` |
   | Power | Prevent sleep, phone on the charger | Android may stop the server otherwise |

3. Tap **Start server**. The app shows `http://PHONE_IP:8080`.
   - Video: `http://PHONE_IP:8080/video` (MJPEG)
   - Audio: `http://PHONE_IP:8080/audio.wav`
4. The phone and the computer running `umpire` **must be on the same network, and that network must let devices talk to each other**. See [Troubleshooting](#troubleshooting). Prefer 5 GHz WiFi.

After changing any setting, run `umpire probe` and trust the *measured* fps, not the one the source reports.

### Option B: phone over USB with `scrcpy` (60 fps, lower latency)

Use this for a permanent install on the Pi. It needs Android 12+ and scrcpy 2.2+, and it doesn't depend on the WiFi network.

```bash
sudo apt install v4l2loopback-dkms adb
sudo modprobe v4l2loopback video_nr=10 card_label=phone exclusive_caps=1
scrcpy --video-source=camera --camera-facing=back --camera-size=1280x720 --camera-fps=60 \
       --v4l2-sink=/dev/video10 --no-playback --no-audio
umpire run -s /dev/video10 ...
```

The phone's microphone doesn't easily reach ALSA this way. For audio, use a USB microphone near the table (`--audio alsa:plughw:1,0`) or piezo sensors under the table on a USB sound card. A Pi 4 may struggle to decode 720p at 60 fps; a Pi 5 should be fine. Check with `umpire probe`.

### Option C: USB camera

```bash
umpire probe -s 0                      # or /dev/video0
umpire run -s /dev/video0 --width 1280 --height 720 --fps 60 ...
```

## Calibrating the table

```bash
umpire table -s http://PHONE_IP:8080/video
```

The command blends several frames to remove players and the ball, finds the table top by colour, and writes:

- `table.json`: the four corners of the playing surface, used by `umpire run --table`. Defaults to `./table.json`.
- `table_preview.jpg`: the frame with the two halves drawn on it (left half blue, right half orange). **Always check it.**

If detection picks the wrong thing:

```bash
umpire table -s ... --seed 400,300                  # a point on the table top
umpire table -s ... --corners "x,y;x,y;x,y;x,y"     # the four corners by hand, any order
```

To find pixel coordinates, grab a frame with `curl -o shot.jpg http://PHONE_IP:8080/shot.jpg` and open it in any image viewer.

The net is placed at the projective centre of the table (u = 0.5), not detected. The halves split there, so the corners should include the whole playing surface on both sides of the net. The net outline in the preview is only a drawing.

## Refereeing a match

```bash
umpire run -s http://PHONE_IP:8080/video --web :8000 --names Ana Beto \
           --audio http://PHONE_IP:8080/audio.wav --bounces audio \
           --record rally.mp4 --log match.jsonl
```

| Option | Default | Meaning |
|---|---|---|
| `--names FIRST_SERVER OTHER` | `Player 1 Player 2` | Player names; the first one serves first |
| `--table FILE` | `table.json` | Calibration from `umpire table` |
| `--ball orange\|white\|any` | `orange` | Ball colour mask |
| `--audio SPEC` | none | Microphone: `http://PHONE:8080/audio.wav`, `alsa:plughw:1,0`, a `.wav` file, `ffmpeg:<input>` |
| `--bounces vision\|audio` | `vision` | What detects bounces. `audio` is recommended at 30 fps; the camera still decides the side |
| `--audio-offset SECONDS` | `0` | Added to audio timestamps; positive if audio arrives before video |
| `--points N` | `11` | Points per game |
| `--games N` | `3` | Games needed to win the match |
| `--rally-timeout SECONDS` | `1.5` | Silence that ends a rally |
| `--web [HOST:]PORT` | off | Live scoreboard in the browser |
| `--show` | off | OpenCV window (needs a display) |
| `--record FILE.mp4` | off | Save raw video for offline tuning |
| `--log FILE.jsonl` | off | Append every point |
| `--realtime` | off | Play files at their real speed |

**Web view**: open `http://HOST_IP:8000` from any device on the network. You can add a point to either side, undo, call a *let*, switch sides and reset.

**Keyboard** (with `--show`): `a` point left, `l` point right, `u` undo, `n` let, `s` switch sides, `q` quit.

**Replaying a recording** to tune detection without playing:

```bash
umpire run -s rally.mp4 --table table.json --show --realtime
```

## How points are decided

1. **Table** (`table.py`): colour detection. Several frames are median-blended to erase players and the ball, then hue windows are tried and the blob that best looks like a table top wins: large, solid, four-sided, saturated, wider than tall. The halves come from the homography of the four corners.
2. **Ball** (`ball.py`): three-frame differencing (so the ball is found whether it is lighter or darker than the background), optionally with a colour mask. Only the table area is searched, and the candidate closest to the predicted position is preferred.
3. **Bounce**:
   - by **vision**: the ball switches from falling to rising over one half;
   - by **audio** (`--bounces audio`): the microphone hears the click, and the camera says which half the ball was over at that moment. If it wasn't over the table, it is treated as a paddle hit and ignored.
4. **Rules** (`rules.py`): when the rally ends, the point goes against the player on the half where the ball last bounced. Also:
   - the serve must bounce first on the server's half;
   - two bounces in a row on the same half end the rally;
   - games to 11 by 2, two serves each and one each from 10-10, side switch between games, and undo. `Match.start_scores` already supports starting with a point advantage (for handicaps).

Why fps matters: a ping pong ball travels at 5–15 m/s, so at 30 fps it moves 15–50 cm between frames and the camera can miss a bounce. That's why audio bounces help so much below 60 fps.

## Troubleshooting

**`could not open video source` / `No route to host`**: the computer can't reach the phone.
- Check the IP shown in IP Webcam (it changes when you join another network) and that the server is running with the app in the foreground.
- Try `ping PHONE_IP` and `curl -sI http://PHONE_IP:8080/`.
- If only the router answers, the WiFi probably has **client isolation** (common on office or guest networks): devices reach the internet but not each other. Use a network without it, the phone's hotspot, or a dedicated router for the table.

**`Address already in use`** with `--web`: another `umpire run` is still listening on that port (often one left running or suspended with Ctrl+Z in another terminal). Stop it (`ss -ltnp 'sport = :8000'` shows the PID) or use another port.

**`table not found`**: the camera is probably too low or the table isn't fully in frame (see [Placement](#placement)). Use `--seed` or `--corners`.

**Audio peak level is low in `umpire probe`**: that's usually fine. The bounce detector looks for clicks 12 dB above the background noise, not an absolute level. Run `probe` while someone bounces a ball near the phone to check that the level rises. Below 0.001 means silence, so check the microphone.

**Audio bounces land on the wrong side**: audio and video arrive with different delays. Adjust `--audio-offset`.

## Roadmap

- Tune detection with real recordings (orange ball, phone, 640x480 or 720p).
- Better audio/vision fusion: use audio to confirm the bounces the camera sees.
- UI to load players and random handicaps.
