"""Command line entry point: `umpire table`, `umpire run`, `umpire probe`."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

from .rules import Match, MatchConfig
from .sources import AudioSource, VideoSource
from .table import TableGeometry, detect_table


def _point(text: str) -> tuple[int, int]:
    x, y = text.split(",")
    return int(x), int(y)


def _add_video_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", "-s", required=True,
                   help="camera index, /dev/videoN, video file, or URL (e.g. http://PHONE:8080/video)")
    p.add_argument("--width", type=int, help="requested capture width (USB cameras)")
    p.add_argument("--height", type=int, help="requested capture height (USB cameras)")
    p.add_argument("--fps", type=int, help="requested capture fps (USB cameras)")
    p.add_argument("--resize", type=int, metavar="W", help="resize frames to this width before processing")


def _resize(frame, width):
    if not width or frame.shape[1] == width:
        return frame
    h = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def cmd_probe(args) -> int:
    src = VideoSource(args.source, args.width, args.height, args.fps)
    n, start, shape = 0, None, None
    for t, frame in src.frames():
        start = start if start is not None else t
        shape = frame.shape
        n += 1
        if t - start >= args.seconds:
            break
    src.close()
    if n < 2:
        print("no frames received")
        return 1
    print(f"{shape[1]}x{shape[0]}, {(n - 1) / (t - start):.1f} fps measured over {n} frames "
          f"(source reports {src.fps:.1f})")
    if args.audio:
        audio = AudioSource(args.audio)
        got, peak = 0.0, 0.0
        for _, block in audio.blocks():
            got += len(block) / audio.format.rate
            peak = max(peak, float(abs(block).max()))
            if got >= 2:
                break
        audio.close()
        print(f"audio: {audio.format.rate} Hz, {audio.format.channels} ch, peak level {peak:.3f} over {got:.1f}s"
              + ("  <- silent, check the microphone" if peak < 0.001 else ""))
    return 0


def cmd_table(args) -> int:
    src = VideoSource(args.source, args.width, args.height, args.fps)
    frames = []
    step = 1 if src.live else max(int(src.fps * args.spacing), 1)
    for i, (_, frame) in enumerate(src.frames()):
        if i % step == 0:
            frames.append(_resize(frame, args.resize))
        if len(frames) >= args.frames:
            break
        if src.live:
            time.sleep(args.spacing)
    src.close()
    if not frames:
        print("no frames received", file=sys.stderr)
        return 1
    h, w = frames[0].shape[:2]
    if args.corners:
        table = TableGeometry.from_corners([_point(c) for c in args.corners.split(";")], (w, h))
    else:
        det = detect_table(frames, seed=_point(args.seed) if args.seed else None)
        if det is None:
            print("table not found; try --seed X,Y on the table top, or --corners", file=sys.stderr)
            return 1
        table = det.geometry
        print(f"table found (hue {det.hue}, score {det.score:.4f})")
    table.save(args.out)
    preview = frames[-1].copy()
    table.draw(preview)
    cv2.imwrite(args.preview, preview)
    print(f"corners {table.corners.round().astype(int).tolist()} -> {args.out}; check {args.preview}")
    return 0


def cmd_run(args) -> int:
    from .umpire import Umpire, UmpireConfig

    src = VideoSource(args.source, args.width, args.height, args.fps)
    table = TableGeometry.load(args.table)
    match = Match(
        config=MatchConfig(points_to_win=args.points, games_to_win=args.games, rally_timeout=args.rally_timeout),
        names=tuple(args.names),
    )
    config = UmpireConfig(
        ball_colour=None if args.ball == "any" else args.ball,
        bounce_source="audio" if args.audio and args.bounces == "audio" else "vision",
        audio_offset=args.audio_offset,
        log_path=Path(args.log) if args.log else None,
    )
    umpire = None
    web = None
    if args.web:
        from .web import WebView

        host, _, port = args.web.rpartition(":")
        web = WebView(host or "0.0.0.0", int(port))
        print(f"web view on http://{host or '0.0.0.0'}:{port}/")

    writer = None
    wall_start = time.monotonic()
    try:
        for t, frame in src.frames():
            frame = _resize(frame, args.resize)
            if umpire is None:
                h, w = frame.shape[:2]
                umpire = Umpire(table.scaled_to((w, h)), match, config)
                if args.audio:
                    _start_audio(args.audio, umpire, src.live)
                if args.record:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
            umpire.process_frame(t, frame)
            if writer is not None:
                writer.write(frame)
            if web is not None:
                while not web.commands.empty():
                    umpire.command(web.commands.get_nowait())
                overlay = umpire.draw(frame)
                ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
                web.publish(jpeg.tobytes(), umpire.state())
            if args.show:
                cv2.imshow("open-umpire", umpire.draw(frame))
                key = cv2.waitKey(1) & 0xFF
                keys = {ord("q"): None, ord("a"): "point_left", ord("l"): "point_right", ord("u"): "undo",
                        ord("s"): "swap", ord("n"): "let"}
                if key == ord("q"):
                    break
                if key in keys:
                    cmd = keys[key]
                    if cmd.startswith("point_"):
                        left = match.left_player
                        cmd = f"point{left if cmd == 'point_left' else 1 - left}"
                    umpire.command(cmd)
            if not src.live and args.realtime:
                time.sleep(max(t - (time.monotonic() - wall_start), 0))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if writer is not None:
            writer.release()
        if umpire is not None:
            print("final:", umpire.state()["scores"], "games", umpire.state()["games"])
            for line in reversed(list(umpire.log)):
                print(line)
    return 0


def _start_audio(spec: str, umpire, video_live: bool) -> None:
    from .audio import OnsetDetector

    audio = AudioSource(spec)
    if audio.live != video_live:
        print("warning: mixing live and recorded sources, timestamps won't line up", file=sys.stderr)
    detector = OnsetDetector(audio.format.rate)

    def loop():
        for t, block in audio.blocks():
            for onset in detector.process(t, block):
                umpire.add_onset(onset)

    threading.Thread(target=loop, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="umpire", description="Automatic table tennis scoreboard")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="check a camera/phone stream: resolution, real fps, audio level")
    _add_video_args(p)
    p.add_argument("--audio", help="audio source to check as well")
    p.add_argument("--seconds", type=float, default=5)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("table", help="find the table and save its geometry")
    _add_video_args(p)
    p.add_argument("--out", default="table.json")
    p.add_argument("--preview", default="table_preview.jpg")
    p.add_argument("--frames", type=int, default=15, help="frames to median-blend (removes players and ball)")
    p.add_argument("--spacing", type=float, default=0.3, help="seconds between blended frames")
    p.add_argument("--seed", help="X,Y of a point on the table top, if auto-detection picks the wrong thing")
    p.add_argument("--corners", help="manual corners 'x,y;x,y;x,y;x,y' (any order)")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("run", help="referee a match")
    _add_video_args(p)
    p.add_argument("--table", default="table.json")
    p.add_argument("--names", nargs=2, default=["Player 1", "Player 2"], metavar=("FIRST_SERVER", "OTHER"),
                   help="first name is on the left and serves first")
    p.add_argument("--ball", choices=["orange", "white", "any"], default="orange")
    p.add_argument("--audio", help="microphone: http://PHONE:8080/audio.wav, alsa:plughw:1,0, file.wav")
    p.add_argument("--bounces", choices=["vision", "audio"], default="vision",
                   help="what detects bounces (audio needs --audio; camera still decides the side)")
    p.add_argument("--audio-offset", type=float, default=0.0, help="seconds added to audio timestamps")
    p.add_argument("--points", type=int, default=11)
    p.add_argument("--games", type=int, default=3, help="games needed to win the match")
    p.add_argument("--rally-timeout", type=float, default=1.5)
    p.add_argument("--web", metavar="[HOST:]PORT", help="serve the live view, e.g. :8000")
    p.add_argument("--show", action="store_true", help="OpenCV window (needs a display)")
    p.add_argument("--record", metavar="FILE.mp4", help="save the raw video for offline tuning")
    p.add_argument("--log", metavar="FILE.jsonl", help="append every point here")
    p.add_argument("--realtime", action="store_true", help="play files at their real speed")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
