"""Detection benchmark: runs a reference video through the real modules.

It serves three purposes:

1. **Tuning the sensitivity** without depending on the camera or the time of day — the
   same video always gives the same result, so `motion.*` can be compared before and
   after a change.
2. **Alert benchmark**: how many events the video produces, with what area, and what the
   AI answers for each one (with cost and tokens).
3. **Demo**: run directly (`python tests/test_benchmark.py`) it prints the whole report,
   with no pytest in between.

The clock is **the video itself** (`VideoSource`), not the wall clock: collecting an event
"sleeps" by skipping frames, exactly as the real pipeline does live, and the result does
not depend on how fast the machine is.

Each event's JPEGs — **the same bytes that went to the model** — are kept in
`data/bench/frames/<video>/NN-tSSS.Ss/`. The folder is recreated on every run.

The video does **not** go into git: it shows the inside of a home (`data/` is
gitignored). Without it, the reference tests are skipped; the harness itself is still
covered by the synthetic video.

    uv run pytest -q -m bench                          # sensitivity, no network
    uv run pytest -q -m live tests/test_benchmark.py   # + real AI classification
    uv run python tests/test_benchmark.py              # full report
"""

from __future__ import annotations

import os
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest

from ia_watcher.ai_client import AIClient, AIResponse, Usage
from ia_watcher.capture import Frame
from ia_watcher.classifier import Analysis
from ia_watcher.config import AI_SECRETS, RiskLevel, Settings, load_settings
from ia_watcher.event import Event, EventBuilder
from ia_watcher.motion import MotionDetector, MotionResult
from ia_watcher.notifier import Alert
from ia_watcher.pipeline import Pipeline
from ia_watcher.storage import Storage

#: Reference video. Override with IA_WATCHER_BENCH_VIDEO=/path/to/video.mp4
BENCH_VIDEO = Path(os.environ.get("IA_WATCHER_BENCH_VIDEO", "data/bench/reference.mp4"))
CONFIG = os.environ.get("IA_WATCHER_BENCH_CONFIG", "config/config.yaml")

#: Where the JPEGs the benchmark sent to the AI are kept — one subfolder per video.
FRAMES_ROOT = Path(os.environ.get("IA_WATCHER_BENCH_FRAMES", "data/bench/frames"))

#: "... 2026-09-03 at 12.58.35.mp4" and friends: date + time with any separator.
_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})\D+(\d{2})\D?(\d{2})\D?(\d{2})")


def started_at_of(path: Path, tz: ZoneInfo) -> tuple[float, str]:
    """Wall-clock time of the first frame, and where it came from.

    Not a detail: the time goes into the prompt and the criteria weigh day vs. night.
    With the epoch-0 default the model would judge a scene from 1969 at 21:00 — night —
    and the benchmark would measure something else. The time in the file name beats the
    mtime, which is usually when the video was copied, not recorded.
    """
    found = _DATE_IN_NAME.search(path.name)
    if found:
        year, month, day, hour, minute, second = (int(g) for g in found.groups())
        try:
            when = datetime(year, month, day, hour, minute, second, tzinfo=tz)
        except ValueError:
            pass  # numbers in the name that don't form a date: fall back to the mtime
        else:
            return when.timestamp(), "file name"
    return path.stat().st_mtime, "file mtime"


# ============================================================ video-driven source


class VideoSource:
    """Reads a video frame by frame and acts as the pipeline's clock.

    Differs from `FrameSource` on purpose: no thread, no reconnecting, no dropping frames
    on its own. What skips frames is the `EventBuilder`, when it "sleeps" between an
    event's images — and that gap is exactly what the benchmark needs to reproduce.
    """

    name = "video://benchmark"

    def __init__(self, path: Path | str, *, started_at: float = 0.0) -> None:
        """`started_at` is the wall-clock time of the first frame — see `started_at_of`."""
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"could not open the video: {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.started_at = started_at
        self.index = 0
        self.skipped = 0

    def __repr__(self) -> str:
        return f"VideoSource({self.path.name}, {self.fps:.1f} fps, {self.index} frames)"

    # -------------------------------------------------- clock (injected into the builder)

    def clock(self) -> float:
        """Elapsed time *in the video*. Replaces the EventBuilder's `time.monotonic`."""
        return self.index / self.fps

    def sleep(self, seconds: float) -> None:
        """Advances the video instead of waiting — and counts the frames nobody saw."""
        for _ in range(round(seconds * self.fps)):
            if self._read() is None:
                return
            self.skipped += 1

    # -------------------------------------------------- reading

    def next_frame(self) -> Frame | None:
        """Next frame for the main loop. None at the end of the video."""
        return self._read()

    def wait_for_frame(self, timeout: float = 5.0, after: int = 0) -> Frame | None:
        """What the `EventBuilder` uses. No waiting here: the next frame is the next one."""
        return self._read()

    def start(self) -> VideoSource:
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._cap.release()

    def _read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        self.index += 1
        return Frame(image=image, ts=self.started_at + self.index / self.fps, seq=self.index)


class _SpyDetector:
    """Passes everything to the real detector and keeps each `MotionResult` for the summary."""

    def __init__(self, inner: MotionDetector) -> None:
        self.inner = inner
        self.results: list[MotionResult] = []

    def update(self, image: np.ndarray) -> MotionResult:
        result = self.inner.update(image)
        self.results.append(result)
        return result


class _SpyAI:
    """Passes calls to the real client and keeps each response — that's where cost and cache
    come from.

    It exists so the `Pipeline` doesn't have to expose anything just for the benchmark.
    """

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.responses: list[AIResponse | None] = []

    def analyze(self, event: Event) -> AIResponse:
        try:
            response = self.inner.analyze(event)
        except Exception:
            self.responses.append(None)  # the AI failed: the pipeline falls back to medium
            raise
        self.responses.append(response)
        return response

    def close(self) -> None:
        self.inner.close()


class _CollectingNotifier:
    """A channel that only keeps what would have gone out — the benchmark sends nothing."""

    name = "benchmark"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


class FakeAI:
    """Fake AI to measure only the detector's sensitivity, with no network and no cost."""

    def __init__(self, risk: str = "medium") -> None:
        self.risk = risk
        self.calls = 0

    def analyze(self, event: Event) -> AIResponse:
        self.calls += 1
        return AIResponse(
            content=f'{{"risk": "{self.risk}", "reason": "offline benchmark",'
            f' "description": "", "confidence": 0.0, "people": 0}}',
            model="fake/offline",
            usage=Usage(),
            latency_seconds=0.0,
        )

    def close(self) -> None:
        pass


# ============================================================ run and report


@dataclass(frozen=True, slots=True)
class BenchEvent:
    """One event from the video, already classified."""

    at_seconds: float  # position in the video, not wall-clock time
    images: int
    area_ratio: float
    analysis: Analysis
    cost: float = 0.0
    cached_tokens: int = 0
    frames_dir: Path | None = None  # where the JPEGs sent to the AI are


@dataclass(slots=True)
class BenchResult:
    """What the video produced, from the detector to the AI's answer."""

    video: Path
    fps: float
    frames: int
    skipped: int
    started_at: float = 0.0
    started_from: str = ""
    timezone: str = "UTC"
    frames_dir: Path | None = None
    events: list[BenchEvent] = field(default_factory=list)
    results: list[MotionResult] = field(default_factory=list)
    alerts: int = 0

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def cost(self) -> float:
        return sum(e.cost for e in self.events)

    def areas(self, *, moving: bool) -> list[float]:
        return [r.area_ratio for r in self.results if not r.warming_up and r.moving is moving]

    @property
    def blind_ratio(self) -> float:
        """Fraction of frames in which the detector could see nothing: warm-up and the
        frames skipped while collecting an event."""
        blind = sum(r.warming_up for r in self.results) + self.skipped
        total = len(self.results) + self.skipped
        return blind / total if total else 0.0

    def report(self) -> str:
        lines = [
            f"video:      {self.video}",
            f"recorded:   {self._start()} ({self.started_from}) — the time that goes in the prompt",
            f"duration:   {self.seconds:.1f}s · {self.frames} frames at {self.fps:.1f} fps",
            f"motion:     {len(self.areas(moving=True))} frames above the threshold",
            f"blind:      {self.blind_ratio:.0%} of frames "
            f"(warm-up + {self.skipped} skipped while collecting)",
            _range("area still ", self.areas(moving=False)),
            _range("area moving", self.areas(moving=True)),
            "",
            f"events:     {len(self.events)} · alerts: {self.alerts} · cost US$ {self.cost:.5f}",
        ]
        if self.frames_dir:
            lines.append(f"frames:     {self.frames_dir}/ (what was sent to the AI)")
        if self.events:
            lines.append("")
            lines.append(f"  {'at':>7}  {'img':>3}  {'area':>6}  {'risk':<6} reason")
            for e in self.events:
                lines.append(
                    f"  {e.at_seconds:6.1f}s  {e.images:>3}  {e.area_ratio:6.1%}  "
                    f"{e.analysis.risk.value:<6} {e.analysis.reason[:52]}"
                )
        return "\n".join(lines)

    def _start(self) -> str:
        when = datetime.fromtimestamp(self.started_at, ZoneInfo(self.timezone))
        return f"{when:%Y-%m-%d %H:%M:%S}"


def _range(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: —"
    p99 = statistics.quantiles(values, n=100)[98] if len(values) > 2 else max(values)
    return (
        f"{label}: min={min(values):.2%} median={statistics.median(values):.2%} "
        f"p99={p99:.2%} max={max(values):.2%}"
    )


def run_bench(
    video: Path | str,
    settings: Settings,
    *,
    ai: object | None = None,
    storage: Storage | None = None,
    frames_dir: Path | str | None = None,
    started_at: float | None = None,
) -> BenchResult:
    """Runs the video through the real `Pipeline` and returns what happened.

    Uses `pipeline.on_frame` and `pipeline.analyze` — the same methods as the 24/7
    service — so the benchmark measures the real behavior, not a reimplementation.

    `frames_dir` keeps each event's JPEGs: **exactly** the bytes sent to the model,
    already resized and compressed. The folder is recreated on every run.
    """
    video = Path(video)
    tz = ZoneInfo(settings.notify.timezone)
    start, origin = (started_at, "given") if started_at else started_at_of(video, tz)
    target = Path(frames_dir) if frames_dir else None
    if target is not None:
        shutil.rmtree(target, ignore_errors=True)

    source = VideoSource(video, started_at=start)
    detector = _SpyDetector(MotionDetector.from_settings(settings))
    notifier = _CollectingNotifier()
    spy_ai = _SpyAI(ai if ai is not None else FakeAI())
    pipeline = Pipeline(
        source=source,
        detector=detector,
        builder=EventBuilder.from_settings(
            settings, source, clock=source.clock, sleep=source.sleep
        ),
        ai=spy_ai,
        notifier=notifier,
        storage=storage or _NullStorage(),
        min_risk=settings.notify.min_risk,
        trigger_frames=settings.motion.trigger_frames,
        max_images=settings.notify.max_images,
        timezone=settings.notify.timezone,
    )

    events: list[BenchEvent] = []
    try:
        while (frame := source.next_frame()) is not None:
            event = pipeline.on_frame(frame)
            if event is None:
                continue
            at = source.clock()
            folder = _save_frames(target, len(events) + 1, event, at) if target else None
            analysis = pipeline.analyze(event)
            usage = spy_ai.responses[-1].usage if spy_ai.responses[-1] else Usage()
            events.append(
                BenchEvent(
                    at_seconds=at,
                    images=len(event.images),
                    area_ratio=event.area_ratio,
                    analysis=analysis,
                    cost=usage.cost,
                    cached_tokens=usage.cached_tokens,
                    frames_dir=folder,
                )
            )
    finally:
        source.stop()

    return BenchResult(
        video=video,
        fps=source.fps,
        frames=source.index,
        skipped=source.skipped,
        started_at=start,
        started_from=origin,
        timezone=settings.notify.timezone,
        frames_dir=target,
        events=events,
        results=detector.results,
        alerts=len(notifier.alerts),
    )


def _save_frames(root: Path, index: int, event: Event, at: float) -> Path:
    """Writes the event's JPEGs: the same bytes the AI received, in the same order."""
    target = root / f"{index:02d}-t{at:06.1f}s"
    target.mkdir(parents=True, exist_ok=True)
    for number, image in enumerate(event.images, start=1):
        (target / f"{number}.jpg").write_bytes(image.data)
    return target


class _NullStorage:
    """The benchmark needs no database: measuring is not watching."""

    keep_hours = 12

    def save_event(self, event: Event) -> None:
        return None

    def save_analysis(self, *args: object) -> None: ...

    def mark_notified(self, *args: object) -> None: ...

    def cleanup(self) -> int:
        return 0

    def close(self) -> None: ...


# ============================================================ synthetic video (self-test)


def make_sample_video(path: Path, *, fps: float = 10.0, size: tuple[int, int] = (320, 240)) -> Path:
    """Generates a short video: still background, someone crossing, still background again."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV build without the mp4v encoder")
    rng = np.random.default_rng(0)
    base = rng.integers(40, 80, size=(h, w, 3), dtype=np.uint8)
    try:
        for _ in range(20):  # still scene: warm-up
            writer.write(base)
        for i in range(40):  # someone crossing the frame
            frame = base.copy()
            x = int(0.1 * w + 0.018 * w * i)
            frame[int(0.35 * h) : int(0.75 * h), x : x + int(0.18 * w)] = 230
            writer.write(frame)
        for _ in range(20):  # still scene again
            writer.write(base)
    finally:
        writer.release()
    return path


# ============================================================ tests


@pytest.fixture
def settings(example_config, fake_env) -> Settings:
    return load_settings(example_config, env=fake_env)


def test_harness_runs_a_video_from_start_to_end(tmp_path: Path, settings: Settings):
    """Proves the benchmark with a synthetic video — always runs, no network, no external file."""
    video = make_sample_video(tmp_path / "sample.mp4")
    settings.event.frames_per_event = 2
    settings.event.frame_interval_seconds = 0.5

    result = run_bench(video, settings)

    assert result.frames > 0
    assert len(result.events) == 1, result.report()
    (event,) = result.events
    assert event.images == 2
    assert event.area_ratio > settings.motion.min_area_ratio
    assert result.skipped > 0  # collection skipped frames, as it does live
    assert event.frames_dir is None  # without frames_dir, nothing is written


def test_frames_sent_to_the_ai_are_kept_on_disk(tmp_path: Path, settings: Settings):
    video = make_sample_video(tmp_path / "sample.mp4")
    settings.event.frames_per_event = 2
    settings.event.frame_interval_seconds = 0.5

    result = run_bench(video, settings, frames_dir=tmp_path / "frames")

    (event,) = result.events
    assert sorted(p.name for p in event.frames_dir.glob("*.jpg")) == ["1.jpg", "2.jpg"]
    assert (event.frames_dir / "1.jpg").read_bytes().startswith(b"\xff\xd8")  # JPEG


def test_prompt_time_comes_from_the_file_name(tmp_path: Path, settings: Settings):
    """Otherwise the prompt would say 1969-12-31 21:00 — night, which the criteria weigh."""
    video = make_sample_video(tmp_path / "WhatsApp Video 2026-09-03 at 12.58.35.mp4")

    result = run_bench(video, settings)

    assert result.started_from == "file name"
    assert "2026-09-03 12:58:35" in result.report()


def test_name_without_a_date_falls_back_to_the_mtime(tmp_path: Path, settings: Settings):
    video = make_sample_video(tmp_path / "sample.mp4")

    result = run_bench(video, settings)

    assert result.started_from == "file mtime"
    assert result.started_at == pytest.approx(video.stat().st_mtime)


def test_the_benchmark_clock_is_the_video_not_the_wall(tmp_path: Path, settings: Settings):
    """Two runs of the same video must give exactly the same result."""
    video = make_sample_video(tmp_path / "sample.mp4")
    settings.event.frames_per_event = 2
    settings.event.frame_interval_seconds = 0.5

    first = run_bench(video, settings)
    second = run_bench(video, settings)

    assert [e.at_seconds for e in first.events] == [e.at_seconds for e in second.events]
    assert first.frames == second.frames
    assert first.blind_ratio == second.blind_ratio


def test_still_scene_produces_no_event(tmp_path: Path, settings: Settings):
    video = tmp_path / "still.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV build without the mp4v encoder")
    rng = np.random.default_rng(0)
    base = rng.integers(40, 80, size=(240, 320, 3), dtype=np.uint8)
    for _ in range(60):
        writer.write(base)
    writer.release()

    result = run_bench(video, settings)

    assert result.events == []


def test_report_does_not_break_without_events(tmp_path: Path, settings: Settings):
    video = make_sample_video(tmp_path / "sample.mp4")
    settings.motion.min_area_ratio = 0.99  # nothing gets through

    text = run_bench(video, settings).report()

    assert "events:     0" in text


# ------------------------------------------------------------ real reference video


def _frames_dir(video: Path) -> Path:
    """One folder per video, so two reference videos don't overwrite each other."""
    return FRAMES_ROOT / video.stem


requires_video = pytest.mark.skipif(
    not BENCH_VIDEO.is_file(),
    reason=f"no reference video at {BENCH_VIDEO} (set IA_WATCHER_BENCH_VIDEO to another one)",
)


@pytest.mark.bench
@requires_video
def test_reference_video_sensitivity(capsys):
    """No network: detector only. This is where `motion.*` is compared before and after a change."""
    result = run_bench(BENCH_VIDEO, load_settings(CONFIG), frames_dir=_frames_dir(BENCH_VIDEO))
    with capsys.disabled():
        print(f"\n{result.report()}\n")

    assert result.frames > 0
    assert result.blind_ratio < 0.5, "detector blind for more than half of the video"


@pytest.mark.live
@requires_video
def test_reference_video_classification(capsys):
    """Sends the video's events to the real model. Spends OpenRouter credits."""
    settings = load_settings(CONFIG)
    settings.secrets.require(*AI_SECRETS)
    with AIClient.from_settings(settings) as ai:
        result = run_bench(BENCH_VIDEO, settings, ai=ai, frames_dir=_frames_dir(BENCH_VIDEO))
    with capsys.disabled():
        print(f"\n{result.report()}\n")

    assert result.events, "the video produced no event to classify"
    assert all(e.analysis.available for e in result.events), "the AI did not answer some event"
    assert all(isinstance(e.analysis.risk, RiskLevel) for e in result.events)


# ============================================================ direct run (demo)


def main(argv: list[str]) -> int:
    video = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else BENCH_VIDEO
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        print("usage: uv run python tests/test_benchmark.py [video.mp4] [--ai]", file=sys.stderr)
        return 2

    settings = load_settings(CONFIG)
    frames = _frames_dir(video)
    if "--ai" in argv:
        settings.secrets.require(*AI_SECRETS)
        with AIClient.from_settings(settings) as ai:
            result = run_bench(video, settings, ai=ai, frames_dir=frames)
    else:
        print("(no --ai: fake classification, detector sensitivity only)\n")
        result = run_bench(video, settings, frames_dir=frames)

    print(result.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
