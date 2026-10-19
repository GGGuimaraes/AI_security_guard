"""Orchestration: ties together capture, detection, event, AI, classification and notification.

Two threads, besides the capture thread (which has its own)::

    main      wait_for_frame → detector → debounce → EventBuilder.collect → queue
    worker    queue → AI → classification → SQLite → notification channel

They are split because an AI call takes seconds and must not pause the watching: while
the model thinks, the main thread keeps looking at the scene.

Nothing here may silence an event: an AI error becomes `medium` "analysis unavailable",
a disk or database failure is logged but does not cancel the alert, and channels never raise.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self

from .ai_client import AIClient, AIResponse, AIUnavailable
from .capture import Frame, FrameSource
from .classifier import Analysis, classify, unavailable
from .config import RiskLevel, Settings
from .event import Event, EventBuilder
from .health import HealthMonitor
from .motion import MotionDetector
from .notifier import Notifier, build_alert, build_notifier
from .storage import Storage

log = logging.getLogger(__name__)

FRAME_TIMEOUT_SECONDS = 2.0  # also how often the loop reacts to stop()
CLEANUP_INTERVAL_SECONDS = 900.0  # 15 min: the retention window is hours, not days
SHUTDOWN_TIMEOUT_SECONDS = 60.0
QUEUE_SIZE = 2


class Analyzer(Protocol):
    """What the pipeline uses from the AI client (`AIClient` already provides it)."""

    def analyze(self, event: Event) -> AIResponse: ...

    def close(self) -> None: ...


class Health(Protocol):
    """What the pipeline uses from the health monitor (`HealthMonitor` already provides it)."""

    def start(self, summary: object = "") -> None: ...

    def check(self, last_frame_at: float | None, summary: object = "") -> None: ...


@dataclass(slots=True)
class PipelineStats:
    """Run counters — logged at shutdown."""

    frames: int = 0
    events: int = 0
    analyses: int = 0
    alerts: int = 0

    def __str__(self) -> str:
        return (
            f"{self.frames} frames, {self.events} events, "
            f"{self.analyses} analyses, {self.alerts} alerts"
        )


class Pipeline:
    """The watch loop. `run()` blocks until `stop()`; `stop()` is safe from another thread."""

    def __init__(
        self,
        *,
        source: FrameSource,
        detector: MotionDetector,
        builder: EventBuilder,
        ai: Analyzer,
        notifier: Notifier,
        storage: Storage,
        health: Health | None = None,
        min_risk: RiskLevel = RiskLevel.MEDIUM,
        trigger_frames: int = 2,
        max_images: int = 4,
        timezone: str = "UTC",
        frame_timeout_seconds: float = FRAME_TIMEOUT_SECONDS,
        cleanup_interval_seconds: float = CLEANUP_INTERVAL_SECONDS,
        shutdown_timeout_seconds: float = SHUTDOWN_TIMEOUT_SECONDS,
        queue_size: int = QUEUE_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.detector = detector
        self.builder = builder
        self.ai = ai
        self.notifier = notifier
        self.storage = storage
        self.health = health
        self.min_risk = min_risk
        self.trigger_frames = trigger_frames
        self.max_images = max_images
        self.timezone = timezone
        self.frame_timeout_seconds = frame_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._clock = clock
        self._stats = PipelineStats()
        self._stop = threading.Event()
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=queue_size)
        self._worker: threading.Thread | None = None
        self._streak = 0  # consecutive frames with motion (debounce)
        self._starving = False  # the source stopped delivering frames
        self._last_frame_at: float | None = None  # epoch of the last frame (camera health)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Self:
        source = FrameSource.from_settings(settings)
        notifier = build_notifier(settings)
        options = dict(
            source=source,
            detector=MotionDetector.from_settings(settings),
            builder=EventBuilder.from_settings(settings, source),
            ai=AIClient.from_settings(settings),
            notifier=notifier,
            storage=Storage.from_settings(settings),
            health=HealthMonitor.from_settings(settings, notifier),
            min_risk=settings.notify.min_risk,
            trigger_frames=settings.motion.trigger_frames,
            max_images=settings.notify.max_images,
            timezone=settings.notify.timezone,
        )
        options.update(kwargs)
        return cls(**options)

    def __repr__(self) -> str:
        return (
            f"Pipeline(source={self.source.name!r}, channel={self.notifier.name!r}, "
            f"min_risk={self.min_risk.value}, trigger_frames={self.trigger_frames})"
        )

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    # ---------------------------------------------------------------- lifecycle

    def run(self) -> None:
        """Watches until `stop()`. Always shuts down source, worker, database and AI client."""
        self._stop.clear()
        self._safe("clean up the history", self.storage.cleanup)
        self._worker = threading.Thread(target=self._work, name="ia_watcher-analysis", daemon=True)
        self._worker.start()
        self.source.start()
        log.info("ia_watcher started — %r", self)
        if self.health is not None:
            self._safe("send the startup notice", self.health.start)
        try:
            self._loop()
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Requests shutdown. The loop exits within `frame_timeout_seconds` (or when a
        collection in progress ends, which takes `frames_per_event × frame_interval_seconds`)."""
        self._stop.set()

    def close(self) -> None:
        """Releases the AI client and the database. Idempotent; `run()` already calls it."""
        self._safe("close the AI client", self.ai.close)
        self._safe("close the database", self.storage.close)

    def _loop(self) -> None:
        last_cleanup = self._clock()
        last_seq = 0
        while not self._stop.is_set():
            frame = self.source.wait_for_frame(timeout=self.frame_timeout_seconds, after=last_seq)
            if frame is None:
                self.on_source_lost()
            else:
                last_seq = frame.seq
                self._last_frame_at = frame.ts
                if (event := self.on_frame(frame)) is not None:
                    self._submit(event)
            if self.health is not None:
                self._safe("check health", self.health.check, self._last_frame_at, self._stats)
            if self._clock() - last_cleanup >= self.cleanup_interval_seconds:
                last_cleanup = self._clock()
                self._safe("clean up the history", self.storage.cleanup)

    def _shutdown(self) -> None:
        self.source.stop()
        if self._worker is not None:
            self._queue.put(None)  # sentinel: the worker finishes what is already queued
            self._worker.join(self.shutdown_timeout_seconds)
            if self._worker.is_alive():
                log.warning("analysis did not finish within %.0fs", self.shutdown_timeout_seconds)
            self._worker = None
        self.close()
        log.info("ia_watcher stopped — %s", self._stats)

    # ---------------------------------------------------------------- main thread

    def on_frame(self, frame: Frame) -> Event | None:
        """Detection + debounce + batch collection. Returns the event ready for analysis."""
        self._starving = False
        self._stats.frames += 1
        result = self.detector.update(frame.image)
        self._streak = self._streak + 1 if result.moving else 0
        if self._streak < self.trigger_frames or not self.builder.ready():
            return None

        self._streak = 0
        event = self.builder.collect(result.area_ratio)
        if event is None:
            return None
        self._stats.events += 1
        log.info(
            "motion: event with %d frame(s), %.1f%% of the frame",
            len(event.images),
            result.area_ratio * 100,
        )
        return event

    def on_source_lost(self) -> None:
        """No frames: resets the debounce, since frames before and after the drop are not
        consecutive. Acts only the first time, so the warning is not repeated every timeout."""
        if self._starving:
            return
        self._starving = True
        self._streak = 0
        log.warning("no frames from the source")

    def _submit(self, event: Event) -> None:
        if self._stop.is_set():
            log.warning("shutting down: collected event will not be analyzed")
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            log.error("analysis queue full; event dropped")

    # ---------------------------------------------------------------- worker

    def _work(self) -> None:
        while (event := self._queue.get()) is not None:
            try:
                self.analyze(event)
            except Exception:  # noqa: BLE001 - the worker must not die and blind the watch
                log.exception("unexpected failure analyzing the event")

    def analyze(self, event: Event) -> Analysis:
        """AI → classification → database → channel. Returns what was decided about the event."""
        stored = self._safe("save the event", self.storage.save_event, event)
        analysis, response = self._ask(event)
        self._stats.analyses += 1
        if stored is not None:
            self._safe(
                "save the analysis", self.storage.save_analysis, stored.id, analysis, response
            )
        if analysis.should_notify(self.min_risk):
            self._deliver(event, analysis, stored)
        else:
            log.info("risk %s below %s: recorded without alert", analysis.risk, self.min_risk)
        return analysis

    def _ask(self, event: Event) -> tuple[Analysis, AIResponse | None]:
        """Calls the AI. Any failure becomes the `medium` fallback — never a silent event."""
        try:
            response = self.ai.analyze(event)
        except AIUnavailable as err:
            return unavailable(err), None
        except Exception as err:  # noqa: BLE001 - a client bug must not silence the event either
            log.exception("unexpected error in the AI client")
            return unavailable(type(err).__name__), None
        return classify(response.content), response

    def _deliver(self, event: Event, analysis: Analysis, stored: object | None) -> None:
        alert = build_alert(event, analysis, timezone=self.timezone, max_images=self.max_images)
        ok = self.notifier.send(alert)  # never raises
        self._stats.alerts += int(ok)
        if stored is not None:
            self._safe("record the delivery", self.storage.mark_notified, stored.id, ok)

    def _safe(self, what: str, call: Callable, *args: object):
        """A full disk or a locked database must not cancel the analysis or the alert."""
        try:
            return call(*args)
        except Exception:  # noqa: BLE001 - see above
            log.exception("failed to %s", what)
            return None
