"""FedLoRA Music desktop client (Toga).

One window, four steps: set up -> prepare songs -> join the federation -> generate.
All heavy work runs as child processes in the ACE-Step Python environment, so this
app stays small and responsive, and model memory is freed when a job ends.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from fedlora_music_ui.settings import (
    Settings,
    child_env,
    generate_cmd,
    identity_cmd,
    open_file_cmd,
    prepare_cmd,
    read_status,
    supernode_cmd,
    write_node_config,
)

_AUDIO_EXT = (".flac", ".wav", ".mp3", ".opus", ".aac")
_LOG_LINES = 600


class Job:
    """A child process whose output is streamed line by line onto the UI loop."""

    def __init__(
        self,
        cmd: list[str],
        loop: asyncio.AbstractEventLoop,
        on_line: Callable[[str], None],
        on_exit: Callable[[int], None],
    ) -> None:
        self._loop = loop
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env(),
            start_new_session=True,  # own process group: stop() reaches ClientApp children
        )
        threading.Thread(target=self._pump, args=(on_line, on_exit), daemon=True).start()

    def _pump(self, on_line: Callable[[str], None], on_exit: Callable[[int], None]) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._loop.call_soon_threadsafe(on_line, line.rstrip())
        code = self.proc.wait()
        self._loop.call_soon_threadsafe(on_exit, code)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if not self.running:
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)


def _col(*children: toga.Widget, margin: int = 6) -> toga.Box:
    return toga.Box(children=list(children), style=Pack(direction=COLUMN, margin=margin))


def _row(*children: toga.Widget) -> toga.Box:
    return toga.Box(children=list(children), style=Pack(direction=ROW, margin_bottom=4))


def _label(text: str, width: int = 170) -> toga.Label:
    return toga.Label(text, style=Pack(width=width, margin_top=6))


def _heading(text: str) -> toga.Label:
    return toga.Label(text, style=Pack(font_weight="bold", font_size=14, margin_top=10))


class FedLoRAApp(toga.App):
    def startup(self) -> None:
        self.settings_path = Path(self.paths.config) / "settings.json"
        self.s = Settings.load(self.settings_path)
        if not self.s.data_dir:
            self.s.data_dir = str(Path(self.paths.data) / "client")

        self.task: Job | None = None  # prepare / identity / generate (one at a time)
        self.node: Job | None = None  # the SuperNode while joined
        self.last_audio: str | None = None
        self._log: deque[str] = deque(maxlen=_LOG_LINES)

        # Setup ---------------------------------------------------------------
        self.ace_in = toga.TextInput(value=self.s.ace_project_root, style=Pack(flex=1))
        self.python_in = toga.TextInput(
            value=self.s.python, placeholder="default: <ACE-Step>/.venv/bin/python",
            style=Pack(flex=1),
        )  # fmt: skip
        self.songs_in = toga.TextInput(value=self.s.songs_dir, style=Pack(flex=1))
        self.data_in = toga.TextInput(value=self.s.data_dir, style=Pack(flex=1))
        self.link_in = toga.TextInput(value=self.s.superlink, style=Pack(flex=1))
        self.ca_in = toga.TextInput(value=self.s.ca_cert, style=Pack(flex=1))
        self.insecure_sw = toga.Switch("Local testing (no TLS, no identity)", value=self.s.insecure)
        self.eps_in = toga.NumberInput(value=self.s.epsilon_budget, min=0.1, step=0.5)
        self.sigma_in = toga.NumberInput(value=self.s.min_noise_multiplier, min=0.1, step=0.1)

        # Actions -------------------------------------------------------------
        self.prepare_btn = toga.Button("Prepare my songs", on_press=self.on_prepare)
        self.identity_btn = toga.Button("Create identity key", on_press=self.on_identity)
        self.join_btn = toga.Button("Join federation", on_press=self.on_join_toggle)
        self.caption_in = toga.TextInput(
            placeholder="e.g. warm lo-fi hip hop, dusty drums, rhodes", style=Pack(flex=1)
        )
        self.duration_in = toga.NumberInput(value=30, min=10, max=600, step=5)
        self.generate_btn = toga.Button("Generate", on_press=self.on_generate)
        self.play_btn = toga.Button("Play last", on_press=self.on_play, enabled=False)

        self.status_lbl = toga.Label("", style=Pack(margin_top=8))
        self.log_view = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, height=220))

        def browse(target: toga.TextInput, file: bool = False) -> toga.Button:
            async def pick(_: toga.Widget, **kwargs: object) -> None:
                dlg = (
                    toga.OpenFileDialog("Choose file", file_types=["crt", "pem"])
                    if file
                    else toga.SelectFolderDialog("Choose folder")
                )
                path = await self.main_window.dialog(dlg)
                if path:
                    target.value = str(path)

            return toga.Button("…", on_press=pick, style=Pack(width=40))

        content = _col(
            _heading("1 · Setup"),
            _row(_label("ACE-Step folder"), self.ace_in, browse(self.ace_in)),
            _row(_label("Python runtime"), self.python_in),
            _row(_label("My songs"), self.songs_in, browse(self.songs_in)),
            _row(_label("Private data folder"), self.data_in, browse(self.data_in)),
            _heading("2 · Privacy (your limits)"),
            _row(_label("Max total ε"), self.eps_in),
            _row(_label("Min noise σ"), self.sigma_in),
            _heading("3 · Federation"),
            _row(_label("Server (host:port)"), self.link_in),
            _row(_label("Server CA certificate"), self.ca_in, browse(self.ca_in, file=True)),
            _row(self.insecure_sw),
            _row(self.prepare_btn, self.identity_btn, self.join_btn),
            _heading("4 · Generate in your style"),
            _row(_label("Describe the music"), self.caption_in),
            _row(_label("Seconds"), self.duration_in, self.generate_btn, self.play_btn),
            self.status_lbl,
            self.log_view,
            margin=14,
        )

        self.main_window = toga.MainWindow(title=self.formal_name, size=(820, 900))
        self.main_window.content = toga.ScrollContainer(content=content, horizontal=False)
        self.on_exit = self._on_exit
        self.main_window.show()
        self._refresh_status()

    async def on_running(self) -> None:
        while True:  # progress comes from local files written by the ClientApp
            await asyncio.sleep(3)
            self._refresh_status()

    # -- helpers -----------------------------------------------------------------
    def _collect(self) -> Settings:
        s = self.s
        s.ace_project_root = self.ace_in.value.strip()
        s.python = self.python_in.value.strip()
        s.songs_dir = self.songs_in.value.strip()
        s.data_dir = self.data_in.value.strip()
        s.superlink = self.link_in.value.strip()
        s.ca_cert = self.ca_in.value.strip()
        s.insecure = bool(self.insecure_sw.value)
        s.epsilon_budget = float(self.eps_in.value or s.epsilon_budget)
        s.min_noise_multiplier = float(self.sigma_in.value or s.min_noise_multiplier)
        s.save(self.settings_path)
        return s

    def _append(self, line: str) -> None:
        self._log.append(line)
        self.log_view.value = "\n".join(self._log)
        self.log_view.scroll_to_bottom()

    async def _blocked(self, action: str) -> bool:
        problems = self._collect().problems(action)
        if problems:
            await self.main_window.dialog(toga.ErrorDialog("Not ready", "\n".join(problems)))
        return bool(problems)

    def _set_busy(self, busy: bool) -> None:
        for b in (self.prepare_btn, self.identity_btn, self.generate_btn):
            b.enabled = not busy

    def _start_task(self, cmd: list[str], on_done: Callable[[int], None]) -> None:
        self._set_busy(True)
        self._append(f"$ {' '.join(cmd[:4])} …")

        def finished(code: int) -> None:
            self._set_busy(False)
            self._append(f"[exit {code}]")
            on_done(code)

        self.task = Job(cmd, self.loop, self._append, finished)

    def _refresh_status(self) -> None:
        try:
            st = read_status(self.s)
        except (OSError, ValueError, KeyError):
            return
        joined = "joined" if self.node and self.node.running else "not joined"
        loss = f" · last loss {st.last_loss:.4f}" if isinstance(st.last_loss, float) else ""
        self.status_lbl.text = (
            f"{joined} · rounds {st.rounds} · ε spent {st.epsilon:.2f} / "
            f"{self.s.epsilon_budget:.1f}{loss}"
        )

    # -- actions -----------------------------------------------------------------
    async def on_prepare(self, _: toga.Widget, **kwargs: object) -> None:
        if not await self._blocked("prepare"):
            self._start_task(prepare_cmd(self.s), lambda code: None)

    async def on_identity(self, _: toga.Widget, **kwargs: object) -> None:
        if await self._blocked("identity"):
            return
        captured: list[str] = []
        cmd = identity_cmd(self.s)
        self._set_busy(True)

        def line(text: str) -> None:
            captured.append(text)
            self._append(text)

        def done(code: int) -> None:
            self._set_busy(False)
            if code == 0 and captured:
                pub = self.s.key_dir / "supernode_key.pub"
                self.loop.create_task(
                    self.main_window.dialog(
                        toga.InfoDialog(
                            "Send this public key to the operator",
                            f"File: {pub}\n\n{captured[-1]}\n\n"
                            "They register it with `flwr supernode register`. "
                            "Your private key never leaves this Mac.",
                        )
                    )
                )

        self.task = Job(cmd, self.loop, line, done)

    async def on_join_toggle(self, _: toga.Widget, **kwargs: object) -> None:
        if self.node and self.node.running:
            self.join_btn.enabled = False
            await asyncio.to_thread(self.node.stop)
            return
        if await self._blocked("join"):
            return
        write_node_config(self.s)
        self._append(f"Joining {self.s.superlink} (ε ≤ {self.s.epsilon_budget}, σ ≥ "
                     f"{self.s.min_noise_multiplier})")  # fmt: skip

        def left(code: int) -> None:
            self.join_btn.text, self.join_btn.enabled = "Join federation", True
            self._append(f"Left federation [exit {code}]")
            self._refresh_status()

        self.node = Job(supernode_cmd(self.s), self.loop, self._append, left)
        self.join_btn.text = "Leave federation"

    async def on_generate(self, _: toga.Widget, **kwargs: object) -> None:
        if await self._blocked("generate"):
            return
        caption = self.caption_in.value.strip()
        if not caption:
            await self.main_window.dialog(toga.ErrorDialog("Missing", "Describe the music."))
            return
        if self.node and self.node.running:
            self._append("Note: training may be running; generation shares GPU memory.")
        outputs: list[str] = []

        def line(text: str) -> None:
            if text.endswith(_AUDIO_EXT) and Path(text).is_file():
                outputs.append(text)
            self._append(text)

        def done(code: int) -> None:
            self._set_busy(False)
            self._append(f"[exit {code}]")
            if code == 0 and outputs:
                self.last_audio = outputs[-1]
                self.play_btn.enabled = True
                self.on_play(None)

        self._set_busy(True)
        cmd = generate_cmd(self.s, caption, float(self.duration_in.value or 30))
        self.task = Job(cmd, self.loop, line, done)

    def on_play(self, _: toga.Widget | None, **kwargs: object) -> None:
        if self.last_audio:
            subprocess.Popen(open_file_cmd(self.last_audio))

    def _on_exit(self, app: toga.App, **kwargs: object) -> bool:
        for job in (self.node, self.task):
            if job is not None:
                job.stop()
        return True


def main() -> FedLoRAApp:
    return FedLoRAApp("FedLoRA Music", "com.fedlora.music")


def run() -> None:
    main().main_loop()
