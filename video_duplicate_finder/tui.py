"""
Terminal UI for video_duplicate_finder.

Usage:
    python tui.py [DIRECTORY]

Keys (results table):
    1 / 2   open (play) file 1 / file 2
    r       reveal both files in Finder / file manager
    d       move one file of the pair to Trash (asks which)
    e       export results to JSON
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

from rich.text import Text
from send2trash import send2trash
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

print(
    "⏳ Loading OpenCV / librosa… (the first start after install can take a minute)",
    flush=True,
)
import video_duplicate_finder as vdf  # noqa: E402

# Console log output would corrupt the TUI screen
logging.root.removeHandler(vdf.stream_handler)
logging.root.addHandler(logging.NullHandler())

METHOD_OPTIONS = [
    ("pHash (recommended)", "phash"),
    ("dHash", "dhash"),
    ("Average hash (fastest)", "average"),
    ("Combined (slowest)", "combined"),
]

PHASE_LABELS = {
    "analyze": "📊 Phase 1/2: analyzing videos",
    "compare": "🔍 Phase 2/2: comparing pairs",
}


def open_path(path: str, reveal: bool = False) -> None:
    """Open a file with the default app (or reveal it in the file manager)"""
    if sys.platform == "darwin":
        cmd = ["open", "-R", path] if reveal else ["open", path]
    elif sys.platform.startswith("win"):
        cmd = ["explorer", "/select,", path] if reveal else ["explorer", path]
    else:
        cmd = ["xdg-open", os.path.dirname(path) if reveal else path]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class TrashScreen(ModalScreen):
    """Ask which file of a duplicate pair should go to the Trash"""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, dup: Dict, keep: int, trashed: Set[str]):
        super().__init__()
        self.dup = dup
        self.keep = keep
        self.trashed = trashed

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Move which file to Trash?", id="dialog-title")
            for n in (1, 2):
                path = self.dup[f"file{n}"]
                mark = "  ⭐ suggested to keep" if self.keep == n else ""
                yield Static(
                    f"[b]File {n}[/b] ({self.dup[f'size{n}_mb']:.1f} MB){mark}\n{path}",
                    classes="dialog-file",
                )
            with Grid(id="dialog-buttons"):
                for n in (1, 2):
                    yield Button(
                        f"🗑  Trash file {n}",
                        id=f"trash{n}",
                        variant="error" if self.keep and self.keep != n else "default",
                        disabled=self.dup[f"file{n}"] in self.trashed,
                    )
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # Focus the button for the file we suggest deleting
        target = {1: "#trash2", 2: "#trash1"}.get(self.keep, "#cancel")
        button = self.query_one(target, Button)
        (button if not button.disabled else self.query_one("#cancel")).focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss({"trash1": 1, "trash2": 2}.get(event.button.id))


class DuplicateFinderApp(App):
    TITLE = "Video Duplicate Finder"

    CSS = """
    #settings {
        height: auto;
        padding: 0 1;
        border: round $primary;
        border-title-color: $primary;
    }
    #settings Horizontal { height: auto; }
    .field { width: 1fr; height: auto; padding-right: 1; }
    .field Label { color: $text-muted; padding-left: 1; }
    #dir-field { width: 3fr; }
    #scan { margin-top: 1; min-width: 14; }
    #status-row { height: 1; margin: 1 1 0 1; }
    #phase { width: 1fr; }
    #progress { width: 50; }
    #summary { margin: 0 1; color: $text-muted; height: 1; }
    #results { height: 1fr; margin: 0 1; }
    #details { height: auto; max-height: 6; margin: 0 1; padding: 0 1;
               border: round $surface-lighten-2; }

    TrashScreen { align: center middle; }
    #dialog { width: 90; max-width: 95%; height: auto; padding: 1 2;
              border: thick $error; background: $surface; }
    #dialog-title { text-style: bold; margin-bottom: 1; }
    .dialog-file { margin-bottom: 1; }
    #dialog-buttons { grid-size: 3; grid-gutter: 1; height: auto; }
    #dialog-buttons Button { width: 100%; }
    """

    BINDINGS = [
        Binding("ctrl+s", "scan", "Scan", priority=True),
        Binding("1", "open(1)", "Play 1"),
        Binding("2", "open(2)", "Play 2"),
        Binding("r", "reveal", "Reveal"),
        Binding("d", "trash", "Trash…"),
        Binding("e", "export", "Export JSON"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, directory: str = ""):
        super().__init__()
        self.initial_directory = directory
        self.duplicates: List[Dict] = []
        self.trashed: Set[str] = set()
        self.scanning = False
        self._last_progress = 0.0

    # ---------- layout ----------

    def compose(self) -> ComposeResult:
        yield Header()
        settings = Vertical(id="settings")
        settings.border_title = "Settings"
        with settings:
            with Horizontal():
                with Vertical(classes="field", id="dir-field"):
                    yield Label("Directory (paste or drag a folder here)")
                    yield Input(
                        value=self.initial_directory,
                        placeholder="~/Movies",
                        id="directory",
                    )
                with Vertical(classes="field"):
                    yield Label("Hash method")
                    yield Select(
                        METHOD_OPTIONS, value="phash", allow_blank=False, id="method"
                    )
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Similarity threshold (0.0-1.0)")
                    yield Input(value="0.85", type="number", id="threshold")
                with Vertical(classes="field"):
                    yield Label("Parallel workers (1-8)")
                    yield Input(value="4", type="integer", id="workers")
                yield Button("▶ Scan", variant="primary", id="scan")
        with Horizontal(id="status-row"):
            yield Label("Ready. Enter a directory and press Scan (ctrl+s).", id="phase")
            yield ProgressBar(id="progress", show_eta=True)
        yield Label("", id="summary")
        yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
        yield Static("", id="details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.add_columns("#", "Similarity", "Keep", "File 1", "MB", "File 2", "MB")
        self.query_one("#details").display = False
        (
            self.query_one("#directory")
            if not self.initial_directory
            else self.query_one("#scan")
        ).focus()

    # ---------- scanning ----------

    @on(Button.Pressed, "#scan")
    @on(Input.Submitted)
    def action_scan(self) -> None:
        if self.scanning:
            self.notify("A scan is already running", severity="warning")
            return

        raw_dir = self.query_one("#directory", Input).value
        if not raw_dir.strip():
            self.notify("Enter a directory to scan", severity="error")
            self.query_one("#directory").focus()
            return
        directory = vdf.clean_path(raw_dir)
        if not os.path.isdir(directory):
            self.notify(f"Not a directory:\n{directory}", severity="error")
            return

        try:
            threshold = float(self.query_one("#threshold", Input).value or 0.85)
        except ValueError:
            threshold = 0.85
        try:
            workers = int(self.query_one("#workers", Input).value or 4)
        except ValueError:
            workers = 4
        threshold = max(0.0, min(1.0, threshold))
        workers = max(1, min(8, workers))
        method = self.query_one("#method", Select).value

        self.query_one("#directory", Input).value = directory
        self.query_one("#threshold", Input).value = str(threshold)
        self.query_one("#workers", Input).value = str(workers)

        self.scanning = True
        self.query_one("#scan", Button).disabled = True
        self.query_one("#results", DataTable).clear()
        self.query_one("#summary", Label).update("")
        self.query_one("#details").display = False
        self.duplicates = []
        self.trashed = set()
        self.run_scan(directory, method, threshold, workers)

    @work(thread=True, exclusive=True)
    def run_scan(
        self, directory: str, method: str, threshold: float, workers: int
    ) -> None:
        count = len(vdf.collect_video_files(directory))
        self.call_from_thread(
            self.query_one("#phase", Label).update,
            f"📁 Found {count} video files",
        )
        started = time.monotonic()
        try:
            duplicates = vdf.find_video_duplicates(
                directory, method, threshold, workers, progress=self._on_progress
            )
        except Exception as e:  # pragma: no cover - shown to the user
            self.call_from_thread(self._scan_failed, str(e))
            return
        self.call_from_thread(
            self._scan_finished, duplicates, count, time.monotonic() - started
        )

    def _on_progress(self, phase: str, done: int, total: int, message: str) -> None:
        # Called from the worker thread, possibly thousands of times - throttle
        now = time.monotonic()
        if done not in (0, total) and now - self._last_progress < 0.1:
            return
        self._last_progress = now
        self.call_from_thread(self._update_progress, phase, done, total, message)

    def _update_progress(self, phase: str, done: int, total: int, message: str):
        label = PHASE_LABELS.get(phase, phase)
        self.query_one("#phase", Label).update(f"{label}  {message}")
        self.query_one("#progress", ProgressBar).update(
            total=max(total, 1), progress=done
        )

    def _scan_failed(self, error: str) -> None:
        self.scanning = False
        self.query_one("#scan", Button).disabled = False
        self.query_one("#phase", Label).update(f"❌ Scan failed: {error}")
        self.notify(error, severity="error", title="Scan failed")

    def _scan_finished(self, duplicates: List[Dict], count: int, elapsed: float):
        self.scanning = False
        self.query_one("#scan", Button).disabled = False
        duplicates.sort(key=lambda d: d["similarity_score"], reverse=True)
        self.duplicates = duplicates
        self.query_one("#phase", Label).update(
            f"✅ Done in {fmt_duration(elapsed)} – analyzed {count} files"
        )
        self._fill_table()
        if duplicates:
            self.query_one("#results", DataTable).focus()
        else:
            self.notify("No duplicates found 🎉", title="Done")

    # ---------- results ----------

    def _file_cell(self, path: str, keep: bool) -> Text:
        name = os.path.basename(path)
        if path in self.trashed:
            return Text(f"🗑 {name}", style="strike dim")
        return Text(name, style="bold green" if keep else "")

    def _fill_table(self) -> None:
        table = self.query_one("#results", DataTable)
        cursor = table.cursor_row
        table.clear()
        savings = 0.0
        for i, dup in enumerate(self.duplicates, 1):
            keep, _ = vdf.suggest_keep(dup)
            if not ({dup["file1"], dup["file2"]} & self.trashed):
                savings += min(dup["size1_mb"], dup["size2_mb"])
            table.add_row(
                str(i),
                f"{dup['similarity_score']:.1%}",
                str(keep) if keep else "?",
                self._file_cell(dup["file1"], keep == 1),
                f"{dup['size1_mb']:.1f}",
                self._file_cell(dup["file2"], keep == 2),
                f"{dup['size2_mb']:.1f}",
                key=str(i - 1),
            )
        if self.duplicates:
            table.move_cursor(row=min(cursor, len(self.duplicates) - 1))
        self.query_one("#summary", Label).update(
            f"🔍 {len(self.duplicates)} duplicate pairs · "
            f"💾 potential savings {savings:.1f} MB · "
            f"🗑 {len(self.trashed)} moved to Trash"
            if self.duplicates
            else ""
        )

    def _current(self) -> Optional[Dict]:
        table = self.query_one("#results", DataTable)
        if not self.duplicates or table.row_count == 0:
            return None
        return self.duplicates[table.cursor_row]

    @on(DataTable.RowHighlighted)
    def show_details(self, event: DataTable.RowHighlighted) -> None:
        dup = self._current()
        details = self.query_one("#details", Static)
        if dup is None:
            details.display = False
            return
        keep, reason = vdf.suggest_keep(dup)
        lines = []
        for n in (1, 2):
            star = " ⭐" if keep == n else ""
            lines.append(
                f"[b]{n}[/b]{star} {dup[f'file{n}']}  "
                f"[dim]{dup[f'size{n}_mb']:.1f} MB · {fmt_duration(dup[f'duration{n}'])}[/dim]"
            )
        lines.append(
            f"[dim]💡 keep file {keep} ({reason})[/dim]"
            if keep
            else f"[dim]💡 {reason}[/dim]"
        )
        details.update("\n".join(lines))
        details.display = True

    def check_action(self, action: str, parameters) -> Optional[bool]:
        # Result actions only make sense when the table has a selection
        if action in ("open", "reveal", "trash", "export"):
            focused_on_input = isinstance(self.focused, Input)
            return bool(self.duplicates) and not focused_on_input
        return True

    def action_open(self, n: int) -> None:
        dup = self._current()
        if dup is None:
            return
        path = dup[f"file{n}"]
        if path in self.trashed or not os.path.exists(path):
            self.notify("File is no longer there", severity="warning")
            return
        open_path(path)

    def action_reveal(self) -> None:
        dup = self._current()
        if dup is None:
            return
        for n in (1, 2):
            if os.path.exists(dup[f"file{n}"]):
                open_path(dup[f"file{n}"], reveal=True)

    def action_trash(self) -> None:
        dup = self._current()
        if dup is None:
            return
        keep, _ = vdf.suggest_keep(dup)

        def done(choice: Optional[int]) -> None:
            if not choice:
                return
            path = dup[f"file{choice}"]
            try:
                send2trash(path)
            except Exception as e:
                self.notify(str(e), severity="error", title="Could not move to Trash")
                return
            self.trashed.add(path)
            self.notify(f"Moved to Trash:\n{os.path.basename(path)}")
            self._fill_table()

        self.push_screen(TrashScreen(dup, keep, self.trashed), done)

    def action_export(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = vdf.clean_path(self.query_one("#directory", Input).value)
        path = os.path.join(directory, f"video_duplicates_{stamp}.json")
        out = []
        for dup in self.duplicates:
            keep, reason = vdf.suggest_keep(dup)
            out.append(
                {
                    **dup,
                    "suggested_keep": keep,
                    "suggestion_reason": reason,
                    "trashed": [
                        p for p in (dup["file1"], dup["file2"]) if p in self.trashed
                    ],
                }
            )
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        except OSError as e:
            self.notify(str(e), severity="error", title="Export failed")
            return
        self.notify(f"Saved {path}", title="Exported")


if __name__ == "__main__":
    DuplicateFinderApp(sys.argv[1] if len(sys.argv) > 1 else "").run()
