#!/usr/bin/env python3
"""Terminal UI for every script in this repo.

Pure standard library (curses) - no dependencies.

Usage:
  ./helpful_scripts_tui.py                     pick a script from the list
  ./helpful_scripts_tui.py --script <folder>   open straight on one script
                                               (what `<script> --tui` does)

Keys:
  ↑/↓ j/k   move               Tab / ←→  switch between scripts and options
  Enter     edit option         space     toggle / next choice
  r         run script          x         stop running script
  d         reset to defaults   PgUp/PgDn scroll log      q  quit
While editing: Enter accept, Esc cancel, Tab complete path, Ctrl+U clear.
File/folder browser: Enter open/pick, ←/Backspace up, type to filter,
  ~ home, space pick folder, Tab type the path by hand, Esc cancel.

Scripts run in the directory the TUI was started from, so their default
./input and ./output folders end up where you expect them.
"""

import curses
import glob
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".heic")
ANSI = re.compile(r"\x1b(\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*\x07|[@-Z\\-_])")


@dataclass
class Option:
    label: str
    flag: str  # "" for a positional argument
    kind: str = "text"  # text | number | bool | choice | file | dir
    default: object = ""
    choices: tuple = ()
    required: bool = False
    show_if: object = None  # callable(values dict) -> bool
    exts: tuple = ()  # file kind: extensions shown in the picker
    value: object = None

    def reset(self):
        self.value = self.default

    def shown(self, values):
        return self.show_if is None or self.show_if(values)

    def display(self):
        if self.kind == "bool":
            return "[x]" if self.value else "[ ]"
        if self.kind == "choice":
            return " ".join(f"‹{c}›" if c == self.value else c for c in self.choices)
        return str(self.value)

    def args(self):
        if self.kind == "bool":
            return [self.flag] if self.value else []
        if str(self.value) == "":
            return []
        return [self.flag, str(self.value)] if self.flag else [str(self.value)]


@dataclass
class Script:
    key: str  # folder name, used by --script
    name: str
    icon: str
    desc: str
    path: str  # relative to the repo root
    options: list = field(default_factory=list)
    runner: str = "bash"  # bash | python | make
    foreground: bool = False  # needs the terminal itself (interactive)
    requires: tuple = ()  # commands the script checks for
    status: str = "idle"  # idle | running | ok | failed
    seconds: float = 0.0

    def __post_init__(self):
        for o in self.options:
            o.reset()

    def values(self):
        return {o.flag or o.label: o.value for o in self.options}

    def visible(self):
        values = self.values()
        return [o for o in self.options if o.shown(values)]

    def args(self):
        return [a for o in self.visible() for a in o.args()]

    def command(self):
        full = os.path.join(REPO_DIR, self.path)
        if self.runner == "make":
            return ["make", "-C", full, "tui"]
        if self.runner == "python":
            return [sys.executable, full] + self.args()
        return ["bash", full] + self.args()

    def missing(self):
        return [c for c in self.requires if shutil.which(c) is None]


def build_scripts():
    labels_on = lambda v: bool(v.get("-l"))  # noqa: E731
    jpg = lambda v: v.get("-f") == "jpg"  # noqa: E731
    return [
        Script(
            "images_diff",
            "Images diff",
            "img",
            "Diff, merged image and gif from two images",
            "images_diff/images_diff.sh",
            [
                Option("First file", "-f1", "file", required=True, exts=IMAGE_EXTS),
                Option("Second file", "-f2", "file", required=True, exts=IMAGE_EXTS),
                Option("Add labels", "-l", "bool", False),
                Option("First title", "-t1", default="OLD", show_if=labels_on),
                Option("Second title", "-t2", default="NEW", show_if=labels_on),
                Option("Border width", "-w", "number", "3"),
                Option("Border color", "-c", default="black"),
                Option("Output path", "-o", "dir", "./output_images"),
                Option("Crop top (px)", "--crop-top", "number", "0"),
                Option("Skip GIF", "-g", "bool", False),
                Option("Skip merged image", "-m", "bool", False),
            ],
            requires=("montage", "ffmpeg"),
        ),
        Script(
            "images_merger",
            "Images merger",
            "img",
            "Merges all images from a folder into one grid",
            "images_merger/images_merger.sh",
            [
                Option("Input folder", "-i", "dir", "./input_images"),
                Option("Output folder", "-o", "dir", "./output_images"),
                Option("Columns", "-n", "number", "4"),
                Option("Border width", "-w", "number", "3"),
                Option("Border color", "-c", default="black"),
            ],
            requires=("montage",),
        ),
        Script(
            "images_resizer",
            "Images resizer",
            "img",
            "Resizes all images from a folder",
            "images_resizer/images_resizer.sh",
            [
                Option("Input folder", "-i", "dir", "./input_images/"),
                Option("Output folder", "-o", "dir", "./output_images"),
                Option("Width", "-w", "number", "400"),
                Option("Height (0 = keep ratio)", "-a", "number", "0"),
                Option("Compress (ImageOptim)", "-c", "bool", False),
            ],
            requires=("mogrify",),
        ),
        Script(
            "images_heic_converter",
            "Images HEIC converter",
            "img",
            "Converts HEIC/HEIF images to PNG or JPG",
            "images_heic_converter/images_heic_converter.sh",
            [
                Option("Input folder", "-i", "dir", "./input_images"),
                Option("Output folder", "-o", "dir", "./output_images"),
                Option("Format", "-f", "choice", "png", ("png", "jpg")),
                Option("JPG quality 1-100", "-q", "number", "90", show_if=jpg),
            ],
            requires=("convert",),
        ),
        Script(
            "images_downloader",
            "Images downloader",
            "img",
            "Downloads random images from picsum.photos to ./output_images",
            "images_downloader/image_downloader.py",
            [
                Option("Number of images", "-n", "number", "5", required=True),
                Option("Width", "-w", "number", "300"),
                Option("Height", "-ht", "number", "300"),
            ],
            runner="python",
        ),
        Script(
            "video_to_gif",
            "Video to gif",
            "vid",
            "Converts every .mov/.mp4 from a folder to a gif",
            "video_to_gif/video_to_gif.sh",
            [
                Option("Input folder", "-i", "dir", "./input_videos"),
                Option("Output folder", "-o", "dir", "./output_images"),
                Option("FPS", "-f", "number", "15"),
            ],
            requires=("ffmpeg",),
        ),
        Script(
            "video_duplicate_finder",
            "Video duplicate finder",
            "vid",
            "Opens its own terminal UI (make tui)",
            "video_duplicate_finder",
            runner="make",
            foreground=True,
            requires=("make",),
        ),
        Script(
            "ios_record_simulator",
            "iOS record simulator",
            "ios",
            "Records the booted simulator (Ctrl+C stops) and makes a gif",
            "ios_record_simulator/ios_record_simulator.sh",
            [
                Option("Output folder", "-o", "dir", "./gifs"),
                Option("FPS", "-f", "number", "15"),
            ],
            foreground=True,
            requires=("xcrun", "ffmpeg"),
        ),
        Script(
            "git_delete_old_branches_local_and_remote",
            "GIT delete old branches",
            "git",
            "Deletes local and remote branches with no recent commits",
            "git_delete_old_branches_local_and_remote/"
            "git_delete_old_branches_local_and_remote.sh",
            [
                Option("Repository folder", "-r", "dir", ".", required=True),
                Option("Older than", "-s", default="6 months ago"),
                Option("Dry run (only list)", "-d", "bool", True),
            ],
            requires=("git",),
        ),
        Script(
            "remove_special_characters",
            "Remove special characters",
            "txt",
            "Copies files with snake_case names, no special characters",
            "remove_special_characters/remove_special_chars.py",
            [
                Option("Input directory", "", "dir", required=True),
                Option("Output directory", "", "dir", "./output", required=True),
            ],
            runner="python",
        ),
    ]


def complete_path(text):
    """Tab completion: the longest common prefix of the matching paths."""
    expanded = os.path.expanduser(text)
    matches = sorted(glob.glob(glob.escape(expanded) + "*"))
    if not matches:
        return text
    prefix = os.path.commonprefix(matches)
    if len(matches) == 1 and os.path.isdir(prefix):
        prefix += "/"
    if text.startswith("~"):
        prefix = "~" + prefix[len(os.path.expanduser("~")) :]
    return prefix if len(prefix) >= len(text) else text


class Picker:
    """File or folder browser shown over the options and log panels."""

    def __init__(self, kind, value, exts=()):
        self.kind = kind  # file | dir
        self.exts = tuple(e.lower() for e in exts)
        self.filter = ""
        self.cur = 0
        self.error = ""
        path = os.path.abspath(os.path.expanduser(value or "."))
        select = ""
        while not os.path.isdir(path):  # start from the closest existing folder
            select = os.path.basename(path)
            parent = os.path.dirname(path)
            if parent == path:
                path = os.getcwd()
                break
            path = parent
        self.cd(path, select)

    def cd(self, path, select=""):
        try:
            names = os.listdir(path)
        except OSError as e:
            self.error = f"✖ {e.strerror}"
            return
        self.path, self.names, self.filter, self.error = path, names, "", ""
        entries = self.entries()
        labels = [e[0] for e in entries]
        self.cur = labels.index(select) if select in labels else 0

    def entries(self):
        """(name, is_dir) rows: action row, "..", folders, then files."""
        rows = []
        if self.kind == "dir" and not self.filter:
            rows.append(("✔ Use this folder", None))
        if not self.filter and os.path.dirname(self.path) != self.path:
            rows.append(("..", True))
        show_hidden = self.filter.startswith(".")
        dirs, files = [], []
        for name in sorted(self.names, key=str.lower):
            if name.startswith(".") and not show_hidden:
                continue
            if self.filter and self.filter.lower() not in name.lower():
                continue
            if os.path.isdir(os.path.join(self.path, name)):
                dirs.append((name, True))
            elif self.kind == "file":
                if not self.exts or name.lower().endswith(self.exts):
                    files.append((name, False))
        return rows + dirs + files

    def result(self, name=None):
        """Path for the option: relative to cwd when inside it."""
        path = os.path.join(self.path, name) if name else self.path
        rel = os.path.relpath(path, os.getcwd())
        if rel == ".":
            return "."
        if not rel.startswith(".."):
            return "./" + rel
        home = os.path.expanduser("~")
        return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


class Runner:
    """Runs a script in a background thread, streaming output lines to a queue."""

    def __init__(self, log):
        self.log = log
        self.proc = None
        self.thread = None
        self.stopped = False

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def run(self, script):
        def work():
            script.status, script.seconds = "running", 0.0
            start = time.time()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                self.proc = subprocess.Popen(
                    script.command(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                    env=env,
                )
                for line in self.proc.stdout:
                    line = ANSI.sub("", line.rstrip("\r\n").split("\r")[-1])
                    self.log.put(("", line.expandtabs(4)))
                rc = self.proc.wait()
            except OSError as e:
                self.log.put(("err", f"✖ {e}"))
                rc = 1
            script.seconds = time.time() - start
            script.status = "ok" if rc == 0 else "failed"
            if self.stopped:
                self.log.put(("warn", "• Stopped"))
            elif rc == 0:
                self.log.put(("ok", f"✔ Done in {script.seconds:.0f}s"))
            else:
                self.log.put(("err", f"✖ Exited with code {rc}"))

        self.stopped = False
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.stopped = True
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                pass


# colour pair ids
C_OK, C_ERR, C_WARN, C_ACCENT, C_TITLE, C_KEY = range(1, 7)
BADGE = {"idle": (" ", 0), "running": ("●", C_WARN), "ok": ("✔", C_OK)}
BADGE["failed"] = ("✖", C_ERR)


class App:
    def __init__(self, scr, scripts, start=None):
        self.scr = scr
        self.scripts = scripts
        self.cur = 0
        self.opt = 0
        self.pane = "scripts"  # scripts | options
        self.edit = None  # text being edited, or None
        self.picker = None  # Picker while choosing a file or folder
        self.last_dir = "."  # folder of the last picked file or folder
        self.log_q = queue.Queue()
        self.lines = []  # (style, text)
        self.scroll = 0
        self.runner = Runner(self.log_q)
        if start:
            self.cur = next(i for i, s in enumerate(scripts) if s.key == start)
            self.pane = "options"
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        for pair, fg, bg in (
            (C_OK, curses.COLOR_GREEN, -1),
            (C_ERR, curses.COLOR_RED, -1),
            (C_WARN, curses.COLOR_YELLOW, -1),
            (C_ACCENT, curses.COLOR_CYAN, -1),
            (C_TITLE, curses.COLOR_BLACK, curses.COLOR_CYAN),
            (C_KEY, curses.COLOR_BLACK, curses.COLOR_WHITE),
        ):
            curses.init_pair(pair, fg, bg)
        scr.timeout(100)
        scr.keypad(True)
        self.styles = {
            "": 0,
            "head": self.col(C_ACCENT, curses.A_BOLD),
            "step": self.col(C_WARN),
            "err": self.col(C_ERR, curses.A_BOLD),
            "warn": self.col(C_WARN),
            "ok": self.col(C_OK, curses.A_BOLD),
        }

    @property
    def script(self):
        return self.scripts[self.cur]

    def col(self, c, extra=0):
        return (curses.color_pair(c) if c else 0) | extra

    def log(self, style, text):
        self.log_q.put((style, text))

    def drain(self):
        while True:
            try:
                self.lines.append(self.log_q.get_nowait())
            except queue.Empty:
                break
        del self.lines[:-5000]

    # drawing ---------------------------------------------------------------

    def put(self, y, x, text, attr=0, width=None):
        h, w = self.scr.getmaxyx()
        limit = w - x - 1 if width is None else min(width, w - x - 1)
        if 0 <= y < h and 0 <= x < w - 1 and limit > 0:
            try:
                self.scr.addnstr(y, x, text, limit, attr)
            except curses.error:
                pass

    def box(self, y, x, bh, bw, title="", focus=False):
        attr = self.col(C_ACCENT, 0 if focus else curses.A_DIM)
        self.put(y, x, "╭" + "─" * (bw - 2) + "╮", attr)
        for r in range(1, bh - 1):
            self.put(y + r, x, "│", attr)
            self.put(y + r, x + bw - 1, "│", attr)
        self.put(y + bh - 1, x, "╰" + "─" * (bw - 2) + "╯", attr)
        if title:
            self.put(y, x + 2, f" {title} ", self.col(C_ACCENT, curses.A_BOLD))

    def draw(self):
        scr = self.scr
        scr.erase()
        h, w = scr.getmaxyx()
        if h < 24 or w < 72:
            self.put(0, 0, "Terminal too small (min 72x24)", self.col(C_ERR))
            scr.refresh()
            return
        spin = SPINNER[int(time.time() * 10) % len(SPINNER)]

        # banner
        self.put(
            0, 0, " ⚙  Helpful scripts".ljust(w - 1), self.col(C_TITLE, curses.A_BOLD)
        )
        right = f"in {os.getcwd()} "
        right = right if len(right) < w - 24 else "…" + right[-(w - 25) :]
        self.put(0, w - 1 - len(right), right, self.col(C_TITLE))

        # scripts panel
        sw = 38
        most = max(len(x.options) for x in self.scripts)
        top = max(len(self.scripts), most + 4) + 2
        top = min(top, h - 8)
        self.box(2, 1, top, sw, "Scripts", self.pane == "scripts")
        for i, s in enumerate(self.scripts[: top - 2]):
            y = 3 + i
            focus = i == self.cur
            sym, colour = BADGE[s.status]
            if s.status == "running":
                sym = spin
            if focus:
                mark = "▌" if self.pane == "scripts" else "›"
                self.put(y, 2, mark, self.col(C_ACCENT, curses.A_BOLD))
            attr = curses.A_BOLD if focus else 0
            self.put(y, 4, s.icon, self.col(C_ACCENT, curses.A_DIM))
            self.put(y, 9, s.name, attr, sw - 12)
            self.put(y, sw - 2, sym, self.col(colour, curses.A_BOLD))

        # options panel
        s = self.script
        ox, ow = sw + 2, w - sw - 3
        self.box(2, ox, top, ow, s.name, self.pane == "options")
        self.put(3, ox + 2, s.desc, curses.A_DIM, ow - 4)
        missing = s.missing()
        if missing:
            note = f"⚠ not installed: {', '.join(missing)}"
            self.put(4, ox + 2, note, self.col(C_WARN), ow - 4)
        visible = s.visible()
        lw = max([len(o.label) for o in visible] + [10]) + 2
        y = 5
        if not visible:
            self.put(y, ox + 2, "No options - press r to run.", curses.A_DIM)
        for i, o in enumerate(visible):
            focus = self.pane == "options" and i == self.opt
            if focus:
                self.put(y, ox + 1, "▌", self.col(C_ACCENT, curses.A_BOLD))
            label = o.label + (" *" if o.required else "")
            self.put(y, ox + 3, label, curses.A_BOLD if focus else curses.A_DIM)
            vx = ox + 3 + lw + 2
            if focus and self.edit is not None:
                text = self.edit + "▏"
                vis = ow - (vx - ox) - 2
                self.put(y, vx, text[-vis:], self.col(C_KEY))
            else:
                attr = self.col(C_ACCENT) if focus else 0
                value = o.display()
                if not value:
                    value = "required" if o.required else "-"
                    attr = self.col(C_ERR if o.required else 0, curses.A_DIM)
                self.put(y, vx, value, attr, ow - (vx - ox) - 2)
            y += 1
        # command preview
        cy = 2 + top - 2
        cmd = " ".join(self.quote(a) for a in self.preview())
        self.put(cy, ox + 2, "$ " + cmd, self.col(C_OK), ow - 4)

        # log panel
        ly = 2 + top
        lh = h - ly - 1
        title = "Log" + (f"  ↑{self.scroll}" if self.scroll else "")
        self.box(ly, 1, lh, w - 2, title)
        avail = lh - 2
        end = len(self.lines) - self.scroll
        for k, (style, text) in enumerate(self.lines[max(0, end - avail) : end]):
            self.put(ly + 1 + k, 3, text, self.styles.get(style, 0), w - 6)
        if not self.lines:
            hint = "Pick a script, Tab to its options, Enter to edit, r to run."
            self.put(ly + 1, 3, hint, curses.A_DIM)

        if self.picker is not None:
            self.draw_picker(ox, 2, ow, h - 3)

        # footer
        x = 1
        state = ("RUNNING", C_WARN) if self.runner.busy else ("READY", C_OK)
        if self.edit is not None:
            state = ("EDIT", C_ACCENT)
        if self.picker is not None:
            state = ("BROWSE", C_ACCENT)
        badge = self.col(state[1], curses.A_REVERSE | curses.A_BOLD)
        self.put(h - 1, x, f" {state[0]} ", badge)
        x += len(state[0]) + 4
        if self.picker is not None:
            keys = (("⏎", "open/choose"), ("←", "up"), ("~", "home"))
            if self.picker.kind == "dir":
                keys += (("space", "choose"),)
            keys += (("tab", "type path"), ("esc", "cancel"))
        elif self.edit is not None:
            keys = (("⏎", "accept"), ("esc", "cancel"), ("tab", "complete"))
        else:
            keys = (("tab", "pane"), ("⏎", "edit"), ("space", "toggle"))
            keys += (("r", "run"), ("x", "stop"), ("d", "defaults"), ("q", "quit"))
        for key, text in keys:
            self.put(h - 1, x, f" {key} ", self.col(C_KEY, curses.A_BOLD))
            self.put(h - 1, x + len(key) + 2, f" {text}", curses.A_DIM)
            x += len(key) + len(text) + 5
        scr.refresh()

    @staticmethod
    def quote(arg):
        return arg if re.fullmatch(r"[\w@%+=:,./-]+", arg) else f"'{arg}'"

    def preview(self):
        s = self.script
        if s.runner == "make":
            return ["make", "-C", s.path, "tui"]
        prefix = ["python3"] if s.runner == "python" else []
        return prefix + ["./" + s.path] + s.args()

    # actions ---------------------------------------------------------------

    def current_option(self):
        visible = self.script.visible()
        if not visible:
            return None
        self.opt = min(self.opt, len(visible) - 1)
        return visible[self.opt]

    def activate(self, step=1, manual=False):
        o = self.current_option()
        if o is None:
            return
        if o.kind == "bool":
            o.value = not o.value
        elif o.kind == "choice":
            i = o.choices.index(o.value)
            o.value = o.choices[(i + step) % len(o.choices)]
        elif o.kind in ("file", "dir") and not manual:
            # an empty option starts where the last pick happened
            start = str(o.value) or self.last_dir
            self.picker = Picker(o.kind, start, o.exts)
        else:
            self.edit = str(o.value)

    def finish_edit(self, accept):
        o = self.current_option()
        if accept and o is not None:
            o.value = self.edit.strip()
        self.edit = None

    def run(self):
        s = self.script
        if self.runner.busy:
            self.log("warn", "⚠ A script is already running (x stops it)")
            return
        for i, o in enumerate(s.visible()):
            if o.required and str(o.value).strip() == "":
                self.log("err", f"✖ {o.label} is required")
                self.pane, self.opt = "options", i
                return
        self.lines.clear()
        self.scroll = 0
        cmd = " ".join(self.quote(a) for a in self.preview())
        self.log("head", f"━━ {s.name} ━━")
        self.log("step", f"$ {cmd}")
        if s.foreground:
            self.run_in_terminal(s, cmd)
        else:
            self.runner.run(s)

    def run_in_terminal(self, s, cmd):
        """Interactive scripts get the real terminal, then the TUI comes back."""
        curses.def_prog_mode()
        curses.endwin()
        print(f"\n━━ {s.name} ━━\n$ {cmd}\n", flush=True)
        start = time.time()
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is for the script
        try:
            proc = subprocess.Popen(s.command(), preexec_fn=restore_sigint)
            rc = proc.wait()
        except OSError as e:
            print(f"✖ {e}")
            rc = 1
        finally:
            signal.signal(signal.SIGINT, old)
        s.seconds = time.time() - start
        s.status = "ok" if rc == 0 else "failed"
        try:
            input(f"\n[exit code {rc}] Press Enter to go back to the TUI...")
        except (EOFError, KeyboardInterrupt):
            pass
        curses.reset_prog_mode()
        self.scr.refresh()
        if rc == 0:
            self.log("ok", f"✔ Done in {s.seconds:.0f}s (ran in the terminal)")
        else:
            self.log("err", f"✖ Exited with code {rc} (ran in the terminal)")

    # input -----------------------------------------------------------------

    def key(self):
        try:
            return self.scr.get_wch()
        except curses.error:
            return None

    def handle_edit(self, k):
        o = self.current_option()
        if k in ("\n", "\r", curses.KEY_ENTER):
            self.finish_edit(True)
        elif k == "\x1b":
            self.finish_edit(False)
        elif k in ("\x7f", "\b", curses.KEY_BACKSPACE):
            self.edit = self.edit[:-1]
        elif k == "\x15":  # Ctrl+U
            self.edit = ""
        elif k == "\t":
            self.edit = complete_path(self.edit)
        elif isinstance(k, str) and k.isprintable():
            if o is not None and o.kind == "number" and not k.isdigit():
                return
            self.edit += k

    def picked(self, value):
        """Store the picked path and move on to the next option."""
        p = self.picker
        self.current_option().value = value
        self.last_dir = p.path
        self.picker = None
        self.opt = min(self.opt + 1, len(self.script.visible()) - 1)

    def handle_picker(self, k):
        p = self.picker
        entries = p.entries()
        if k == "\x1b":
            self.picker = None
        elif k == "\t":  # type the path by hand instead
            self.picker = None
            self.activate(manual=True)
        elif k in (curses.KEY_DOWN,) and entries:
            p.cur = (p.cur + 1) % len(entries)
        elif k in (curses.KEY_UP,) and entries:
            p.cur = (p.cur - 1) % len(entries)
        elif k == curses.KEY_NPAGE and entries:
            p.cur = min(p.cur + 10, len(entries) - 1)
        elif k == curses.KEY_PPAGE:
            p.cur = max(p.cur - 10, 0)
        elif k in (curses.KEY_LEFT,) or (
            k in ("\x7f", "\b", curses.KEY_BACKSPACE) and not p.filter
        ):
            p.cd(os.path.dirname(p.path), os.path.basename(p.path))
        elif k in ("\x7f", "\b", curses.KEY_BACKSPACE):
            p.filter, p.cur = p.filter[:-1], 0
        elif k in ("\n", "\r", curses.KEY_ENTER, curses.KEY_RIGHT) and entries:
            name, is_dir = entries[min(p.cur, len(entries) - 1)]
            if is_dir is None:  # "Use this folder"
                if k != curses.KEY_RIGHT:
                    self.picked(p.result())
            elif is_dir:
                if name == "..":
                    p.cd(os.path.dirname(p.path), os.path.basename(p.path))
                else:
                    p.cd(os.path.join(p.path, name))
            elif k != curses.KEY_RIGHT:
                self.picked(p.result(name))
        elif k == " " and p.kind == "dir" and entries:
            name, is_dir = entries[min(p.cur, len(entries) - 1)]
            if is_dir and name != "..":  # pick the highlighted folder
                self.picked(p.result(name))
        elif k == "~":
            p.cd(os.path.expanduser("~"))
        elif isinstance(k, str) and k.isprintable():
            p.filter, p.cur = p.filter + k, 0

    def draw_picker(self, x, y, bw, bh):
        p = self.picker
        what = "file" if p.kind == "file" else "folder"
        self.box(y, x, bh, bw, f"Choose {what}", True)
        for r in range(1, bh - 1):
            self.put(y + r, x + 1, " " * (bw - 2))
        path = p.path
        if len(path) > bw - 6:
            path = "…" + path[-(bw - 7) :]
        self.put(y + 1, x + 2, path, self.col(C_ACCENT, curses.A_BOLD))
        if p.error:
            info, attr = p.error, self.col(C_ERR)
        elif p.filter:
            info, attr = f"filter: {p.filter}▏", self.col(C_WARN)
        else:
            info = "type to filter"
            if p.exts:
                info += "  ·  " + " ".join(e.lstrip(".") for e in p.exts)
            attr = curses.A_DIM
        self.put(y + 2, x + 2, info, attr, bw - 4)
        entries = p.entries()
        avail = bh - 4
        p.cur = min(p.cur, max(len(entries) - 1, 0))
        first = max(0, min(p.cur - avail // 2, len(entries) - avail))
        if not entries:
            self.put(y + 3, x + 4, "nothing matches", curses.A_DIM)
        for i, (name, is_dir) in enumerate(entries[first : first + avail]):
            row = y + 3 + i
            focus = first + i == p.cur
            if is_dir is None:
                label, attr = name, self.col(C_OK, curses.A_BOLD)
            elif is_dir:
                label, attr = name + "/", self.col(C_ACCENT)
            else:
                label, attr = name, 0
            if focus:
                attr |= curses.A_REVERSE
                self.put(row, x + 2, "▶", self.col(C_ACCENT, curses.A_BOLD))
            self.put(row, x + 4, f" {label} ", attr, bw - 7)
        if len(entries) > avail:
            self.put(y + bh - 1, x + bw - 12, f" {p.cur + 1}/{len(entries)} ")

    def loop(self):
        while True:
            self.drain()
            self.draw()
            k = self.key()
            if k is None or k == curses.KEY_RESIZE:
                continue
            if self.picker is not None:
                self.handle_picker(k)
                continue
            if self.edit is not None:
                self.handle_edit(k)
                continue
            n = len(self.scripts)
            option = self.current_option() if self.pane == "options" else None
            choice = option is not None and option.kind == "choice"
            if k == "q":
                if self.runner.busy:
                    self.runner.stop()
                return
            elif k == curses.KEY_LEFT and choice:
                self.activate(-1)
            elif k == curses.KEY_RIGHT and choice:
                self.activate(1)
            elif k in ("\x1b", curses.KEY_LEFT):
                self.pane = "scripts"
            elif k == curses.KEY_RIGHT:
                self.pane = "options"
            elif k in ("\t", curses.KEY_BTAB):
                self.pane = "options" if self.pane == "scripts" else "scripts"
            elif k in (curses.KEY_DOWN, "j"):
                if self.pane == "scripts":
                    self.cur, self.opt = (self.cur + 1) % n, 0
                elif self.script.visible():
                    self.opt = (self.opt + 1) % len(self.script.visible())
            elif k in (curses.KEY_UP, "k"):
                if self.pane == "scripts":
                    self.cur, self.opt = (self.cur - 1) % n, 0
                elif self.script.visible():
                    self.opt = (self.opt - 1) % len(self.script.visible())
            elif k in ("\n", "\r", curses.KEY_ENTER):
                if self.pane == "scripts":
                    self.pane = "options" if self.script.options else self.pane
                    if not self.script.options:
                        self.run()
                else:
                    self.activate()
            elif k == " " and self.pane == "options":
                self.activate()
            elif k == "r":
                self.run()
            elif k == "x":
                self.runner.stop()
            elif k == "d":
                for o in self.script.options:
                    o.reset()
            elif k == curses.KEY_PPAGE:
                self.scroll = min(self.scroll + 10, max(0, len(self.lines) - 1))
            elif k == curses.KEY_NPAGE:
                self.scroll = max(self.scroll - 10, 0)


def restore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def main():
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return
    scripts = build_scripts()
    start = None
    if "--script" in argv:
        i = argv.index("--script")
        start = argv[i + 1] if i + 1 < len(argv) else ""
        if start not in [s.key for s in scripts]:
            keys = ", ".join(s.key for s in scripts)
            sys.exit(f"Unknown script '{start}'. Choose one of: {keys}")
    if not sys.stdout.isatty():
        sys.exit("The TUI needs an interactive terminal")
    os.environ.setdefault("ESCDELAY", "25")
    curses.wrapper(lambda scr: App(scr, scripts, start).loop())


if __name__ == "__main__":
    main()
