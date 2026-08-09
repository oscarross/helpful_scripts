# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

This is a **flat collection of independent, standalone scripts** — there is no shared library, build system, or entrypoint. Each top-level directory is one script (mostly `bash`, a couple of `python3`) with its own `README.md` documenting parameters and examples. Changes to one script's directory essentially never require touching another's; treat each directory as its own small project.

Two directories (`ios_xcode_cleaner/`, `ios_unused_swift_files_variables/`) are **git submodules** pointing at third-party repos — do not edit files inside them. After cloning, run `git submodule update --init --recursive` to populate them.

`ios_record_simulator/helpers/video_to_gif.sh` is a **symlink** to `video_to_gif/video_to_gif.sh`, not a copy — editing one edits both.

## Commands

Lint everything (ShellCheck for bash, ruff for python), matching what CI runs on every PR:
```bash
pip install pre-commit
pre-commit install        # one-time, to run on every commit
pre-commit run --all-files
```

Run an individual linter directly:
```bash
shellcheck --severity=warning path/to/script.sh
ruff check path/to/script.py
```

There is no test suite (see the `README.md` disclaimer: "The Scripts are not tested therefore use it on your own risk!"). Verify shell changes with `bash -n <script>` for syntax and a manual run; there is no automated way to exercise them.

## Conventions used by the bash scripts

All scripts (except the two git submodules) start with:
```bash
#!/bin/bash          # or #!/usr/bin/env bash
set -euo pipefail
```
Keep this whenever adding or editing a script — see the git history for why (a mid-pipeline failure being masked by a downstream command that still succeeds, e.g. `git log <bad-ref> | wc -l` silently producing `0` instead of erroring).

Argument parsing follows one of two patterns, pick whichever fits:
- `getopts "h...:"` for scripts whose flags are single dashed letters (most scripts, e.g. `images_merger.sh`, `images_resizer.sh`, `video_to_gif.sh`).
- A custom hand-rolled parser (`images_diff.sh`) when multi-character flags like `-f1`/`-f2`/`-t1`/`-t2` are needed, since `getopts` can't express those.

Scripts generally follow this internal layout: `# Params` (default values as plain vars), `show_help()` (used by `-h`, printed via heredoc), optional `show_install_info_*()` for missing dependencies (checked with `command -v <tool>`), `show_variables()` to echo the effective config, then the parsing loop, then the actual work. Preserve this shape when extending a script rather than introducing a new style.

Lint baseline: ShellCheck is run at `--severity=warning` (not the stricter `style`/`info` levels) — see `.pre-commit-config.yaml`. When a script violates a warning-or-above check, fix the underlying issue (e.g. rewriting a fragile loop, quoting an expansion) rather than suppressing it; there is no repo-wide shellcheck exclusion file, so any real warning should get fixed at the source.

## Conventions used by the python scripts

`ruff.toml` pins the lint rule set explicitly (`select = ["E4", "E7", "E9", "F"]`, i.e. pyflakes + basic syntax/error checks) rather than relying on ruff's evolving default set — this is deliberate so lint results don't change silently across ruff versions. Extend this list intentionally rather than removing the pin.
