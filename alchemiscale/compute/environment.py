"""
:mod:`alchemiscale.compute.environment` --- compute environment capture
=======================================================================

Best-effort capture of the software environment a compute service executes
`Task`\\ s in, for durable execution provenance (issue #106).

We try a sequence of package managers --- ``micromamba``, ``mamba``, ``conda``,
then ``pip`` --- and take the first that yields a usable package listing. Some
information about the execution environment is better than none; a service whose
environment cannot be introspected simply records no environment.

The capture is done once per compute service (the environment is fixed for the
service's lifetime) and copied into durable provenance server-side, deduplicated
so that identical environments across services/claims are stored once.
"""

import datetime
import json
import shutil
import subprocess

# (tool, argv) pairs tried in order; the first that returns a parseable package
# listing wins. conda-family tools and pip both support a JSON listing.
_CAPTURE_COMMANDS: list[tuple[str, list[str]]] = [
    ("micromamba", ["micromamba", "list", "--json"]),
    ("mamba", ["mamba", "list", "--json"]),
    ("conda", ["conda", "list", "--json"]),
    ("pip", ["pip", "list", "--format=json"]),
]


def _parse_packages(payload: str) -> dict[str, str]:
    """Parse a conda/pip ``--json`` listing into a ``{name: version}`` map.

    Both conda-family ``list --json`` and ``pip list --format=json`` emit a JSON
    array of objects carrying ``name`` and ``version`` keys.
    """
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("unexpected package listing shape")
    packages = {}
    for entry in data:
        name = entry.get("name")
        version = entry.get("version")
        if name is not None and version is not None:
            packages[str(name)] = str(version)
    if not packages:
        raise ValueError("no packages parsed from listing")
    return packages


def capture_environment(
    timeout: float = 60.0,
    commands: list[tuple[str, list[str]]] | None = None,
) -> dict | None:
    """Capture the current software environment, best-effort.

    Tries ``micromamba``/``mamba``/``conda``/``pip`` in order and returns the
    first successful listing as::

        {"tool": "conda", "packages": {name: version, ...}, "captured_at": "<iso>"}

    Returns ``None`` if no tool is available or none produces a usable listing
    (a missing tool, a non-zero exit, a timeout, or unparseable output all cause
    a fall-through to the next tool). Never raises.

    Parameters
    ----------
    timeout
        Per-command timeout, in seconds.
    commands
        Override the (tool, argv) sequence; primarily for testing.
    """
    for tool, argv in commands if commands is not None else _CAPTURE_COMMANDS:
        if shutil.which(argv[0]) is None:
            continue
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0 or not proc.stdout:
            continue
        try:
            packages = _parse_packages(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            continue

        return {
            "tool": tool,
            "packages": packages,
            "captured_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }

    return None
