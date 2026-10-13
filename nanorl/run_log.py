"""Durable local JSONL events; each invocation has its own configuration snapshot."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def config_dict(config):
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in asdict(config).items()}


class RunLog:
    def __init__(self, run_dir: Path, config):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        sessions = run_dir / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        self.config_path = sessions / f"{self.session_id}.json"
        with self.config_path.open("x", encoding="utf-8") as stream:
            json.dump(config_dict(config), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.path = run_dir / "metrics.jsonl"
        # A process killed during write may leave a partial last line. Separate sessions
        # with a newline so subsequent complete records remain independently readable.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
        self.stream = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields):
        record = {"event": event, "session_id": self.session_id,
                  "time": datetime.now(timezone.utc).isoformat(), **fields}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        print(line, flush=True)

    def close(self):
        self.stream.close()
