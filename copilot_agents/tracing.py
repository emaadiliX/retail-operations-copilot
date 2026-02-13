"""Trace logging for the multi-agent pipeline. Tracks which agent ran,
what it produced, how long it took, and whether it succeeded or failed."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TraceEntry:
    agent_name: str
    stage: str                              # plan / research / draft / verify / deliver
    status: str = "pending"                 # pending / running / completed / error
    input_preview: str = ""
    output_preview: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "agent": self.agent_name,
            "stage": self.stage,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
            "input_preview": self.input_preview[:200],
            "output_preview": self.output_preview[:300],
            "error": self.error_message,
            "metadata": self.metadata,
        }


class TraceLog:

    def __init__(self):
        self.entries: List[TraceEntry] = []
        self.pipeline_start = 0.0
        self.pipeline_end = 0.0

    def start_pipeline(self):
        self.pipeline_start = time.time()

    def end_pipeline(self):
        self.pipeline_end = time.time()

    def get_total_duration(self):
        if self.pipeline_end and self.pipeline_start:
            return round(self.pipeline_end - self.pipeline_start, 2)
        return 0.0

    def begin(self, agent_name, stage, input_preview=""):
        entry = TraceEntry(
            agent_name=agent_name,
            stage=stage,
            status="running",
            input_preview=input_preview,
            start_time=time.time(),
        )
        self.entries.append(entry)
        return entry

    def complete(self, entry, output_preview="", **extra):
        entry.end_time = time.time()
        entry.duration_seconds = entry.end_time - entry.start_time
        entry.status = "completed"
        entry.output_preview = output_preview
        entry.metadata.update(extra)

    def fail(self, entry, error):
        entry.end_time = time.time()
        entry.duration_seconds = entry.end_time - entry.start_time
        entry.status = "error"
        entry.error_message = error

    def format_for_display(self):
        lines = ["=" * 70, "  AGENT TRACE LOG", "=" * 70]

        for i, entry in enumerate(self.entries, 1):
            if entry.status == "completed":
                icon = "OK"
            elif entry.status == "error":
                icon = "ERR"
            elif entry.status == "running":
                icon = "..."
            else:
                icon = "--"

            lines.append(
                f"\n[{icon}] Step {i}: {entry.agent_name} ({entry.stage})")
            lines.append(f"     Duration : {entry.duration_seconds:.2f}s")

            if entry.input_preview:
                preview = entry.input_preview[:120]
                suffix = "..." if len(entry.input_preview) > 120 else ""
                lines.append(f"     Input    : {preview}{suffix}")

            if entry.output_preview:
                preview = entry.output_preview[:200]
                suffix = "..." if len(entry.output_preview) > 200 else ""
                lines.append(f"     Output   : {preview}{suffix}")

            if entry.error_message:
                lines.append(f"     Error    : {entry.error_message}")

            if entry.metadata:
                for k, v in entry.metadata.items():
                    lines.append(f"     {k}: {v}")

        lines.append(
            f"\nTotal pipeline time: {self.get_total_duration():.2f}s")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_list(self):
        return [e.to_dict() for e in self.entries]
