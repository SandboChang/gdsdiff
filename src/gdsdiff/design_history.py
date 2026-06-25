from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .exact_oracle import format_twice_area
from .policy import PolicyClassificationResult


@dataclass(frozen=True)
class DesignHistoryEntry:
    title: str
    policy_result: PolicyClassificationResult
    diff_gds_path: str | Path | None = None
    tests_run: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    entry_date: date | None = None

    def render_markdown(self) -> str:
        entry_date = self.entry_date or date.today()
        lines = [
            f"### {entry_date.isoformat()} - {self.title}",
            "",
            f"- Status: {self.policy_result.status.value}.",
            f"- Expected diff area dbu^2: {format_twice_area(self.policy_result.expected_twice_area)}.",
            f"- Unexpected diff area dbu^2: {format_twice_area(self.policy_result.unexpected_twice_area)}.",
        ]
        if self.diff_gds_path is not None:
            lines.append(f"- Diff GDS: `{self.diff_gds_path}`.")
        if self.tests_run:
            lines.append("- Tests run:")
            lines.extend(f"  - `{test}`" for test in self.tests_run)
        if self.notes:
            lines.append("- Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        lines.append("- Layer summary:")
        for layer in self.policy_result.layers:
            lines.append(
                "  - "
                f"{layer.layer.layer}:{layer.layer.datatype} expected="
                f"{format_twice_area(layer.expected_twice_area)} unexpected="
                f"{format_twice_area(layer.unexpected_twice_area)} components="
                f"{len(layer.components)}"
            )
        return "\n".join(lines) + "\n"


def render_design_history_entry(
    title: str,
    policy_result: PolicyClassificationResult,
    *,
    diff_gds_path: str | Path | None = None,
    tests_run: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    entry_date: date | None = None,
) -> str:
    return DesignHistoryEntry(
        title=title,
        policy_result=policy_result,
        diff_gds_path=diff_gds_path,
        tests_run=tests_run,
        notes=notes,
        entry_date=entry_date,
    ).render_markdown()
