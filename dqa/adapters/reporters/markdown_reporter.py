"""
Adapter Reporter que emite el reporte de drift en formato Markdown.
"""

from __future__ import annotations

import io
from typing import Optional, TextIO

from dqa.domain.models import DatasetReport, DriftLevel

_EMOJI = {DriftLevel.STABLE: "🟢", DriftLevel.WARNING: "🟡", DriftLevel.ALERT: "🔴"}


class MarkdownReporter:
    """Genera reportes Markdown."""

    def report(self, result: DatasetReport, output: Optional[TextIO] = None) -> str:
        """
        Genera el reporte de drift en Markdown.

        Args:
            result: DatasetReport con los resultados del análisis.
            output: Stream de escritura opcional. Si se omite, retorna el string.

        Returns:
            El contenido Markdown generado como string.
        """
        buf = output or io.StringIO()
        lines: list[str] = [
            "# DQA Drift Report\n\n",
            f"**Overall status:** {_EMOJI[result.overall_level]} {result.overall_level}\n\n",
        ]

        if result.schema_diff.has_changes:
            lines.append("## Schema Drift\n\n")
            for col in result.schema_diff.added:
                lines.append(f"- **Added:** `{col}`\n")
            for col in result.schema_diff.removed:
                lines.append(f"- **Removed:** `{col}`\n")
            for col, (old, new) in result.schema_diff.type_changed.items():
                lines.append(f"- **Type changed:** `{col}` — `{old}` → `{new}`\n")
            lines.append("\n")

        lines.append("## Column Analysis\n")
        for col in result.columns:
            emoji = _EMOJI[col.worst_level]
            lines.append(f"\n### `{col.name}` — {emoji} {col.worst_level}\n\n")
            lines.append("| Metric | Key | Value |\n|--------|-----|-------|\n")
            for metric_name, analysis in col.results.items():
                for k, v in analysis.details.items():
                    lines.append(f"| {metric_name} | {k} | {v} |\n")

        content = "".join(lines)
        buf.write(content)
        return content
