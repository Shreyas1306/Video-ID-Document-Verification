"""
Report Generation Module
========================
Assemble structured verification reports and JSON summaries from pipeline outputs.

Submodules:
    - report_generator:  Build final verification report (dict/JSON/text)
"""

from src.report.report_generator import VerificationReportGenerator

__all__ = [
    "VerificationReportGenerator",
]
