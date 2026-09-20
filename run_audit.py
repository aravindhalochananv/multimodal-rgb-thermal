"""Run the dataset audit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.audit import DatasetAudit

if __name__ == "__main__":
    audit = DatasetAudit()
    results = audit.run_full_audit()
