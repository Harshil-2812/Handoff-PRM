"""Results Logger: Writes incremental rows adhering to the frozen results schema and manages resume state."""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from workstream_c.harness.config import EXTENDED_SCHEMA_COLUMNS, FROZEN_SCHEMA_COLUMNS

logger = logging.getLogger("ResultsLogger")


class ResultsLogger:
    """Incremental CSV writer and resume manager conforming strictly to the frozen results schema."""

    def __init__(self, output_csv_path: Path):
        self.csv_path = Path(output_csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = EXTENDED_SCHEMA_COLUMNS
        self.completed_keys: Set[Tuple[str, str, int, str]] = set()
        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV with headers if it does not exist, or loads completed keys for resuming."""
        if not self.csv_path.exists() or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            logger.info(f"Initialized new results CSV at {self.csv_path}")
        else:
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get("task_id", "")
                        arm = row.get("arm", "")
                        seed_str = row.get("seed", "")
                        htype = row.get("handoff_type", "")
                        if tid and arm and seed_str and htype:
                            try:
                                seed = int(seed_str)
                                self.completed_keys.add((tid, arm, seed, htype))
                            except ValueError:
                                pass
                logger.info(f"Resuming: found {len(self.completed_keys)} already-completed runs in {self.csv_path}")
            except Exception as e:
                logger.warning(f"Error reading existing CSV {self.csv_path}: {e}")

    def is_completed(self, task_id: str, arm: str, seed: int, handoff_type: str) -> bool:
        """Returns True if the (task_id, arm, seed, handoff_type) run has already been recorded."""
        return (task_id, arm, seed, handoff_type) in self.completed_keys

    def log_trial(self, trial_data: Dict[str, Any]):
        """Logs a single trial result incrementally to CSV and memory."""
        # Ensure all frozen columns are present
        row = {}
        for col in self.fieldnames:
            val = trial_data.get(col, "")
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            row[col] = val

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
            f.flush()

        tid = trial_data.get("task_id", "")
        arm = trial_data.get("arm", "")
        seed = int(trial_data.get("seed", 0))
        htype = trial_data.get("handoff_type", "")
        if tid and arm and htype:
            self.completed_keys.add((tid, arm, seed, htype))

