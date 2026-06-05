"""Runner that wires a GameFeatureConfig to Redshift + ETLScheduler.

Use ``FeaturePipelineRunner(cfg).run_from_cli()`` from a per-game job script
under ``jobs/ss0x/``. The runner handles argparse, DataLoader construction,
and the two ``run_incremental_job`` calls (one for the enriched output, one
for the grouped output).
"""

import argparse
import logging
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend
from bituslabs_ds.features.config import SEMANTIC_FIELDS, GameFeatureConfig
from bituslabs_ds.features.sql.pipeline import compose_enriched_query, compose_grouped_query
from bituslabs_ds.s3_utils import read_json_from_s3, write_json_to_s3

logger = logging.getLogger(__name__)

# Sidecar written at the dataset root (the output_prefix dir, parent of
# features_enriched/ and features_grouped/). It records the config that produced
# the dataset so a later run can detect incompatible-semantics drift. The name
# starts with '_' and ends in .json so the cluster loaders' ``\.parquet$`` glob
# never picks it up.
_CONFIG_SIDECAR_NAME = "_feature_config.json"

# S3 error codes that mean "sidecar not written yet" (first run / pre-sidecar
# dataset) rather than a real failure.
_MISSING_OBJECT_CODES = {"NoSuchKey", "404", "NoSuchBucket"}


def _read_config_sidecar(sidecar_path: str) -> dict | None:
    """Return the parsed sidecar dict, or None if it doesn't exist yet.

    Re-raises any S3 error that isn't a missing-object error, so genuine
    permission / connectivity problems aren't silently swallowed.
    """
    try:
        return read_json_from_s3(sidecar_path)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _MISSING_OBJECT_CODES:
            return None
        raise


class FeaturePipelineRunner:
    """Glue between a ``GameFeatureConfig`` and ETLScheduler."""

    def __init__(self, cfg: GameFeatureConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # CLI entry point used by jobs/ss0x/etl_feature_engineer.py
    # ------------------------------------------------------------------
    def run_from_cli(self, argv: list[str] | None = None) -> None:
        args = self._parse_args(argv)
        setup_logging(
            f"{LOCAL_ROOT}/jobs/log",
            log_filename=f"etl_feature_engineer_{self.cfg.game_id.lower()}.log",
        )

        if self.cfg.requires_full_history and not args.overwrite:
            logger.warning(
                "%s: requires_full_history=True but --overwrite was not passed. "
                "Pass --overwrite when the SQL semantics have changed (new columns, "
                "different thresholds, etc.) so the existing S3 dataset is recomputed "
                "from scratch under the new logic.",
                self.cfg.game_id,
            )

        loader = DataLoader(
            backend=RedshiftBackend(
                host=REDSHIFT_HOST,
                database="slot-machine",
                user=get_redshift_user(),
                password=get_redshift_password(),
                port=REDSHIFT_PORT,
                bastion_ip=args.bastion_ip,
            )
        )
        try:
            self.run(loader, overwrite=args.overwrite)
        finally:
            loader.close()

    # ------------------------------------------------------------------
    # Programmatic entry point (used by tests + custom drivers)
    # ------------------------------------------------------------------
    def run(self, loader: DataLoader, *, overwrite: bool = False) -> None:
        storage_root = f"{DEFAULT_ETL_OUTPUT}/jobs/{self.cfg.output_prefix}"
        sidecar_path = f"{storage_root}/{_CONFIG_SIDECAR_NAME}"

        # Fail fast (before any Redshift work) if the existing dataset was built
        # with incompatible semantics and this isn't an --overwrite rebuild.
        self._check_config_drift(sidecar_path, overwrite=overwrite)

        scheduler = ETLScheduler(
            loader,
            storage_root,
            lookback_days=self.cfg.effective_lookback_days(),
            overwrite=overwrite,
        )
        scheduler.default_start_date = self.cfg.date_start

        scheduler.run_incremental_job(
            job_name="features_enriched",
            query_func=lambda sd: compose_enriched_query(self.cfg, sd),
            key_cols=self._enriched_key_cols(),
            date_col="activity_date",
            partition_level="month",
        )

        scheduler.run_incremental_job(
            job_name="features_grouped",
            query_func=lambda sd: compose_grouped_query(self.cfg, sd),
            key_cols=self._grouped_key_cols(),
            date_col="activity_date",
            partition_level="month",
        )

        # Record the config that produced this dataset, for provenance and for
        # the next run's drift check.
        self._write_config_sidecar(sidecar_path)

    # ------------------------------------------------------------------
    # Config sidecar / drift guard
    # ------------------------------------------------------------------
    def _check_config_drift(self, sidecar_path: str, *, overwrite: bool) -> None:
        """Hard-fail if the existing dataset's semantic config differs from the
        current one and we're not doing an --overwrite rebuild.

        No sidecar (first run, or a pre-sidecar dataset) -> no check. Pair the
        initial run of a pre-sidecar dataset with --overwrite to re-seed it.
        """
        existing = _read_config_sidecar(sidecar_path)
        if existing is None:
            return

        prev_cfg = existing.get("config", {})
        diffs = {
            field: {"existing": prev_cfg.get(field), "current": cur_value}
            for field, cur_value in self.cfg.semantic_signature().items()
            if prev_cfg.get(field) != cur_value
        }
        if not diffs:
            return

        if overwrite:
            logger.warning(
                "%s: feature config drift detected, but --overwrite was passed; "
                "rebuilding the dataset from scratch under the new config. Changed: %s",
                self.cfg.game_id,
                diffs,
            )
            return

        raise RuntimeError(
            f"{self.cfg.game_id}: feature config drift vs the existing dataset at {sidecar_path}.\n"
            f"Changed semantic fields (existing -> current): {diffs}\n"
            "Rows already on S3 were produced under different semantics and cannot be "
            "safely appended (e.g. agg_group buckets would mix different bin_size "
            "values). Re-run with --overwrite to rebuild the dataset from scratch."
        )

    def _write_config_sidecar(self, sidecar_path: str) -> None:
        payload = {
            "config": self.cfg.to_dict(),
            "semantic_fields": list(SEMANTIC_FIELDS),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_to_s3(payload, sidecar_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _enriched_key_cols(self) -> list[str]:
        return ["user_id", "ai_group", *self.cfg.partition_cols, "spin_id"]

    def _grouped_key_cols(self) -> list[str]:
        return [
            "user_id",
            "ai_group",
            *self.cfg.partition_cols,
            "session_start_date",
            "session_group",
            "agg_group",
        ]

    def _parse_args(self, argv: list[str] | None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"ETL Feature Engineer for {self.cfg.game_id}",
        )
        parser.add_argument(
            "--bastion-ip",
            type=str,
            default=DEFAULT_BASTION_IP,
            help=f"Bastion IP address for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help=(
                "Overwrite existing S3 output (full reload from date_start). "
                "Use when SQL semantics changed (new columns, threshold changes) "
                "so the dataset is recomputed from scratch."
            ),
        )
        return parser.parse_args(argv)


__all__ = ["FeaturePipelineRunner"]
