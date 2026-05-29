"""Runner that wires a GameFeatureConfig to Redshift + ETLScheduler.

Use ``FeaturePipelineRunner(cfg).run_from_cli()`` from a per-game job script
under ``jobs/ss0x/``. The runner handles argparse, DataLoader construction,
and the two ``run_incremental_job`` calls (one for the enriched output, one
for the grouped output).
"""

import argparse
import logging

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
from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql.pipeline import compose_enriched_query, compose_grouped_query

logger = logging.getLogger(__name__)


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
