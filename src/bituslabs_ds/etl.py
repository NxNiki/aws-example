import calendar
import logging
import numbers
import os
import re
import shutil
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Sequence, Union

import awswrangler as wr
import boto3
import pandas as pd
import paramiko
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import redshift_connector

from bituslabs_ds.s3_utils import (
    OutputDir,
    join_output_path,
    normalize_storage_root,
    output_path_as_str,
    read_local_cache,
    save_local_cache,
)

# ---------------- Constants / logging ----------------

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_BASTION_KEY_FILE: Optional[str] = None  # temp file path when using BASTION_KEY_CONTENT


def _get_bastion_key_path() -> Optional[str]:
    """
    Resolve bastion SSH key path. Supports:
    - BASTION_KEY_PATH: path to .pem file (local Docker volume mount)
    - BASTION_KEY_CONTENT: raw PEM from Secrets Manager (ECS); writes to temp file on first use
    """
    global _BASTION_KEY_FILE
    path = os.environ.get("BASTION_KEY_PATH")
    if path and os.path.isfile(path):
        return path
    content = os.environ.get("BASTION_KEY_CONTENT")
    if content:
        if _BASTION_KEY_FILE is None:
            import tempfile

            fd, _BASTION_KEY_FILE = tempfile.mkstemp(suffix=".pem")
            os.write(fd, content.encode() if isinstance(content, str) else content)
            os.close(fd)
            os.chmod(_BASTION_KEY_FILE, 0o600)
        return _BASTION_KEY_FILE
    return None


# ---------------- Port forwarding helpers ----------------
def _shuttle_data(source, destination):
    """Helper to move data between two socket-like objects/channels."""
    try:
        while True:
            data = source.recv(1024)
            if not data:
                break
            destination.sendall(data)
    except Exception:
        pass
    finally:
        if hasattr(source, "close"):
            source.close()
        if hasattr(destination, "close"):
            destination.close()


def _forward_tunnel(local_port, remote_host, remote_port, transport, stop_event):
    """
    Listens on a local port and forwards connections to the remote host
    via the Paramiko SSH transport. Runs indefinitely in a daemon thread.
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Allow the socket to check the stop_event periodically
        sock.bind(("127.0.0.1", local_port))
        sock.listen(5)

        while not stop_event.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue  # Just loop back and check stop_event

            chan = transport.open_channel("direct-tcpip", (remote_host, remote_port), (addr[0], addr[1]))

            if chan is None:
                conn.close()
                continue

            threading.Thread(target=_shuttle_data, args=(conn, chan), daemon=True).start()
            threading.Thread(target=_shuttle_data, args=(chan, conn), daemon=True).start()

    except Exception as e:
        print(f"SSH Tunnel listener failed on port {local_port}: {e}")
        if "sock" in locals() and sock:
            sock.close()


# ---------------- Safe Athena query builder ----------------
class SafeAthenaQuery:
    """Provides SQL injection safe query building for Athena."""

    @staticmethod
    def escape(value):
        if value is None:
            return "NULL"
        if isinstance(value, numbers.Number):
            return str(value)
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (date, datetime)):
            return f"'{value.isoformat()}'"
        if isinstance(value, str):
            safe_value = value.replace("'", "''")
            return f"'{safe_value}'"
        if isinstance(value, (list, tuple)):
            return "(" + ", ".join(SafeAthenaQuery.escape(v) for v in value) + ")"
        raise ValueError(f"Unsupported type: {type(value)}")

    @staticmethod
    def build(query_template, params):
        safe_params = {k: SafeAthenaQuery.escape(v) for k, v in params.items()}
        return query_template.format(**safe_params)


# ---------------- Database backends ----------------
class DatabaseBackend(ABC):
    """Abstract base class for all database backends."""

    @abstractmethod
    def execute(self, query, params=None) -> Sequence[Any]:
        """Executes a query and returns raw results (list of rows/dicts)."""
        raise NotImplementedError

    @abstractmethod
    def query_to_df(self, query, params=None) -> pd.DataFrame:
        """Executes a query and returns a pandas DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes the connection and cleans up resources."""
        raise NotImplementedError


# ---------------- Redshift backend ----------------


class RedshiftBackend(DatabaseBackend):

    WRITE_KEYWORDS = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE)\b", re.IGNORECASE)

    def __init__(
        self,
        host,
        database,
        user,
        password,
        port=5439,
        bastion_ip: Optional[str] = None,
        bastion_user="ubuntu",
        local_port=5433,
    ):
        self.conn = None
        self.ssh = None
        self.tunnel_thread = None

        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port

        self.bastion_ip = bastion_ip
        self.bastion_user = bastion_user
        self.local_port = local_port
        self._stop_tunnel = threading.Event()  #

    def _check_query(self, query):
        """
        check if query contains writing operations.
        """
        # Remove lines that start with '--' (SQL comment) or '#' (Python/hash comment)
        query_lines = [
            line
            for line in query.splitlines()
            if not line.lstrip().startswith("--") and not line.lstrip().startswith("#")
        ]
        stripped_query = "\n".join(query_lines)
        if self.WRITE_KEYWORDS.search(stripped_query):
            raise RuntimeError("RedshiftBackend is read-only. Write queries are not allowed.")

    def connect(self):
        if self.conn is not None:
            return self.conn

        if self.bastion_ip:
            ssh_pkey = _get_bastion_key_path()
            if not ssh_pkey:
                raise RuntimeError(
                    "Bastion key required when using --bastion-ip. Set either BASTION_KEY_PATH (path to .pem) "
                    "or BASTION_KEY_CONTENT (from Secrets Manager). Omit --bastion-ip for direct connect."
                )
            logger.info(f"Establishing SSH Tunnel via {self.bastion_ip}...")

            # 1. Establish SSH connection
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Reset the stop event
            self._stop_tunnel.clear()

            try:
                self.ssh.connect(
                    hostname=self.bastion_ip,
                    username=self.bastion_user,
                    key_filename=ssh_pkey,
                    timeout=10,
                )
            except Exception as e:
                print(f"Error connecting to Bastion host: {e}")
                raise

            # 2. Start Tunnel
            self.tunnel_thread = threading.Thread(
                target=_forward_tunnel,
                args=(
                    self.local_port,
                    self.host,
                    self.port,
                    self.ssh.get_transport(),
                    self._stop_tunnel,
                ),
            )
            self.tunnel_thread.daemon = True
            self.tunnel_thread.start()
            time.sleep(1)

            # 3. Connect DB via Tunnel
            host = "127.0.0.1"
            port = self.local_port
        else:
            host = self.host
            port = self.port

        try:
            logger.info(f"Connecting to Redshift with host: {host}, port: {port}")
            self.conn = redshift_connector.connect(
                host=host,
                port=port,
                database=self.database,
                user=self.user,
                password=self.password,
                ssl=True,
            )
        except Exception as e:
            print(f"Error connecting to Redshift: {e}")
            raise

        return self.conn

    def execute(self, query, params=None) -> Sequence[Any]:

        self._check_query(query)
        conn = self.connect()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if cursor.description:  # If it returns rows
                return cursor.fetchall()
            else:
                return []
        finally:
            cursor.close()

    def query_to_df(self, query, params=None) -> pd.DataFrame:

        self._check_query(query)
        conn = self.connect()

        if params:
            # redshift_connector/wrangler doesn't support params with pandas directly easily
            raise NotImplementedError("Parameterized queries not supported for pandas in this backend")

        # We pass the existing tunnel connection 'con'
        return wr.redshift.read_sql_query(query, con=conn)

    def close(self) -> None:
        self._stop_tunnel.set()  # Tell the thread to stop
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        time.sleep(1)


# ---------------- Athena backend ----------------
class AthenaBackend(DatabaseBackend):
    def __init__(
        self, database, output_location, region: str = "us-west-2", ctas_approach: bool = False, timeout: int = 300
    ):
        self.database = database
        self.output_location = output_location
        self.session = boto3.Session(region_name=region)
        self.ctas_approach = ctas_approach

    def execute(self, query, params=None) -> Sequence[Any]:

        df = self.query_to_df(query, params)
        return df.to_dict("records")

    def query_to_df(self, query, params=None) -> pd.DataFrame:

        if params:
            query = SafeAthenaQuery.build(query, params)

        logger.info("Executing Athena query via awswrangler...")

        # Wrangler handles submission, polling (wait loop), and result fetching automatically
        # ctas_approach=True uses CREATE TABLE AS SELECT to avoid 32MB row size limit
        df = wr.athena.read_sql_query(
            sql=query,
            database=self.database,
            s3_output=self.output_location,
            ctas_approach=self.ctas_approach,
            boto3_session=self.session,
        )
        return df

    def close(self) -> None:
        pass  # Athena is stateless


# ---------------- DataLoader ----------------
class DataLoader:
    def __init__(self, backend: DatabaseBackend):
        self.backend = backend

    def execute(self, query, params=None) -> Sequence[Any]:
        return self.backend.execute(query, params)

    def query_to_df(self, query, local_cache: Optional[str] = None, reload: bool = False, params=None) -> Any:

        logger.info(f"Execute query: \n{query}")

        if local_cache and os.path.exists(local_cache) and not reload:
            logger.info(f"Read local cache file: {local_cache}")
            df = read_local_cache(local_cache_path=local_cache)
        else:
            start_time = time.time()
            df = self.backend.query_to_df(query, params)
            exec_time = time.time() - start_time
            logger.info(f"Query execution time: {exec_time:.3f} seconds")
            if local_cache:
                save_local_cache(df, local_cache_path=local_cache)
        return df

    def close(self) -> None:
        self.backend.close()


# ---------------- ETL scheduler ----------------

# Define the allowed partition levels for type safety
PartitionLevel = Literal["none", "year", "month", "day"]
AggCol = Literal["activity_date", "activity_week", "activity_month"]


def effective_start_date(stats_agg_col: AggCol, start_date: str) -> str:
    """
    Truncate start_date to the start of the aggregation period (day/week/month)
    so incremental jobs always fetch full periods and avoid losing data when
    lookback is shorter than a full week or month.
    """
    dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    if stats_agg_col == "activity_month":
        dt = dt.replace(day=1)
    elif stats_agg_col == "activity_week":
        # Monday as start of week, aligned with DATE_TRUNC('week', ...) usage.
        dt = dt - timedelta(days=dt.weekday())

    return dt.strftime("%Y-%m-%d")


def _partition_intersects_incremental_start(partition_level: PartitionLevel, part: dict[str, str], start: date) -> bool:
    """
    True if a hive partition may contain rows with activity on or after ``start``.
    Used to scope compaction to partitions touched by an incremental window.
    """
    try:
        y = int(part["year"])
    except (KeyError, ValueError):
        return False
    if partition_level == "year":
        return date(y, 12, 31) >= start
    try:
        m = int(part["month"])
    except (KeyError, ValueError):
        return False
    if partition_level == "month":
        last_d = calendar.monthrange(y, m)[1]
        return date(y, m, last_d) >= start
    try:
        d = int(part["day"])
    except (KeyError, ValueError):
        return False
    if partition_level == "day":
        return date(y, m, d) >= start
    return False


def _local_leaf_partition_dirs(
    root: Path, partition_cols: List[str], partition_level: PartitionLevel, start: date
) -> List[Path]:
    """Hive leaf directories under root that intersect the incremental start date."""
    if not partition_cols:
        return [root] if root.exists() else []
    depth = len(partition_cols)
    seen: set[Path] = set()
    for f in root.rglob("*.parquet"):
        rel_parts = f.relative_to(root).parts
        if len(rel_parts) < 2:
            continue
        dir_parts = rel_parts[:-1]
        if len(dir_parts) < depth:
            continue
        leaf = dir_parts[:depth]
        dims: dict[str, str] = {}
        ok = True
        for seg, col in zip(leaf, partition_cols, strict=True):
            if "=" not in seg or not seg.startswith(f"{col}="):
                ok = False
                break
            dims[col] = seg.split("=", 1)[1]
        if not ok:
            continue
        if _partition_intersects_incremental_start(partition_level, dims, start):
            seen.add(root.joinpath(*leaf))
    return sorted(seen)


class ETLScheduler:
    """
    A scheduler to manage incremental ETL jobs with partitioned Parquet storage.
    Supports dynamic watermark detection and customizable lookback windows.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        storage_root: OutputDir,
        lookback_days: int = 3,
        overwrite: bool = False,
        default_start_date: str = "2025-01-01",
    ):
        self.loader = data_loader
        self._is_s3, self._storage_root = normalize_storage_root(storage_root)
        self.lookback_days = lookback_days
        self.default_start_date = default_start_date
        self.overwrite = overwrite

    @property
    def is_s3(self) -> bool:
        """True if storage root is an S3 path."""
        return self._is_s3

    def _job_path(self, job_name: str) -> OutputDir:
        """Return the output path for a job (Path for local, str for S3)."""
        return join_output_path(self._storage_root, job_name)

    def _get_partition_cols(self, level: PartitionLevel) -> List[str]:
        """Maps partition level to actual column names."""
        mapping = {"none": [], "year": ["year"], "month": ["year", "month"], "day": ["year", "month", "day"]}
        return mapping.get(level, ["year", "month"])

    def _dataset_has_incomplete_columns(self, job_path: OutputDir, required_columns) -> bool:
        """
        Return True if any existing Parquet file under job_path is missing one or more of the
        required_columns. This is used to detect mixed schemas (old files without newly added
        metrics) so we can trigger a full reload instead of appending incompatible data.

        On error (e.g. S3 listing/permission, metadata read failure), returns False and logs
        a WARNING. Use --overwrite to force a full reload when schema may have changed.
        """
        try:
            required_set = set(required_columns)

            if self._is_s3:
                path_str = output_path_as_str(job_path)
                if not path_str.endswith("/"):
                    path_str = path_str + "/"
                try:
                    objects = wr.s3.list_objects(path=path_str, suffix=".parquet")
                except Exception as e:
                    logger.warning(
                        f"Schema check: could not list S3 objects at {job_path}: {type(e).__name__}: {e}. "
                        "Skipping schema validation; incremental append will proceed. "
                        "If new columns were added, run with --overwrite to force a full reload."
                    )
                    return False

                # Exclude sibling dirs: "daily_stats/" must not match "daily_stats_pa/..."
                path_str_slash = path_str.rstrip("/") + "/"
                objects = [o for o in objects if o.startswith(path_str_slash)]

                for obj in objects:
                    try:
                        columns_types, _ = wr.s3.read_parquet_metadata(path=obj, dataset=False)
                    except Exception as e:
                        logger.debug(f"Could not read metadata for {obj}: {e}")
                        continue
                    existing = set(columns_types.keys())
                    missing = required_set - existing
                    if missing:
                        logger.info(
                            f"Schema check: file {obj} missing {len(missing)} columns: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}. "
                            "Triggering full reload."
                        )
                        return True

                if len(objects) == 0:
                    logger.info(
                        f"Schema check: no parquet files found under {path_str}; treating as no existing data (will append)."
                    )
                return False

            # Local filesystem
            path = job_path if isinstance(job_path, Path) else Path(job_path)
            if not path.exists():
                return False

            for f in path.rglob("*.parquet"):
                try:
                    schema = pq.read_schema(f)
                except Exception as e:
                    logger.debug(f"Could not read schema for {f}: {e}")
                    continue
                existing = {schema.field(i).name for i in range(schema.num_fields)}
                if not required_set.issubset(existing):
                    return True

            return False

        except Exception as e:
            logger.warning(
                f"Schema completeness check failed at {job_path}: {type(e).__name__}: {e}. "
                "Skipping schema validation; incremental append will proceed. "
                "If new columns were added, run with --overwrite to force a full reload."
            )
            return False

    def _get_max_date(self, job_path: OutputDir, date_col: str) -> Optional[datetime]:
        """
        Scans the partitioned Parquet dataset to find the maximum processed date.
        More robust for mixed schemas on S3: if some old files are missing the
        target date column, we fall back to reading the full schema and only
        use files where the column exists.
        """
        if self._is_s3:
            # S3 list uses prefix matching: "daily_stats" also matches
            # "daily_stats_return_user/...". Use trailing slash so we only
            # read under this job's path.
            path_str = output_path_as_str(job_path)
            if not path_str.endswith("/"):
                path_str = path_str + "/"
            try:
                try:
                    # Fast path: only load the requested date column
                    df = wr.s3.read_parquet(
                        path=path_str,
                        columns=[date_col],
                        dataset=True,
                    )
                except Exception as e:
                    # Fallback: schema evolution / mixed files can break the
                    # column-pruned read. Load full schema instead.
                    logger.warning(
                        f"Could not read watermark column '{date_col}' with column-pruned "
                        f"read in {job_path}: {e}. Falling back to full-schema scan."
                    )
                    df = wr.s3.read_parquet(
                        path=path_str,
                        dataset=True,
                    )
                    if date_col not in df.columns:
                        logger.warning(
                            f"Date column '{date_col}' not found in dataset at {job_path}. "
                            f"Watermark detection will return None."
                        )
                        return None

                if df.empty:
                    return None
                return pd.to_datetime(df[date_col]).max()
            except Exception as e:
                logger.warning(f"Could not detect watermark in {job_path}: {e}")
                return None
        # Local: job_path is Path
        path = job_path if isinstance(job_path, Path) else Path(job_path)
        if not path.exists() or not any(path.iterdir()):
            return None
        try:
            dataset = pq.ParquetDataset(output_path_as_str(path), use_legacy_dataset=False)
            table = dataset.read(columns=[date_col])
            if table.num_rows == 0:
                return None
            max_dt = pd.to_datetime(table.to_pandas()[date_col]).max()
            return max_dt
        except Exception as e:
            logger.warning(f"Could not detect watermark in {job_path}: {e}")
            return None

    def _compact_partitions(
        self,
        job_name: str,
        key_cols: List[str],
        partition_level: PartitionLevel = "month",
        *,
        overwrite: bool,
        start_date_str: str,
        default_start_date: str,
    ) -> None:
        """
        Merge small files and drop duplicate keys (keep latest by _processed_at when present).

        Runs only for incremental jobs: skipped when ``overwrite`` is True or when the job
        used ``default_start_date`` (initial / full-history load). When skipped, historical
        partitions are not scanned.

        For hive-partitioned data, only partitions that can contain rows on or after the
        incremental ``start_date_str`` are read and rewritten (e.g. month-level layout and a
        start date in late March compacts only ``year=…/month=3/`` onward, not older months).
        """
        if overwrite:
            logger.info(f"[{job_name}] Skipping compaction (overwrite mode).")
            return
        if start_date_str == default_start_date:
            logger.info(f"[{job_name}] Skipping compaction (default start date / full window).")
            return

        incremental_start = pd.to_datetime(start_date_str).date()
        partition_cols = self._get_partition_cols(partition_level)
        logger.info(
            f"[{job_name}] Starting scoped de-duplicating compaction from {incremental_start} "
            f"(partition_level={partition_level})..."
        )

        job_path = self._job_path(job_name)
        local_target_dirs: Optional[List[Path]] = None
        try:
            if self._is_s3:
                path_str = output_path_as_str(job_path)
                if not path_str.endswith("/"):
                    path_str = path_str + "/"
                if partition_level == "none":
                    df = wr.s3.read_parquet(path=path_str, dataset=True)
                else:

                    def _pf(part: dict[str, str]) -> bool:
                        return _partition_intersects_incremental_start(partition_level, part, incremental_start)

                    df = wr.s3.read_parquet(path=path_str, dataset=True, partition_filter=_pf)
            else:
                path = job_path if isinstance(job_path, Path) else Path(job_path)
                if not path.exists() or not any(path.iterdir()):
                    return
                if partition_level == "none":
                    dataset = ds.dataset(output_path_as_str(path), format="parquet")
                    df = dataset.to_table().to_pandas()
                else:
                    local_target_dirs = _local_leaf_partition_dirs(
                        path, partition_cols, partition_level, incremental_start
                    )
                    if not local_target_dirs:
                        logger.info(f"[{job_name}] No partitions intersect incremental start; compaction skipped.")
                        return
                    chunks: List[pd.DataFrame] = []
                    for d in local_target_dirs:
                        sub = ds.dataset(output_path_as_str(d), format="parquet")
                        chunks.append(sub.to_table().to_pandas())
                    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

            if df.empty:
                logger.info(f"[{job_name}] Compaction found no rows in selected scope.")
                return

            # De-duplication (full rows within each selected partition set; keys should not span partitions)
            sort_cols = key_cols + ["_processed_at"] if "_processed_at" in df.columns else key_cols
            df = df.sort_values(by=sort_cols, ascending=True)
            df = df.drop_duplicates(subset=key_cols, keep="last")

            if self._is_s3:
                s3_write_mode: Literal["overwrite", "overwrite_partitions"] = (
                    "overwrite_partitions" if partition_cols else "overwrite"
                )
                wr.s3.to_parquet(
                    df=df,
                    path=output_path_as_str(job_path),
                    dataset=True,
                    partition_cols=partition_cols,
                    mode=s3_write_mode,
                    index=False,
                )
            else:
                path = job_path if isinstance(job_path, Path) else Path(job_path)
                if partition_level == "none":
                    temp_path = path.with_suffix(".tmp")
                    if temp_path.exists():
                        shutil.rmtree(temp_path)
                    clean_table = pa.Table.from_pandas(df, preserve_index=False)
                    pq.write_to_dataset(
                        clean_table,
                        root_path=output_path_as_str(temp_path),
                        partition_cols=partition_cols,
                        basename_template="compact_part_{i}.parquet",
                        existing_data_behavior="overwrite_or_ignore",
                    )
                    shutil.rmtree(path)
                    temp_path.rename(path)
                else:
                    if local_target_dirs is None:
                        raise RuntimeError("local_target_dirs unset during hive compaction")
                    for d in local_target_dirs:
                        if d.exists():
                            shutil.rmtree(d)
                    clean_table = pa.Table.from_pandas(df, preserve_index=False)
                    pq.write_to_dataset(
                        clean_table,
                        root_path=output_path_as_str(path),
                        partition_cols=partition_cols,
                        basename_template="compact_part_{i}.parquet",
                        existing_data_behavior="overwrite_or_ignore",
                    )

            logger.info(f"[{job_name}] Scoped compaction complete (partitions consolidated and de-duplicated).")
        except Exception as e:
            logger.error(f"[{job_name}] Compaction failed: {e}")

    def run_incremental_job(
        self,
        job_name: str,
        query_func: Callable[[str], str],
        key_cols: List[str],
        date_col: str = "activity_date",
        lookback: Optional[int] = None,
        partition_level: PartitionLevel = "month",
    ) -> None:
        """
        Executes an incremental ETL job. We assume if lookback days >= 30, the query get stats by month. So will truncate
        start date to frist day of month to avoid partial calculation of the month. Similar logic is applied to weekly
        stats if lookback >= 7.

        Args:
            job_name: The directory name for the specific ETL output.
            query_func: A function that takes a start_date string and returns a SQL query.
            date_col: The column used to determine the watermark (max date).
            lookback: Override for the default class lookback_days.
            partition_cols: Columns to use for Parquet partitioning on disk.
        """
        job_path = self._job_path(job_name)
        days_to_lookback = lookback if lookback is not None else self.lookback_days

        # 1. Detect Watermark (skip if overwrite mode)
        last_date = None if self.overwrite else self._get_max_date(job_path, date_col)

        if last_date:
            # Shift back to handle late-arriving data
            start_dt_obj = last_date - timedelta(days=days_to_lookback)
            if days_to_lookback >= 30:
                # For monthly stats, always start from the first day of the month
                start_dt_obj = start_dt_obj.replace(day=1)
                logger.info(f"[{job_name}] truncate query start date to the first day of month: {start_dt_obj}")
            elif days_to_lookback >= 7:
                # For weekly stats, always start from the first day of the week (Monday)
                start_dt_obj = start_dt_obj - timedelta(days=start_dt_obj.weekday())
                logger.info(f"[{job_name}] truncate query start date to the first day of week: {start_dt_obj}")

            start_date_str = start_dt_obj.strftime("%Y-%m-%d")
            logger.info(f"[{job_name}] Incremental start: {start_date_str} (Lookback: {days_to_lookback}d)")
        else:
            # Initial Load (or overwrite mode)
            start_date_str = self.default_start_date
            if self.overwrite:
                logger.info(f"[{job_name}] Overwrite mode: full reload from {start_date_str}")
            else:
                logger.info(f"[{job_name}] No existing data found. Starting full load from {start_date_str}")

        # 2. Fetch Data via the provided Loader
        try:
            sql = query_func(start_date_str)
            df = self.loader.query_to_df(query=sql)
        except Exception as e:
            logger.error(f"[{job_name}] Failed to fetch data from database: {e}")
            return

        if df is None or df.empty:
            logger.info(f"[{job_name}] No new records to process.")
            return

        partition_cols = self._get_partition_cols(partition_level)
        write_mode: Literal["append", "overwrite"] = "overwrite" if self.overwrite else "append"

        # 2b. Schema change detection (append mode): if any existing file is missing
        # one of the current query's columns, do a full reload and overwrite.
        if write_mode == "append":
            required_columns = set(df.columns)
            has_incomplete = self._dataset_has_incomplete_columns(job_path, required_columns)
            if has_incomplete:
                logger.info(
                    f"[{job_name}] Schema change detected (e.g. new columns in query). "
                    "Doing full reload to keep dataset consistent."
                )
                start_date_str = self.default_start_date
                try:
                    sql = query_func(start_date_str)
                    df = self.loader.query_to_df(query=sql)
                except Exception as e:
                    logger.error(f"[{job_name}] Full reload fetch failed: {e}")
                    return
                if df is None or df.empty:
                    logger.warning(f"[{job_name}] Full reload returned no data.")
                    return
                write_mode = "overwrite"
            else:
                logger.info(
                    f"[{job_name}] Schema check passed (existing data has all {len(required_columns)} columns)."
                )

        # 3. Data Preparation & Partitioning
        # Ensure date_col is datetime objects for extraction
        df[date_col] = pd.to_datetime(df[date_col])

        # Add the bookkeeping + partition columns in a single concat rather than
        # one assignment at a time: repeated df[col]=... inserts fragment a wide
        # frame (these datasets have ~120 columns) and trigger pandas'
        # PerformanceWarning. concat also returns a de-fragmented frame.
        new_cols: dict[str, Any] = {"_processed_at": datetime.now()}
        if "year" in partition_cols:
            new_cols["year"] = df[date_col].dt.year
        if "month" in partition_cols:
            new_cols["month"] = df[date_col].dt.month
        if "day" in partition_cols:
            new_cols["day"] = df[date_col].dt.day
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # --- Type Conversion and Error Handling ---
        for col in df.columns:
            # We check for 'object' because SQL Decimals arrive in Pandas as Objects
            if df[col].dtype == "object":
                try:
                    # pd.to_numeric handles Decimals, Strings, and Integers efficiently
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception as e:
                    # We use 'errors=raise' above to catch the specific column that fails
                    # Then we log a warning but keep 'object' to prevent the whole job from crashing
                    logger.warning(
                        f"[{job_name}] Column '{col}' could not be converted to numeric. "
                        f"Type remains 'object'. Error: {e}"
                    )
                    # Optional: Print the first few values to see what the problem is
                    # logger.debug(f"Sample values for {col}: {df[col].head(3).tolist()}")

        # 4. Atomic Write with Partitioning
        if self._is_s3:
            wr.s3.to_parquet(
                df=df,
                path=output_path_as_str(job_path),
                dataset=True,
                partition_cols=partition_cols,
                mode=write_mode,
                index=False,
            )
        else:
            try:
                if write_mode == "overwrite":
                    path = job_path if isinstance(job_path, Path) else Path(job_path)
                    if path.exists():
                        shutil.rmtree(path)
                        logger.info(f"[{job_name}] Cleared existing output for overwrite.")
                df.to_parquet(
                    path=output_path_as_str(job_path),
                    index=False,
                    engine="pyarrow",
                    partition_cols=partition_cols,
                    existing_data_behavior="overwrite_or_ignore",
                )
                logger.info(f"[{job_name}] Successfully updated partitions. New max date: {df[date_col].max().date()}")
            except Exception as e:
                logger.error(f"[{job_name}] Failed to save parquet data: {e}")

        self._compact_partitions(
            job_name=job_name,
            key_cols=key_cols,
            partition_level=partition_level,
            overwrite=self.overwrite,
            start_date_str=start_date_str,
            default_start_date=self.default_start_date,
        )


if __name__ == "__main__":

    # ----- Redshift (Bastion Tunnel) -----
    # redshift_loader = DataLoader(
    #     backend=RedshiftBackend(
    #         host="your-host",
    #         database="dev",
    #         user="awsuser",
    #         password="xxxx",
    #     )
    # )

    # ----- Athena -----
    athena_loader = DataLoader(
        backend=AthenaBackend(
            database="my_athena_db",
            output_location="s3://my-athena-result-bucket/",
        )
    )
