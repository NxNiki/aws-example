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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import awswrangler as wr
import boto3
import pandas as pd
import paramiko
import pyarrow as pa
import pyarrow.parquet as pq
import redshift_connector

from bituslabs_ds.s3_utils import parse_s3_path, read_local_cache, read_to_pandas_df, save_local_cache

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ssh_pkey = os.environ["BASTION_KEY_PATH"]


# ---------------- Port Forwarding Helpers ----------------
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


def _forward_tunnel(local_port, remote_host, remote_port, transport):
    """
    Listens on a local port and forwards connections to the remote host
    via the Paramiko SSH transport. Runs indefinitely in a daemon thread.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", local_port))
        sock.listen(5)

        while True:
            conn, addr = sock.accept()
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


# ---------------- Safe Athena Query Builder ----------------
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
        if isinstance(value, (datetime.date, datetime.datetime)):
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


# ---------------- Backend Base ----------------
class DatabaseBackend(ABC):
    """Abstract base class for all database backends."""

    @abstractmethod
    def execute(self, query, params=None):
        """Executes a query and returns raw results (list of rows/dicts)."""
        pass

    @abstractmethod
    def query_to_df(self, query, params=None):
        """Executes a query and returns a pandas DataFrame."""
        pass

    @abstractmethod
    def close(self):
        """Closes the connection and cleans up resources."""
        pass


# ---------------- Redshift Backend ----------------
class RedshiftBackend(DatabaseBackend):

    WRITE_KEYWORDS = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE)\b", re.IGNORECASE)

    def __init__(
        self,
        host,
        database,
        user,
        password,
        port=5439,
        bastion_ip="13.215.212.244",
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

    def _check_query(self, query):
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
        if self.conn is None:
            # 1. Establish SSH connection
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
                ),
            )
            self.tunnel_thread.daemon = True
            self.tunnel_thread.start()
            time.sleep(1)

            # 3. Connect DB via Tunnel
            try:
                self.conn = redshift_connector.connect(
                    host="127.0.0.1",
                    port=self.local_port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    ssl=True,
                )
            except Exception as e:
                print(f"Error connecting to Redshift: {e}")
                raise

        return self.conn

    def execute(self, query, params=None):

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

    def query_to_df(self, query, params=None):

        self._check_query(query)
        conn = self.connect()

        if params:
            # redshift_connector/wrangler doesn't support params with pandas directly easily
            raise NotImplementedError("Parameterized queries not supported for pandas in this backend")

        # We pass the existing tunnel connection 'con'
        return wr.redshift.read_sql_query(query, con=conn)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None


# ---------------- Athena Backend ----------------
class AthenaBackend(DatabaseBackend):
    def __init__(self, database, output_location, region="us-west-2", ctas_approach=False):
        self.database = database
        self.output_location = output_location
        self.session = boto3.Session(region_name=region)
        self.ctas_approach = ctas_approach

    def execute(self, query, params=None):

        df = self.query_to_df(query, params)
        return df.to_dict("records")

    def query_to_df(self, query, params=None):

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

    def close(self):
        pass  # Athena is stateless


# ---------------- Unified DataLoader  ----------------
class DataLoader:
    def __init__(self, backend: DatabaseBackend):
        self.backend = backend

    def execute(self, query, params=None):
        return self.backend.execute(query, params)

    def query_to_df(self, query, local_cache: Optional[str] = None, reload: bool = False, params=None):

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

    def close(self):
        self.backend.close()


# ---------------- ETL Scheduler   ----------------

# Define the allowed partition levels for type safety
PartitionLevel = Literal["none", "year", "month", "day"]


class ETLScheduler:
    """
    A scheduler to manage incremental ETL jobs with partitioned Parquet storage.
    Supports dynamic watermark detection and customizable lookback windows.
    """

    def __init__(self, data_loader: DataLoader, storage_root: Union[str, Path], lookback_days: int = 3):
        self.loader = data_loader
        self.storage_root = Path(storage_root)
        self.lookback_days = lookback_days
        self.default_start_date = "2025-01-01"

        # Ensure storage root exists
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _get_partition_cols(self, level: PartitionLevel) -> List[str]:
        """Maps partition level to actual column names."""
        mapping = {"none": [], "year": ["year"], "month": ["year", "month"], "day": ["year", "month", "day"]}
        return mapping.get(level, ["year", "month"])

    def _get_max_date(self, job_path: Path, date_col: str) -> Optional[datetime]:
        """
        Scans the partitioned Parquet dataset to find the maximum processed date.
        """
        if not job_path.exists() or not any(job_path.iterdir()):
            return None

        try:
            # Read only the necessary column from the metadata to save memory/time
            dataset = pq.ParquetDataset(str(job_path), use_legacy_dataset=False)
            table = dataset.read(columns=[date_col])

            if table.num_rows == 0:
                return None

            max_dt = pd.to_datetime(table.to_pandas()[date_col]).max()
            return max_dt
        except Exception as e:
            logger.warning(f"Could not detect watermark in {job_path}: {e}")
            return None

    def _compact_partitions(self, job_name: str, key_cols: List[str], partition_level: PartitionLevel = "month"):
        """
        Internal housekeeping: Merges files and removes duplicates using _processed_at.
        """
        import pyarrow.dataset as ds  # Import the modern dataset API

        job_path = self.storage_root / job_name
        if not job_path.exists() or not any(job_path.iterdir()):
            return

        logger.info(f"[{job_name}] Starting de-duplicating compaction...")

        try:
            # 1. Load the entire dataset using the modern API
            # This is more robust against the "Must provide schema" error
            dataset = ds.dataset(str(job_path), format="parquet", partitioning="hive")
            table = dataset.to_table()
            df = table.to_pandas()

            if df.empty:
                return

            # 2. De-duplication Logic
            # Ensure sorting columns exist before sorting
            sort_cols = key_cols + ["_processed_at"] if "_processed_at" in df.columns else key_cols
            df = df.sort_values(by=sort_cols, ascending=True)
            df = df.drop_duplicates(subset=key_cols, keep="last")

            # 3. Temporary storage for the "clean" write
            temp_path = job_path.with_suffix(".tmp")
            if temp_path.exists():
                shutil.rmtree(temp_path)

            # 4. Write back using pyarrow table to preserve the original schema
            # We use preserve_index=False to keep the parquet files clean
            clean_table = pa.Table.from_pandas(df, preserve_index=False)

            pq.write_to_dataset(
                clean_table,
                root_path=str(temp_path),
                partition_cols=self._get_partition_cols(partition_level),
                basename_template="compact_part_{i}.parquet",
                existing_data_behavior="overwrite_or_ignore",
            )

            # 5. Atomic Swap
            shutil.rmtree(job_path)
            temp_path.rename(job_path)

            logger.info(f"[{job_name}] Compaction complete. Partitions consolidated and de-duplicated.")
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
        Executes an incremental ETL job.

        Args:
            job_name: The directory name for the specific ETL output.
            query_func: A function that takes a start_date string and returns a SQL query.
            date_col: The column used to determine the watermark (max date).
            lookback: Override for the default class lookback_days.
            partition_cols: Columns to use for Parquet partitioning on disk.
        """
        job_path = self.storage_root / job_name
        days_to_lookback = lookback if lookback is not None else self.lookback_days

        # 1. Detect Watermark
        last_date = self._get_max_date(job_path, date_col)

        if last_date:
            # Shift back to handle late-arriving data
            start_dt_obj = last_date - timedelta(days=days_to_lookback)
            start_date_str = start_dt_obj.strftime("%Y-%m-%d")
            logger.info(f"[{job_name}] Incremental start: {start_date_str} (Lookback: {days_to_lookback}d)")
        else:
            # Initial Load
            start_date_str = self.default_start_date
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

        # 3. Data Preparation & Partitioning
        # Ensure date_col is datetime objects for extraction
        df[date_col] = pd.to_datetime(df[date_col])
        df["_processed_at"] = datetime.now()

        partition_cols = self._get_partition_cols(partition_level)

        if "year" in partition_cols:
            df["year"] = df[date_col].dt.year
        if "month" in partition_cols:
            df["month"] = df[date_col].dt.month
        if "day" in partition_cols:
            df["day"] = df[date_col].dt.day

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
        # Note: 'overwrite_or_ignore' prevents file accumulation within existing partitions
        try:
            df.to_parquet(
                path=str(job_path),
                index=False,
                engine="pyarrow",
                partition_cols=partition_cols,
                existing_data_behavior="overwrite_or_ignore",
            )
            logger.info(f"[{job_name}] Successfully updated partitions. New max date: {df[date_col].max().date()}")
        except Exception as e:
            logger.error(f"[{job_name}] Failed to save parquet data: {e}")

        self._compact_partitions(job_name=job_name, key_cols=key_cols, partition_level=partition_level)


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
