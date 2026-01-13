import datetime
import logging
import numbers
import os
import re
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import awswrangler as wr
import boto3
import pandas as pd
import paramiko
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
