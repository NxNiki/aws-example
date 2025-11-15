import datetime
import numbers
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import boto3
import pandas as pd
import paramiko
import redshift_connector
from sqlalchemy import create_engine, text

from bituslabs_ds.s3_utils import read_local_cache, save_local_cache

ssh_pkey = os.environ["BASTION_KEY_PATH"]

# ---------------- Port Forwarding Helpers ----------------


def _shuttle_data(source, destination):
    """Helper to move data between two socket-like objects/channels."""
    try:
        while True:
            # Receive up to 1024 bytes
            data = source.recv(1024)
            if not data:
                break
            destination.sendall(data)
    except Exception:
        # Expected on connection close
        pass
    finally:
        # Ensure both sides are closed
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
        # Bind to the local port (e.g., 127.0.0.1:5433)
        sock.bind(("127.0.0.1", local_port))
        sock.listen(5)

        while True:
            # Accept a connection from the local machine (e.g., redshift_connector)
            conn, addr = sock.accept()

            # Open a Paramiko channel through the SSH transport to the remote host
            chan = transport.open_channel("direct-tcpip", (remote_host, remote_port), (addr[0], addr[1]))

            if chan is None:
                conn.close()
                continue

            # Start two shuttle threads to move data between the local connection and the Paramiko channel
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
            # Escape single quotes by doubling them
            safe_value = value.replace("'", "''")
            return f"'{safe_value}'"
        if isinstance(value, (list, tuple)):
            # Escape each item and join for IN clauses
            return "(" + ", ".join(SafeAthenaQuery.escape(v) for v in value) + ")"
        raise ValueError(f"Unsupported type: {type(value)}")

    @staticmethod
    def build(query_template, params):
        """
        Builds a safe query by escaping parameters and formatting the template.
        Example: build("SELECT * FROM table WHERE id = {user_id}", {"user_id": 123})
        """
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


# ---------------- Redshift Backend (redshift_connector) ----------------
class RedshiftBackend(DatabaseBackend):
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

    def connect(self):
        if self.conn is None:
            # --- 1. Establish SSH connection to Bastion ---
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

            # --- 2. Start Local Port Forwarding Tunnel ---
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

            # --- 3. Connect using redshift_connector to local forwarded port ---
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
        conn = self.connect()
        if params:
            # redshift_connector does not support parameterized queries with pandas directly
            # so we interpolate safely (or handle manually)
            raise NotImplementedError("Parameterized queries not supported for pandas in this backend")
        return pd.read_sql(query, conn)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None


# ---------------- Athena Backend ----------------
class AthenaBackend(DatabaseBackend):
    def __init__(self, database, output_location, region="us-west-2"):
        self.database = database
        self.output_location = output_location
        self.client = boto3.client("athena", region_name=region)

    def execute(self, query, params=None):
        if params:
            query = SafeAthenaQuery.build(query, params)

        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.output_location},
        )
        execution_id = response["QueryExecutionId"]

        # Wait for completion
        while True:
            status = self.client.get_query_execution(QueryExecutionId=execution_id)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(1)  # Wait 1 second before polling again

        if state != "SUCCEEDED":
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "Unknown reason.")
            raise RuntimeError(f"Athena query failed: {state}. Reason: {reason}")

        # Fetch results
        result = self.client.get_query_results(QueryExecutionId=execution_id)
        rows = result["ResultSet"]["Rows"]

        if len(rows) <= 1:
            return []

        headers = [col["VarCharValue"] for col in rows[0]["Data"]]
        data = [[col.get("VarCharValue") for col in r["Data"]] for r in rows[1:]]
        return [dict(zip(headers, r)) for r in data]

    def query_to_df(self, query, params=None):
        rows = self.execute(query, params)
        return pd.DataFrame(rows)

    def close(self):
        pass  # Athena is stateless


# ---------------- Unified DataLoader ----------------
class DataLoader:
    def __init__(self, backend: DatabaseBackend):
        self.backend = backend

    def execute(self, query, params=None):
        return self.backend.execute(query, params)

    def query_to_df(self, query, local_cache: Optional[str] = None, reload: bool = False, params=None):

        if local_cache and os.path.exists(local_cache) and not reload:
            df = read_local_cache(local_cache_path=local_cache)
        else:
            df = self.backend.query_to_df(query, params)
            if local_cache:
                save_local_cache(df, local_cache_path=local_cache)
        return df

    def close(self):
        self.backend.close()


if __name__ == "__main__":

    # ----- Redshift (Bastion Tunnel) -----
    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="your-host",
            database="dev",
            user="awsuser",
            password="xxxx",
        )
    )

    # ----- Athena -----
    athena_loader = DataLoader(
        backend=AthenaBackend(
            database="my_athena_db",
            output_location="s3://my-athena-result-bucket/",
        )
    )
