import datetime
import numbers
import os
import socket
import threading
import time
from abc import ABC, abstractmethod

import boto3
import pandas as pd
import paramiko
import redshift_connector

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
    @abstractmethod
    def execute(self, query, params=None): ...
    @abstractmethod
    def query_to_df(self, query, params=None): ...
    @abstractmethod
    def close(self): ...


# ---------------- Redshift Backend ----------------
class RedshiftBackend(DatabaseBackend):
    def __init__(self, host, database, user, password, port=5439):
        self.conn = None
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.local_port = 5433

    def connect(self):
        # --- 1. Establish SSH connection to Bastion ---
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.ssh.connect(hostname="13.215.212.244", username="ubuntu", key_filename=ssh_pkey, timeout=10)
        except Exception as e:
            print(f"Error connecting to Bastion host: {e}")
            raise

        # --- 2. Start Local Port Forwarding Tunnel in a separate thread (FIX) ---
        # Uses the custom _forward_tunnel function to replace the broken paramiko.forward.forward_tunnel
        self.tunnel_thread = threading.Thread(
            target=_forward_tunnel,
            args=(
                self.local_port,
                self.host,  # Remote Redshift Host
                self.port,  # Remote Redshift Port
                self.ssh.get_transport(),  # SSH transport
            ),
        )
        self.tunnel_thread.daemon = True
        self.tunnel_thread.start()

        # Give the tunnel a moment to establish the port binding
        time.sleep(0.5)

        # --- 3. Connect Redshift to the Local Port ---
        self.conn = redshift_connector.connect(
            # Connect to the local forwarded address
            host="127.0.0.1",
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.local_port,  # Use the locally forwarded port
            ssl=True,
        )

        return self.conn

    def execute(self, query, params=None):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params) if params else cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def query_to_df(self, query, params=None):
        conn = self.connect()
        return pd.read_sql(query, conn, params=params)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


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
        if state != "SUCCEEDED":
            raise RuntimeError(f"Athena query failed: {state}")

        # Fetch results
        result = self.client.get_query_results(QueryExecutionId=execution_id)
        rows = result["ResultSet"]["Rows"]
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

    def query_to_df(self, query, params=None):
        return self.backend.query_to_df(query, params)

    def close(self):
        self.backend.close()


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    # ----- Redshift -----
    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="your-host",
            database="dev",
            user="awsuser",
            password="xxxx",
        )
    )
    df_rs = redshift_loader.query_to_df("SELECT * FROM your_table LIMIT 10;")
    print(df_rs)
    redshift_loader.close()

    # ----- Athena -----
    athena_loader = DataLoader(
        backend=AthenaBackend(
            database="my_athena_db",
            output_location="s3://my-athena-result-bucket/",
        )
    )
    df_ath = athena_loader.query_to_df("SELECT * FROM my_table LIMIT 10;")
    print(df_ath)
