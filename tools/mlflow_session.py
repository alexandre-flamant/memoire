import subprocess
import os
import signal
import time

class MLFlowSession:
    """
    Manages an MLflow tracking server instance using a SQLite database backend.
    """

    def __init__(self, DIRECTORY, PORT, DB_NAME="mlruns.db", WORKERS=4):
        """
        Initialize the MLFlowSession.

        Args:
            DIRECTORY (str): Directory for artifacts and where the SQLite DB will reside.
            PORT (int): Port for the MLflow server.
            DB_NAME (str): SQLite database filename for backend store (default: 'mlruns.db').
            WORKERS (int): Number of Gunicorn worker threads (default: 4).
        """
        self.PORT = PORT
        self.DIRECTORY = os.path.abspath(DIRECTORY)
        self.DB_PATH = os.path.join(self.DIRECTORY, DB_NAME)
        self.WORKERS = WORKERS
        self.server = None

    def start(self):
        """
        Start the MLflow server with a SQLite backend and file-based artifact storage.
        """
        if self.status() == 0:
            print(f"Server is already running at http://127.0.0.1:{self.PORT}")
            return

        try:
            os.makedirs(self.DIRECTORY, exist_ok=True)

            backend_uri = f"sqlite:///{self.DB_PATH}"
            artifact_root = f"file://{self.DIRECTORY}"

            self.server = subprocess.Popen([
                "mlflow", "server",
                "--backend-store-uri", backend_uri,
                "--default-artifact-root", artifact_root,
                "--host", "127.0.0.1",
                "--port", str(self.PORT),
                "--workers", str(self.WORKERS)
            ], preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            time.sleep(10)

            if self.server.poll() is not None:
                stdout, stderr = self.server.communicate()
                print("Failed to start MLflow server.")
                print("Error output:")
                print(stderr.decode().strip())
                self.server = None
                raise Exception("Could not start the server.")

            print(f"MLflow server started at http://127.0.0.1:{self.PORT}")
            print(f"Metadata database: {self.DB_PATH}")
            print(f"Artifacts directory: {self.DIRECTORY}")
            print(f"Workers: {self.WORKERS}")
            print(f"Process group: {os.getpgid(self.server.pid)}")

        except FileNotFoundError:
            print("Error: 'mlflow' command not found. Is MLflow installed and in your PATH?")
            self.server = None
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.server = None

    def terminate(self):
        """
        Gracefully terminate the MLflow server and its child processes.
        """
        if self.status() == -1:
            print("Server not running.")
            return
        os.killpg(os.getpgid(self.server.pid), signal.SIGTERM)
        self.server = None

    def kill(self):
        """
        Forcefully kill the MLflow server and its child processes.
        Should only be used if terminate() fails.
        """
        if self.status() == -1:
            print("Server not running.")
            return
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)
        self.server = None

    def status(self):
        """
        Check if the MLflow server is running.

        Returns:
            int: 0 if running, -1 if not running.
        """
        if self.server is None:
            return -1
        if self.server.poll() is not None:
            self.server = None
            return -1
        return 0

    def url(self):
        """
        Returns the full URL of the MLflow server.

        Returns:
            str: The server URL (e.g., "http://127.0.0.1:5000")
        """
        return f"http://127.0.0.1:{self.PORT}"