from prometheus_client import start_http_server
from prometheus_client import Gauge
import psutil
import time

# =========================
# SYSTEM METRICS
# =========================

CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU Usage'
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System Memory Usage'
)

DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'System Disk Usage'
)

# =========================

def collect_metrics():

    CPU_USAGE.set(
        psutil.cpu_percent()
    )

    MEMORY_USAGE.set(
        psutil.virtual_memory().percent
    )

    DISK_USAGE.set(
        psutil.disk_usage('/').percent
    )

if __name__ == "__main__":

    # exporter port
    start_http_server(8000)

    print(
        "Prometheus Exporter Running on Port 8000"
    )

    while True:

        collect_metrics()

        time.sleep(5)