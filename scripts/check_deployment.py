"""Wait for the backend to report the approved model version."""

import argparse
import time

import requests


def check_deployment(url, version):
    if not url:
        raise SystemExit("Set the BACKEND_HEALTH_URL repository variable")

    for attempt in range(1, 61):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            health = response.json()

            if (
                health.get("model_loaded") is True
                and str(health.get("version")) == str(version)
            ):
                print(f"Backend is serving model version {version}")
                return
        except (requests.RequestException, ValueError):
            # The Space can be unavailable while its Docker image rebuilds.
            pass

        print(f"Waiting for model version {version} ({attempt}/60)")
        if attempt < 60:
            time.sleep(10)

    raise SystemExit("Backend did not start the approved model version")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Backend /health URL")
    parser.add_argument("--version", required=True, help="Approved model version")
    args = parser.parse_args()
    check_deployment(args.url, args.version)
