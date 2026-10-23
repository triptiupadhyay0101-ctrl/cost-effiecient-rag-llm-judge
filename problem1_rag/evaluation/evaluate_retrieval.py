import json
import csv
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "evaluation_dataset.json"
RESULTS_PATH = PROJECT_DIR / "evaluation" / "retrieval_results.csv"


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    results = []

    for item in dataset:
        question = item["question"]

        payload = {
            "question": question,
            "top_k": 3
        }

        start = time.perf_counter()

        try:
            response = requests.post(
                f"{BASE_URL}/retrieve",
                json=payload,
                timeout=60
            )
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            data = response.json()

            results.append({
                "question": question,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "response": json.dumps(data, ensure_ascii=False)
            })

            print(f"Question: {question}")
            print(f"Status: {response.status_code}")
            print(f"Latency: {latency_ms:.2f} ms")
            print(f"Response: {data}\n")

        except Exception as error:
            results.append({
                "question": question,
                "status_code": "ERROR",
                "latency_ms": "",
                "response": str(error)
            })
            print(f"Error for '{question}': {error}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "status_code",
                "latency_ms",
                "response"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()