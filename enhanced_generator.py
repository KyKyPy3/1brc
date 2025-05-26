#!/usr/bin/env python3

import random
import datetime
import ipaddress
import argparse
import sys
import time
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Callable

def generate_logs(output_file: str, num_entries: int, batch_size: int = 100000) -> None:
    """Generate log entries with interesting response time distributions."""

    start_time = time.time()
    last_report_time = start_time

    # Sample data for generation
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    method_weights = [0.7, 0.15, 0.1, 0.04, 0.01]  # 70% GET, 15% POST, etc.

    paths = [
        "/api/users", "/api/products", "/api/orders", "/api/login", "/api/logout",
        "/api/profile", "/api/settings", "/api/search", "/api/cart", "/api/checkout",
        "/api/admin", "/api/metrics", "/api/health", "/api/docs", "/api/register",
        "/api/reset-password", "/api/verify", "/api/comments", "/api/reviews", "/api/categories"
    ]

    # Status codes with realistic weights
    status_codes = [200, 201, 204, 400, 401, 403, 404, 500, 503]
    status_weights = [0.75, 0.05, 0.05, 0.03, 0.04, 0.02, 0.03, 0.02, 0.01]  # 75% 200 OK, etc.

    # Generate a set of realistic IP addresses
    print("Generating IP addresses...", end="", flush=True)
    num_ips = 10000  # Number of unique IPs
    ip_addresses = []
    for i in range(num_ips):
        # Generate IPv4 addresses across different ranges
        ip = str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
        ip_addresses.append(ip)

    # Some IPs will be more frequent (e.g., heavy users or bots)
    frequent_ips = random.sample(ip_addresses, 100)
    ip_selection_pool = ip_addresses + frequent_ips * 10  # Add frequent IPs multiple times
    print(" Done")

    # Generate a realistic timestamp base
    start_date = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2023, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
    timestamp_range = (end_date - start_date).total_seconds()

    print(f"Starting generation of {num_entries:,} log entries with enhanced distributions")
    print(f"Output file: {output_file}")

    # Calculate estimated file size
    avg_line_length = 67  # Approximate average length in bytes
    estimated_size_gb = (num_entries * avg_line_length) / (1024**3)
    print(f"Estimated output size: {estimated_size_gb:.2f} GB")

    # Define different response time distributions for endpoints
    # This will create interesting min/avg/max spreads for competition
    endpoint_distributions = {}

    # Create varied distribution functions for different endpoints
    for path in paths:
        dist_type = random.choice([
            "normal",         # Normal distribution
            "lognormal",      # Log-normal (right-skewed)
            "bimodal",        # Bimodal distribution
            "heavytail",      # Heavy-tailed distribution
            "spiky",          # Normal with occasional spikes
            "cyclical",       # Cyclical performance pattern
            "plateaued",      # Plateaued distribution with flat sections
            "sliding_window", # Distribution with shifting window over time
        ])

        # Base parameters - will be adjusted per distribution
        base_mean = random.randint(20, 200)
        base_std = random.randint(10, 100)

        # Distribution-specific parameters
        if dist_type == "normal":
            params = {
                "mean": base_mean,
                "std": base_std,
                "min_val": 1,
                "max_val": base_mean + base_std * 5
            }
        elif dist_type == "lognormal":
            params = {
                "mean": math.log(base_mean),
                "sigma": 0.5 + random.random(),
                "min_val": 1,
                "max_val": base_mean * 10
            }
        elif dist_type == "bimodal":
            params = {
                "mean1": base_mean // 2,
                "std1": base_std // 3,
                "mean2": base_mean * 2,
                "std2": base_std,
                "weight": 0.7,  # Weight of first distribution
                "min_val": 1,
                "max_val": base_mean * 5
            }
        elif dist_type == "heavytail":
            params = {
                "scale": base_mean // 3,
                "shape": 1.5 + random.random() * 2,  # Pareto shape parameter
                "min_val": 1,
                "max_val": base_mean * 20
            }
        elif dist_type == "spiky":
            params = {
                "mean": base_mean,
                "std": base_std // 2,
                "spike_prob": 0.01 + random.random() * 0.03,  # 1-4% chance of spike
                "spike_mult": 5 + random.randint(5, 15),  # 5-20x multiplier
                "min_val": 1,
                "max_val": base_mean * 25
            }
        elif dist_type == "cyclical":
            params = {
                "base_mean": base_mean,
                "amplitude": base_std,
                "period": random.randint(50000, 200000),  # Period in number of requests
                "min_val": 1,
                "max_val": base_mean * 3
            }
        elif dist_type == "plateaued":
            params = {
                "levels": [
                    random.randint(5, 20),
                    random.randint(30, 70),
                    random.randint(100, 200),
                    random.randint(300, 600)
                ],
                "weights": [0.6, 0.25, 0.1, 0.05],
                "variation": base_std // 5,
                "min_val": 1,
                "max_val": 1000
            }
        elif dist_type == "sliding_window":
            params = {
                "min_mean": base_mean // 2,
                "max_mean": base_mean * 2,
                "std": base_std,
                "window_size": random.randint(100000, 300000),  # Size of the sliding window
                "min_val": 1,
                "max_val": base_mean * 5
            }

        endpoint_distributions[path] = {"type": dist_type, "params": params}

    # Global patterns affecting all endpoints
    global_patterns = {
        "daily_cycle": {
            "enabled": random.choice([True, False]),
            "amplitude": random.uniform(0.2, 0.5),  # Affects response times by ±20-50%
            "peak_hour": random.randint(9, 17)      # Peak load hour (9 AM to 5 PM)
        },
        "gradual_drift": {
            "enabled": random.choice([True, False]),
            "direction": random.choice(["increase", "decrease"]),
            "max_change": random.uniform(0.05, 0.2)  # Max 5-20% change over the year
        },
        "service_outages": {
            "enabled": random.choice([True, False]),
            "count": random.randint(2, 5),         # Number of outages
            "duration": random.randint(1000, 5000),  # Duration in entries
            "multiplier": random.uniform(5, 15)    # Response time multiplier during outage
        }
    }

    # Print endpoint distribution information
    print("\nEndpoint response time distribution types:")
    for path, dist in endpoint_distributions.items():
        print(f"{path}: {dist['type']} distribution")

    # Track count per endpoint
    endpoint_counts = defaultdict(int)

    # Track file size for progress
    bytes_written = 0
    report_interval = 0.5  # seconds

    # Outage periods - precompute where outages will occur if enabled
    outage_periods = []
    if global_patterns["service_outages"]["enabled"]:
        outage_count = global_patterns["service_outages"]["count"]
        for _ in range(outage_count):
            start_idx = random.randint(0, num_entries - 1)
            duration = global_patterns["service_outages"]["duration"]
            outage_periods.append((start_idx, start_idx + duration))

    # Open the output file
    with open(output_file, 'w', buffering=8*1024*1024) as f:  # 8MB buffer
        log_batch = []

        for i in range(num_entries):
            # Generate timestamp with increasing order but some randomness
            seconds_offset = (i / num_entries) * timestamp_range
            # Add some jitter for more realistic timestamp gaps
            jitter = random.expovariate(1.0) * 0.1
            timestamp = start_date + datetime.timedelta(seconds=seconds_offset + jitter)
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            # Hour of day for daily cycle
            hour = timestamp.hour

            # Select other fields
            ip = random.choice(ip_selection_pool)
            method = random.choices(methods, weights=method_weights, k=1)[0]
            path = random.choice(paths)
            status = random.choices(status_codes, weights=status_weights, k=1)[0]

            # Count requests per endpoint
            endpoint_counts[path] += 1
            count = endpoint_counts[path]

            # Get distribution for this endpoint
            dist = endpoint_distributions[path]
            dist_type = dist["type"]
            params = dist["params"]

            # Generate base response time
            resp_time = 0

            if dist_type == "normal":
                resp_time = int(random.normalvariate(params["mean"], params["std"]))

            elif dist_type == "lognormal":
                resp_time = int(math.exp(random.normalvariate(params["mean"], params["sigma"])))

            elif dist_type == "bimodal":
                if random.random() < params["weight"]:
                    resp_time = int(random.normalvariate(params["mean1"], params["std1"]))
                else:
                    resp_time = int(random.normalvariate(params["mean2"], params["std2"]))

            elif dist_type == "heavytail":
                # Pareto distribution for heavy tails
                resp_time = int(random.paretovariate(params["shape"]) * params["scale"])

            elif dist_type == "spiky":
                if random.random() < params["spike_prob"]:
                    # Generate a spike
                    resp_time = int(random.normalvariate(params["mean"] * params["spike_mult"],
                                                        params["std"] * 2))
                else:
                    resp_time = int(random.normalvariate(params["mean"], params["std"]))

            elif dist_type == "cyclical":
                # Sinusoidal pattern
                cycle_pos = (count % params["period"]) / params["period"] * 2 * math.pi
                multiplier = 1 + math.sin(cycle_pos) * 0.5  # Oscillates between 0.5 and 1.5
                resp_time = int(params["base_mean"] * multiplier + random.normalvariate(0, params["amplitude"]))

            elif dist_type == "plateaued":
                # Choose from different plateaus with some variation
                level = random.choices(params["levels"], weights=params["weights"], k=1)[0]
                resp_time = int(level + random.normalvariate(0, params["variation"]))

            elif dist_type == "sliding_window":
                # Mean shifts over time
                progress = min(1.0, count / params["window_size"])
                current_mean = params["min_mean"] + progress * (params["max_mean"] - params["min_mean"])
                resp_time = int(random.normalvariate(current_mean, params["std"]))

            # Apply global patterns

            # Daily cycle - busier during work hours
            if global_patterns["daily_cycle"]["enabled"]:
                peak_hour = global_patterns["daily_cycle"]["peak_hour"]
                hour_diff = min(abs(hour - peak_hour), 24 - abs(hour - peak_hour))
                # Maximum effect at peak hour (multiplier = 1 + amplitude)
                # Minimum effect at 12 hours from peak (multiplier = 1 - amplitude)
                daily_effect = 1 + global_patterns["daily_cycle"]["amplitude"] * (1 - hour_diff / 12)
                resp_time = int(resp_time * daily_effect)

            # Gradual drift over the year
            if global_patterns["gradual_drift"]["enabled"]:
                year_progress = i / num_entries
                if global_patterns["gradual_drift"]["direction"] == "increase":
                    drift_effect = 1 + global_patterns["gradual_drift"]["max_change"] * year_progress
                else:
                    drift_effect = 1 - global_patterns["gradual_drift"]["max_change"] * year_progress
                resp_time = int(resp_time * drift_effect)

            # Service outages
            if global_patterns["service_outages"]["enabled"]:
                for start, end in outage_periods:
                    if start <= i < end:
                        resp_time = int(resp_time * global_patterns["service_outages"]["multiplier"])
                        break

            # Error response times usually differ
            if status >= 400:
                if status < 500:  # Client errors usually faster
                    resp_time = max(1, int(resp_time * 0.7))
                else:  # Server errors usually slower
                    resp_time = int(resp_time * 1.3)

            # Apply min/max constraints
            resp_time = max(params["min_val"], min(params["max_val"], resp_time))

            # Format the log entry
            log_entry = f"{timestamp_str} {ip} {method} {path} {status} {resp_time}\n"
            log_batch.append(log_entry)
            bytes_written += len(log_entry)

            # Write batch to file when batch is full
            if len(log_batch) >= batch_size:
                f.writelines(log_batch)
                log_batch = []

                # Show progress
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed = current_time - start_time
                    progress = (i / num_entries) * 100

                    # Calculate speed and ETA
                    entries_per_sec = i / elapsed if elapsed > 0 else 0
                    mb_per_sec = (bytes_written / 1024 / 1024) / elapsed if elapsed > 0 else 0

                    # Calculate remaining time
                    if entries_per_sec > 0:
                        remaining_entries = num_entries - i
                        est_remaining_time = remaining_entries / entries_per_sec
                        eta_min = int(est_remaining_time / 60)
                        eta_sec = int(est_remaining_time % 60)
                        eta_str = f"{eta_min}m {eta_sec}s"
                    else:
                        eta_str = "calculating..."

                    # Print progress on the same line
                    progress_msg = (
                        f"\rProgress: {progress:.2f}% - "
                        f"Entries: {i:,}/{num_entries:,} - "
                        f"Speed: {entries_per_sec:.1f} entries/s ({mb_per_sec:.2f} MB/s) - "
                        f"ETA: {eta_str}"
                    )
                    print(progress_msg, end="", flush=True)
                    last_report_time = current_time

        # Write any remaining entries
        if log_batch:
            f.writelines(log_batch)

    # Get stats of the generated data
    print("\n\nGenerating statistics for endpoints...")
    endpoint_stats = {}

    for path in paths:
        entry_count = endpoint_counts[path]
        distribution = endpoint_distributions[path]

        # For simplicity, estimate the stats based on distribution parameters
        # In real use, we would calculate this from the actual generated values
        if distribution["type"] == "normal":
            params = distribution["params"]
            mean = params["mean"]
            std = params["std"]
            min_val = max(params["min_val"], int(mean - 3 * std))
            max_val = min(params["max_val"], int(mean + 3 * std))
        elif distribution["type"] == "lognormal":
            params = distribution["params"]
            mu = params["mean"]
            sigma = params["sigma"]
            mean = math.exp(mu + sigma**2/2)
            var = (math.exp(sigma**2) - 1) * math.exp(2*mu + sigma**2)
            std = math.sqrt(var)
            min_val = params["min_val"]
            max_val = min(params["max_val"], int(mean + 3 * std))
        elif distribution["type"] == "bimodal":
            params = distribution["params"]
            mean = params["mean1"] * params["weight"] + params["mean2"] * (1 - params["weight"])
            min_val = params["min_val"]
            max_val = params["max_val"]
        elif distribution["type"] == "heavytail":
            params = distribution["params"]
            shape = params["shape"]
            scale = params["scale"]
            if shape > 1:
                mean = scale * shape / (shape - 1)
            else:
                mean = scale * 5  # Arbitrary for shape ≤ 1
            min_val = params["min_val"]
            max_val = params["max_val"]
        else:
            # For other distributions, use a simple approximation
            min_val = params["min_val"]
            max_val = params["max_val"]
            mean = (min_val + max_val) / 2

        endpoint_stats[path] = {
            "count": entry_count,
            "min": min_val,
            "avg": mean,
            "max": max_val,
            "distribution": distribution["type"]
        }

    # Print endpoint statistics
    print("\nEstimated endpoint statistics:")
    print("{:<20} {:<15} {:<10} {:<10} {:<10} {:<15}".format(
        "Endpoint", "Distribution", "Count", "Min", "Avg", "Max"))
    print("-" * 80)

    for path, stats in endpoint_stats.items():
        print("{:<20} {:<15} {:<10} {:<10} {:<10.1f} {:<10}".format(
            path, stats["distribution"], stats["count"], stats["min"], stats["avg"], stats["max"]))

    total_time = time.time() - start_time
    final_size_gb = bytes_written / (1024**3)

    # Print final statistics
    print(f"\nGeneration complete!")
    print(f"Generated {num_entries:,} log entries in {total_time:.2f} seconds")
    print(f"Average speed: {num_entries/total_time:.1f} entries/second")
    print(f"Output file size: {final_size_gb:.2f} GB")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced log data with interesting distributions")
    parser.add_argument("--output", type=str, default="billion_logs.txt",
                        help="Output file path (default: billion_logs.txt)")
    parser.add_argument("--count", type=int, default=1_000_000_000,
                        help="Number of log entries to generate (default: 1 billion)")
    parser.add_argument("--batch", type=int, default=100_000,
                        help="Batch size for writing (default: 100,000)")

    args = parser.parse_args()

    generate_logs(args.output, args.count, args.batch)
