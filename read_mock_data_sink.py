import polars as pl
import psutil
import os
import gc
import time


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def print_memory_info(stage):
    """Print memory usage information"""
    memory_gb = get_memory_usage()
    print(f"{stage}: {memory_gb:.2f} GB")


# Initial memory
print("=" * 60)
print("Memory Usage: Reading mock_data_sink")
print("=" * 60)
print_memory_info("Initial memory")

# Read mock_data_sink
print("\n" + "-" * 60)
print("Reading mock_data_sink")
print("-" * 60)
print_memory_info("Before scan_parquet")

df = pl.scan_parquet("mock_data_sink", hive_partitioning=True)
print_memory_info("After scan_parquet (lazy)")

print("Collecting data...")
df = df.collect()
print_memory_info("After collect()")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes: {df.dtypes}")

# Clear the DataFrame and force garbage collection
print("\nClearing DataFrame...")
del df
gc.collect()
time.sleep(5)
print_memory_info("After clearing and GC")

print("\n" + "=" * 60)
print("Complete!")
print("=" * 60)
