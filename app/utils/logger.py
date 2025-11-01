import sys
import os
from loguru import logger

# Create 'log' directory if it doesn't exist
os.makedirs("log", exist_ok=True)

# Remove default handler
logger.remove()

# Define custom log format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Console sink (colored output)
logger.add(
    sys.stderr,
    level="INFO",
    format=log_format,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# File sink (persistent logs)
logger.add(
    "log/app.log",
    level="DEBUG",  # Save detailed logs
    format=log_format,
    rotation="10 MB",  # Rotate after 10 MB
    retention="10 days",  # Keep logs for 10 days
    compression="zip",  # Compress old logs
    enqueue=True,  # Thread/process safe
)