"""
Quant Trading Daemon — Production entry point.
Schedules the daily live trading cycle using APScheduler with timezone awareness.
Handles graceful shutdown on SIGTERM/SIGINT.
"""
import logging
import signal

from apscheduler.schedulers.blocking import BlockingScheduler

from config.logging_config import setup_logging
from config.settings import SCHEDULE_TIME, SCHEDULE_TIMEZONE
from live_trader import run_daily_cycle

setup_logging()
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler(timezone=SCHEDULE_TIMEZONE)


def job():
    logger.info("--- Triggering daily live cycle ---")
    try:
        run_daily_cycle()
    except Exception as e:
        logger.error(f"Error in daily cycle: {e}", exc_info=True)
    logger.info("--- Daily live cycle finished ---")


def graceful_shutdown(signum, frame):
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}. Shutting down gracefully...")
    scheduler.shutdown(wait=False)


# Register signal handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Run once immediately on startup
logger.info("Starting Quant Trading Daemon...")
job()

# Schedule daily at market open (ET)
hour, minute = SCHEDULE_TIME.split(":")
scheduler.add_job(job, "cron", hour=int(hour), minute=int(minute))

logger.info(f"Daemon scheduled at {SCHEDULE_TIME} {SCHEDULE_TIMEZONE}. Waiting...")
scheduler.start()
