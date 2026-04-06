"""Email alert notifications for daily automation failures."""

import os
import logging

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check if email alerting is configured via environment variables."""
    return bool(os.environ.get('SMTP_SERVER') and os.environ.get('ALERT_EMAIL_TO'))


def send_failure_alert(message: str, failure_step: str = "", log_file: str = "") -> None:
    """Send email alert on automation failure. No-op if email is not configured."""
    if not is_email_configured():
        logger.info("Email alerts not configured - skipping notification")
        return
    logger.warning(f"[ALERT] Failure in '{failure_step}': {message}")
