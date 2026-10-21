"""
utils/email.py
--------------
Email delivery service for password reset and account notifications.

Configuration (via environment variables):
  SMTP_HOST       - SMTP server hostname (e.g. smtp.gmail.com)
  SMTP_PORT       - SMTP port, defaults to 587
  SMTP_USERNAME   - SMTP authentication username
  SMTP_PASSWORD   - SMTP authentication password
  EMAIL_FROM      - Sender address shown to recipients
  APP_BASE_URL    - Base URL of the frontend app (for building reset links)

If SMTP_HOST is not set, emails are printed to the console log so the
application keeps working during local development without email setup.
"""

import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("email_service")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@medexplain.ai")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:5500")


def _build_reset_html(reset_link: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background:#f4f7fb; margin:0; padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding: 40px 20px;">
          <table width="580" cellpadding="0" cellspacing="0" style="background:#fff; border-radius:20px; box-shadow:0 8px 30px rgba(26,86,219,0.10); overflow:hidden;">
            <tr><td style="background:linear-gradient(135deg,#1a56db,#0ea5b7); padding:32px 40px; text-align:center;">
              <h1 style="color:#fff; margin:0; font-size:1.6rem;">&#10084; MedExplain AI</h1>
              <p style="color:rgba(255,255,255,0.85); margin:8px 0 0;">Password Reset Request</p>
            </td></tr>
            <tr><td style="padding:40px;">
              <h2 style="color:#1e293b; font-size:1.2rem; margin-top:0;">Reset Your Password</h2>
              <p style="color:#64748b; line-height:1.7;">We received a request to reset the password for your MedExplain AI account.
              Click the button below to choose a new password. This link expires in <strong>30 minutes</strong>.</p>
              <div style="text-align:center; margin:32px 0;">
                <a href="{reset_link}" style="background:linear-gradient(135deg,#1a56db,#0ea5b7); color:#fff; text-decoration:none;
                   padding:14px 36px; border-radius:12px; font-weight:700; font-size:1rem; display:inline-block;">
                  Reset Password
                </a>
              </div>
              <p style="color:#94a3b8; font-size:0.85rem;">Or copy and paste this link into your browser:<br>
                <a href="{reset_link}" style="color:#1a56db; word-break:break-all;">{reset_link}</a>
              </p>
              <hr style="border:none; border-top:1px solid #e2e8f0; margin:28px 0;">
              <p style="color:#94a3b8; font-size:0.8rem; margin:0;">
                If you did not request a password reset, please ignore this email. Your account remains secure.
                <br>Never share this link with anyone.
              </p>
            </td></tr>
          </table>
          <p style="color:#94a3b8; font-size:0.78rem; margin-top:20px;">MedExplain AI &mdash; For educational purposes only. Not medical advice.</p>
        </td></tr>
      </table>
    </body>
    </html>
    """


def _build_changed_html(user_name: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background:#f4f7fb; margin:0; padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding:40px 20px;">
          <table width="580" cellpadding="0" cellspacing="0" style="background:#fff; border-radius:20px; box-shadow:0 8px 30px rgba(26,86,219,0.10); overflow:hidden;">
            <tr><td style="background:linear-gradient(135deg,#1a56db,#0ea5b7); padding:32px 40px; text-align:center;">
              <h1 style="color:#fff; margin:0; font-size:1.6rem;">&#10084; MedExplain AI</h1>
              <p style="color:rgba(255,255,255,0.85); margin:8px 0 0;">Password Changed</p>
            </td></tr>
            <tr><td style="padding:40px;">
              <h2 style="color:#1e293b; font-size:1.2rem; margin-top:0;">Hi {user_name},</h2>
              <p style="color:#64748b; line-height:1.7;">Your MedExplain AI account password was successfully changed.
              If you made this change, no further action is needed.</p>
              <p style="color:#64748b; line-height:1.7;">If you did <strong>not</strong> make this change, please <a href="{APP_BASE_URL}/forgot_password.html" style="color:#1a56db;">reset your password immediately</a> and contact support.</p>
              <hr style="border:none; border-top:1px solid #e2e8f0; margin:28px 0;">
              <p style="color:#94a3b8; font-size:0.8rem; margin:0;">MedExplain AI &mdash; For educational purposes only. Not medical advice.</p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """


async def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    """
    Sends a password-reset email to the given address.
    Falls back to console logging when SMTP is not configured.
    Returns True on success (or console-log fallback), False on delivery failure.
    """
    subject = "Reset your MedExplain AI password"
    html_body = _build_reset_html(reset_link)

    if not SMTP_HOST:
        logger.warning(
            "[EMAIL SERVICE] SMTP not configured. Printing reset link to console.\n"
            "To: %s\nSubject: %s\nReset link: %s",
            to_email, subject, reset_link,
        )
        return True

    return await _send_email(to_email, subject, html_body)


async def send_password_changed_email(to_email: str, user_name: str) -> bool:
    """
    Sends a notification email when a password is successfully changed.
    Falls back to console logging when SMTP is not configured.
    """
    subject = "Your MedExplain AI password was changed"
    html_body = _build_changed_html(user_name)

    if not SMTP_HOST:
        logger.info(
            "[EMAIL SERVICE] SMTP not configured. Password-changed notification for %s suppressed.",
            to_email,
        )
        return True

    return await _send_email(to_email, subject, html_body)


async def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Internal helper: builds MIME message and sends via aiosmtplib."""
    try:
        import aiosmtplib  # imported here so missing package doesn't crash app at startup
    except ImportError:
        logger.error("aiosmtplib not installed. Install it: pip install aiosmtplib")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USERNAME or None,
            password=SMTP_PASSWORD or None,
            start_tls=True,
        )
        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)
        return False
