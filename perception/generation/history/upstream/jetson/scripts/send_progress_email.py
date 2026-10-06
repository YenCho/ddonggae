import argparse
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


DEFAULT_TO = "jaeyoungi@snu.ac.kr"
DEFAULT_CONFIG = "config/email_smtp.json"
DEFAULT_TIMEOUT = 300


def env_bool(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_message(sender, recipient, subject, body):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


def add_attachments(msg, attachments):
    for attachment in attachments:
        path = Path(attachment)
        if not path.exists() or not path.is_file():
            print(f"skip missing attachment: {path}")
            continue
        data = path.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=path.name,
        )


def load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Missing email config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def config_value(config, key, env_name, default=None):
    value = config.get(key)
    if value is not None:
        return value
    return os.getenv(env_name, default)


def smtp_candidates(host, port, use_tls):
    candidates = [(host, port, use_tls)]
    # Common fallback for providers like Gmail when STARTTLS on 587 is unavailable.
    if use_tls and port != 465:
        candidates.append((host, 465, False))
    return candidates


def deliver_via_smtp(msg, host, port, use_tls):
    context = ssl.create_default_context()
    if use_tls:
        smtp = smtplib.SMTP(timeout=DEFAULT_TIMEOUT)
        smtp._host = host
        smtp.connect(host, port)
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        return smtp
    smtp = smtplib.SMTP_SSL(timeout=DEFAULT_TIMEOUT, context=context)
    smtp._host = host
    smtp.connect(host, port)
    smtp.ehlo()
    return smtp


def send_email(args):
    config = load_config(args.config)
    host = config_value(config, "host", "SMTP_HOST")
    port = int(config_value(config, "port", "SMTP_PORT", 587))
    user = config_value(config, "user", "SMTP_USER")
    password = config_value(config, "password", "SMTP_PASSWORD")
    sender = config_value(config, "from", "SMTP_FROM", user or "")
    use_tls = bool(config.get("tls")) if "tls" in config else env_bool("SMTP_TLS", True)

    missing = [
        name for name, value in {
            "SMTP_HOST": host,
            "SMTP_USER": user,
            "SMTP_PASSWORD": password,
            "SMTP_FROM or SMTP_USER": sender,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit("Missing SMTP settings: " + ", ".join(missing))

    msg = build_message(sender, args.to, args.subject, args.body)
    add_attachments(msg, args.attach)
    if args.dry_run:
        print("DRY RUN: email was not sent")
        print(f"SMTP: {host}:{port} tls={use_tls}")
        print(f"From: {sender}")
        print(f"To: {args.to}")
        print(f"Subject: {args.subject}")
        print(args.body)
        return

    attempts = []
    for candidate_host, candidate_port, candidate_use_tls in smtp_candidates(host, port, use_tls):
        try:
            with deliver_via_smtp(msg, candidate_host, candidate_port, candidate_use_tls) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
            mode = "STARTTLS" if candidate_use_tls else "SSL"
            print(f"sent email to {args.to} via {candidate_host}:{candidate_port} ({mode})")
            return
        except OSError as exc:
            attempts.append(f"{candidate_host}:{candidate_port} tls={candidate_use_tls} -> {type(exc).__name__}: {exc}")
            continue
        except smtplib.SMTPException as exc:
            attempts.append(f"{candidate_host}:{candidate_port} tls={candidate_use_tls} -> {type(exc).__name__}: {exc}")
            continue

    raise SystemExit(
        "Failed to send email. Tried: " + " | ".join(attempts)
    )


def main():
    parser = argparse.ArgumentParser(description="Send progress notification email through SMTP config/env vars.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--to", default=os.getenv("SMTP_TO", DEFAULT_TO))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--attach", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    send_email(args)


if __name__ == "__main__":
    main()
