"""
ساخت گزارش HTML خوانا (به زبان فارسی، راست‌چین) از سیگنال‌ها و ارسال آن از طریق ایمیل.
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SIGNAL_COLORS = {"LONG": "#16a34a", "SHORT": "#dc2626"}
SIGNAL_LABELS_FA = {"LONG": "لانگ (خرید)", "SHORT": "شورت (فروش)"}


def _fmt(value):
    if value is None:
        return "-"
    try:
        decimals = 2 if abs(value) >= 100 else (3 if abs(value) >= 1 else 6)
        return f"{value:,.{decimals}f}"
    except Exception:  # noqa: BLE001
        return str(value)


def _signal_card(result):
    color = SIGNAL_COLORS.get(result["signal"], "#6b7280")
    label = SIGNAL_LABELS_FA.get(result["signal"], result["signal"])
    reasons_html = "".join(
        f'<li style="margin-bottom:4px;">{reason}</li>' for reason in result.get("reasons", [])
    )

    return f"""
    <div style="border:1px solid #e5e7eb;border-right:6px solid {color};border-radius:8px;
                padding:16px;margin-bottom:16px;background:#ffffff;direction:rtl;text-align:right;
                font-family:Tahoma,Arial,sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="font-size:18px;font-weight:bold;">
          {result['symbol']} <span style="color:#6b7280;font-size:13px;">({result['market']})</span>
        </span>
        <span style="background:{color};color:#fff;padding:4px 12px;border-radius:6px;font-weight:bold;">
          {label}
        </span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:10px;">
        <tr>
          <td style="padding:4px 0;color:#6b7280;">قیمت ورود (Entry)</td>
          <td style="padding:4px 0;font-weight:bold;">{_fmt(result.get('entry'))}</td>
          <td style="padding:4px 0;color:#6b7280;">اطمینان سیگنال</td>
          <td style="padding:4px 0;font-weight:bold;">{result.get('confidence')}%</td>
        </tr>
        <tr>
          <td style="padding:4px 0;color:#6b7280;">حد ضرر (Stop Loss)</td>
          <td style="padding:4px 0;font-weight:bold;color:#dc2626;">{_fmt(result.get('stop_loss'))}</td>
          <td style="padding:4px 0;color:#6b7280;">تایم‌فریم</td>
          <td style="padding:4px 0;font-weight:bold;">{result.get('timeframe')}</td>
        </tr>
        <tr>
          <td style="padding:4px 0;color:#6b7280;">هدف اول (TP1)</td>
          <td style="padding:4px 0;font-weight:bold;color:#16a34a;">
            {_fmt(result.get('take_profit_1'))} (R:R {result.get('risk_reward_1')})
          </td>
          <td style="padding:4px 0;color:#6b7280;">هدف دوم (TP2)</td>
          <td style="padding:4px 0;font-weight:bold;color:#16a34a;">
            {_fmt(result.get('take_profit_2'))} (R:R {result.get('risk_reward_2')})
          </td>
        </tr>
      </table>
      <div style="font-size:13px;color:#374151;">
        <b>دلایل سیگنال:</b>
        <ul style="margin:6px 0 0 0;padding-right:18px;">{reasons_html}</ul>
      </div>
    </div>
    """


def _watch_card(result):
    flags_html = "".join(
        f'<li style="margin-bottom:4px;">{flag}</li>' for flag in result.get("watch_flags", [])
    )
    return f"""
    <div style="border:1px solid #e5e7eb;border-right:6px solid #f59e0b;border-radius:8px;
                padding:14px 16px;margin-bottom:14px;background:#fffbeb;direction:rtl;text-align:right;
                font-family:Tahoma,Arial,sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:16px;font-weight:bold;">
          {result['symbol']} <span style="color:#6b7280;font-size:13px;">({result['market']})</span>
        </span>
        <span style="color:#6b7280;font-size:13px;">قیمت فعلی: {_fmt(result.get('price'))}</span>
      </div>
      <ul style="margin:6px 0 0 0;padding-right:18px;font-size:13px;color:#78350f;">{flags_html}</ul>
    </div>
    """


def build_html_report(results, generated_at_str):
    actionable = [r for r in results if r.get("signal") in ("LONG", "SHORT")]
    actionable.sort(key=lambda r: r["confidence"], reverse=True)

    watchlist = [r for r in results if r.get("watch_flags")]
    watchlist.sort(key=lambda r: len(r.get("watch_flags", [])), reverse=True)

    gold_results = [r for r in actionable if r["market"] == "GOLD"]
    crypto_results = [r for r in actionable if r["market"] == "CRYPTO"]

    body = ""
    if gold_results:
        body += '<h3 style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;">🥇 طلا</h3>'
        body += "".join(_signal_card(r) for r in gold_results)
    if crypto_results:
        body += (
            '<h3 style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;">'
            "💰 ارزهای دیجیتال</h3>"
        )
        body += "".join(_signal_card(r) for r in crypto_results)
    if not actionable:
        body = (
            '<p style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;">'
            "در این بررسی هیچ سیگنال قابل‌اتکایی (لانگ/شورت) یافت نشد. "
            "بازار در وضعیت خنثی یا رنج قرار دارد.</p>"
        )

    watch_body = ""
    if watchlist:
        watch_body = f"""
        <div style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;
                    margin-top:24px;margin-bottom:8px;">
          <h3 style="margin:0 0 4px 0;">🔭 رصد ویژه: احتمال حرکت انفجاری نزدیک</h3>
          <p style="font-size:12px;color:#92400e;margin:0 0 10px 0;">
            این موارد <b>سیگنال معاملاتی نیستند</b> و جهت حرکت (صعود/نزول) هنوز مشخص نیست؛
            صرفا نشانه‌ای از احتمال یک حرکت شارپ نزدیک هستند. فقط برای رصد و آماده‌باش.
          </p>
        </div>
        """
        watch_body += "".join(_watch_card(r) for r in watchlist[:10])

    return f"""
    <html>
    <body style="margin:0;padding:0;background:#f3f4f6;">
      <div style="max-width:640px;margin:0 auto;padding:20px;">
        <div style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;margin-bottom:16px;">
          <h2 style="margin:0;">📊 گزارش سیگنال بازار کریپتو و طلا</h2>
          <p style="color:#6b7280;margin:4px 0 0 0;">زمان تولید گزارش: {generated_at_str}</p>
        </div>
        {body}
        {watch_body}
        <div style="direction:rtl;text-align:right;font-family:Tahoma,Arial,sans-serif;
                    font-size:12px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:12px;margin-top:16px;">
          <p>⚠️ این گزارش صرفاً یک تحلیل خودکار بر پایه اندیکاتورهای تکنیکال و پرایس‌اکشن است
          و توصیه مالی یا سرمایه‌گذاری محسوب نمی‌شود. بازارهای کریپتو و طلا نوسان و ریسک بالایی دارند؛
          پیش از هر تصمیمی تحقیق شخصی (DYOR) انجام دهید و مدیریت سرمایه و ریسک را رعایت کنید.</p>
        </div>
      </div>
    </body>
    </html>
    """


def build_plain_text_report(results):
    """نسخه متنی ساده به‌عنوان fallback برای کلاینت‌های ایمیلی که HTML را نمایش نمی‌دهند."""
    lines = ["گزارش سیگنال بازار کریپتو و طلا", "=" * 40]
    actionable = [r for r in results if r.get("signal") in ("LONG", "SHORT")]

    for r in actionable:
        lines.append(f"{r['symbol']} ({r['market']}) - {r['signal']} - اطمینان {r['confidence']}%")
        lines.append(
            f"  ورود: {r.get('entry')} | حد ضرر: {r.get('stop_loss')} | "
            f"هدف۱: {r.get('take_profit_1')} | هدف۲: {r.get('take_profit_2')}"
        )

    if not actionable:
        lines.append("سیگنال قابل‌اتکایی یافت نشد.")

    watchlist = [r for r in results if r.get("watch_flags")]
    if watchlist:
        lines.append("")
        lines.append("رصد ویژه (احتمال حرکت انفجاری، جهت نامشخص):")
        for r in watchlist[:10]:
            lines.append(f"  {r['symbol']} ({r['market']}) - قیمت: {r.get('price')}")
            for flag in r["watch_flags"]:
                lines.append(f"    - {flag}")

    return "\n".join(lines)


def send_email(subject, html_content, plain_content, smtp_server, smtp_port,
                sender_email, sender_password, recipients):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())
