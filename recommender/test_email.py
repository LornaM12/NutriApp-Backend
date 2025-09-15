import smtplib, ssl
from email.message import EmailMessage

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "lornamunanie32@gmail.com"
SMTP_PASSWORD = "isjg zozf kxnd fnyo"

msg = EmailMessage()
msg["From"] = SMTP_USERNAME
msg["To"] = "lornamunanie@gmail.com"
msg["Subject"] = "Test Email"
msg.set_content("Hello! This is a test.")

context = ssl.create_default_context()
with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
    server.starttls(context=context)
    server.login(SMTP_USERNAME, SMTP_PASSWORD)
    server.send_message(msg)

print("✅ Test email sent")
