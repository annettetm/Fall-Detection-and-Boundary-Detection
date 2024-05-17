import os
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# User configuration
sender_email = 'mathew.annette18@gmail.com'

password = 'yceu ncze noqw cjma'



def send_email(filename,receiver_email, subject='BOUNDARY BREACH ALERT!!!!!', alert_message='Please take immediate action!', detection_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')):

    server = smtplib.SMTP('smtp.gmail.com', 587)
    context = ssl.create_default_context()
    server.starttls(context=context)
    server.login(sender_email, password)
    msg = MIMEMultipart()	
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Email text
    email_body = f'''
    {alert_message}
    Detection Time: {detection_time}
    '''
    print("Sending the email...")
    
    img_data = open(filename, 'rb').read()
    text = MIMEText(email_body)
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(filename))
    msg.attach(image)
            
    server.sendmail(sender_email, receiver_email, msg.as_string())
    print('Email sent!')

    print('Closing the server...')
    server.quit()

if __name__ =="__main__":
    filename = r'static\2.png'
    receiver_email=''
    subject = 'Boundary Breach Alert'
    alert_message = 'A boundary breach has occurred. Please take immediate action!'
    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    send_email(filename, receiver_email, subject, alert_message, detection_time)