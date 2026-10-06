"""The 2020-2021 pipeline's last step: fill index.html's placeholders with the date and the table rows in tables/, then upload it, the two press pages and every PNG and GIF in the working directory by FTP, and move each image into archive/.

Retired on 6 October 2026. The page is deployed with scripts/deploy.py over SFTP, and a run of this would send a 2020 template over the live page. It exits before doing anything; the code below it is kept as it last ran, apart from style.
"""
import ftplib
import glob
import io
import itertools
import os
import time

raise SystemExit('upload.py is retired: deploy with python3 scripts/deploy.py')

with open('index.html') as f:
    html = f.read()

dateToday = time.strftime('%Y-%m-%d')

html = html.replace('xxxDATExxx', dateToday)

paths = glob.glob('tables/table*.txt')
# paths.remove('tables/tableAsiaEastExChina.txt')
try:
    paths.remove('tables/tableChina.txt')  # data cannot be trusted
except ValueError:
    pass
paths.remove('tables/tableEU.txt')
paths.remove('tables/tableUnited_States_of_America.txt')

paths = [
    # 'tables/tableChina.txt',
    'tables/tableEU.txt',
    'tables/tableUnited_States_of_America.txt',
    ] + paths

table = ''
for path in paths:
    if 'Honduras' in path:
        continue  # no test data
    with open(path) as f:
        table += f.read()

html = html.replace('xxxTABLEROWSxxx', table)

with open('.password') as f:
    password = f.read().strip()  # one.com FTP password
domain = 'tommycarstensen.com'
ftp = ftplib.FTP('ftp.'+domain)
ftp.login(domain, password)
ftp.cwd('covid19')
print('index.html')
ftp.storbinary('STOR index.html', io.BytesIO(bytes(html, 'utf-8')))
for basename in ('press_denmark.html', 'press_international.html'):
    print(basename)
    with open(basename, 'rb') as f:
        ftp.storbinary('STOR ' + basename, f)
for path in itertools.chain(glob.glob('*.png'), glob.glob('*.gif')):
    print(path)
    with open(path, 'rb') as f:
        ftp.storbinary('STOR ' + path, f)
    os.rename(path, os.path.join('archive', path))
ftp.close()
