import os
#import requests
from urllib.parse import urljoin
import datetime


def cur_date():
    dt = datetime.datetime.now()

    if dt.month < 10:
        cr_month = str('0' + str(dt.month))
    else:
        cr_month = str(dt.month)

    if dt.day < 10:
        cr_day = str('0' + str(dt.day))
    else:
        cr_day = str(dt.day)

    cur_date1 = str(dt.year) + '.' + \
                cr_month + '.' + \
                cr_day + '.' + \
                str(dt.hour) + '.' + \
                str(dt.minute)
    return str(cur_date1)


def get_size(pth='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(pth):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    return round(total_size/1024/1024, 2)  # in megabytes


def main():
    current_date = cur_date()
    print(f'The current date is:\n{current_date}')


if __name__ == '__main__':
    main()
