from datetime import datetime


def dateStrToDayYear(dateStr):
    return datetime.strptime(dateStr, '%Y-%m-%d').timetuple().tm_yday


