import datetime


def timestamp():
    stamp = datetime.datetime.now()
    return str(stamp).replace(" ","-").replace(":","-")


def timestamp_utc_iso():
    stamp = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    return stamp.isoformat()


def timestamp_local_iso():
    stamp = datetime.datetime.now().astimezone()
    return stamp.isoformat()

