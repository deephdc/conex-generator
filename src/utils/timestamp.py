import datetime


def timestamp():
    """Return a timestamp from datetime.now() without "-" and ":".

    Returns
    -------
    timestamp : str
    """
    stamp = datetime.datetime.now()
    return str(stamp).replace(" ","-").replace(":","-")


def timestamp_utc_iso():
    """Return an ISO compatible timestamp relative to UTC.

    Returns
    -------
    timestamp : str
    """
    stamp = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    return stamp.isoformat()


def timestamp_local_iso():
    """Return an ISO compatible timestamp relative to localtime.

    Returns
    -------
    timestamp : str
    """
    stamp = datetime.datetime.now().astimezone()
    return stamp.isoformat()

