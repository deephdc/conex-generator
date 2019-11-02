from datetime import datetime


def timestamp():
    stamp = datetime.now()
    return str(stamp).replace(" ","-").replace(":","-")

