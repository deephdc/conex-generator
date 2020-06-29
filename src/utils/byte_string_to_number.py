import re

# based on https://stackoverflow.com/a/42865957/2002471
# and https://stackoverflow.com/a/60708339/8088550
units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}

def parse_byte_string(value):
    size = value.upper()
    if not re.match(r' ', size):
        size = re.sub(r'([KMGT]?B)', r' \1', size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number)*units[unit])

