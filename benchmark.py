import math
import time

def avg_time_since(since, iterations):
    now = time.time()
    ms = round(1000 * ((now - since)/iterations))
    s = math.floor(ms / 1000)
    m = math.floor(s / 60)
    return '%dm %ds %dms' % (m, s % 60, ms % 1000)
