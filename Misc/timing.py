from functools import wraps
import time


def time_fxn(the_fxn):
    @wraps(the_fxn)
    def wrapper(*args, **kargs):
        t0 = time.time()
        the_fxn(*args,**kargs)
        t1 = time.time()
        total = t1 - t0
        minutes = total // 60
        seconds = total % 60
        return ' Run time: ' + str(int(minutes)) + ' min ' + str(seconds) + ' sec'
        #return the_fxn.__name__ + ' Run time: ' + str(int(minutes)) + ' min ' + str(seconds) + ' sec'
    return wrapper
