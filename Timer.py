import time


def timeit(func):
    def wrapper(*args, **kwargs):
        name = func
        for attr in ('__qualname__', '__name__'):
            if hasattr(func, attr):
                name = getattr(func, attr)
                break

        logger.info("Start call: {}".format(name))  # If don't want to use log, print will be fine.
        now = time.time()
        result = func(*args, **kwargs)
        using = (time.time() - now)
        logger.info("End call {}, using: {:.1f} s".format(name, using))
        
        return result
    return wrapper