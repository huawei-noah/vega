from functools import wraps, partial


def make_decorator(func):
    """Return wrapped function that acts as decorator if no extra positional args are given."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0:
            return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapped


def singleton(cls):
    """Return wrapped class that has only one instance."""
    inst = []

    def get_instance(*args, **kwargs):
        if not inst:
            inst.append(cls(*args, **kwargs))
        return inst[0]
    return get_instance
