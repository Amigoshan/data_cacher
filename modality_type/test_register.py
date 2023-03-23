common = dict()


def register(dst):
    def dec_register(cls):
        dst[cls.__name__] = cls
        return cls
    return dec_register

@register(common)
class father(object):
    def __init__(self) -> None:
        import ipdb;ipdb.set_trace()

@register(common)
class son0(father):
    def __init__(self) -> None:
        import ipdb;ipdb.set_trace()
        super().__init__()

class son1(father):
    def __init__(self) -> None:
        import ipdb;ipdb.set_trace()
        super().__init__()

print(common)
import ipdb;ipdb.set_trace()