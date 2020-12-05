import numpy as np


class SinGenerator(object):
    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, c["total_len"])
        signal_func = c["signal_func"]
        s = 0.5 * signal_func(2 * np.pi / period1 * t)
        return s.reshape((c["total_len"], 1)), t

    def apply(self, src, op, _):
        src = op(src)
        return src


class DoubleSinGenerator(object):
    def __init__(self):
        self.ops = [lambda x: x * 0.5]

    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, c["total_len"])
        src = np.empty((c["total_len"], 2))
        src[:, 0] = 0.5 * c["signal_func"](2 * np.pi / period1 * t)
        src[:, 1] = self.ops[0](src[:, 0])
        return src, t

    def apply(self, src, op, partial=None):
        # in this simple case we always change the first sine
        res = src.copy()
        res[:, 0] = op(res[:, 0])
        if partial is None:  # else leave it as is
            res[:, 1] = self.ops[0](res[:, 0])
        return res


class EchoGenerator(object):
    def __init__(self, src_signal):
        self.sig = src_signal.copy()

    def __call__(self, c):
        periods = c["periods"]
        t = np.linspace(0, periods, len(self.sig))
        return self.sig, t

    def apply(self, src, op, partial=None):
        if partial is not None:
            out_cols = np.delete(np.arange(src.shape[-1]), partial)
            res = np.empty(src.shape)
            res[:, partial] = op(src[:, partial])
            res[:, out_cols] = src[:, out_cols]
            return res
        else:
            src = op(src)
            return src.copy()
