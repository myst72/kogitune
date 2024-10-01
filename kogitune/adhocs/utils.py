import time
from .stack import adhoc_print, notice

## フォーマット

def format_unit(num: int, scale=1000) -> str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.2f}K"
        elif num < scale**3:
            return f"{num / scale**2:.2f}M"
        elif num < scale**4:
            return f"{num / scale**3:.2f}G"
        elif num < scale**5:
            return f"{num / scale**4:.2f}T"
        elif num < scale**6:
            return f"{num / scale**5:.2f}P"
        else:
            return f"{num / scale**6:.2f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.3f}ms"
        day = num // (3600 * 24)
        num = num % (3600 * 24)
        hour = num // 3600
        num = num % 3600
        min = num // 60
        sec = num % 60
        if day > 0:
            return f"{day}d{hour}h{min}m{sec:.0f}s"
        elif hour > 0:
            return f"{hour}h{min}m{sec:.0f}s"
        elif min > 0:
            return f"{min}m{sec:.0f}s"
        return f"{sec:.3f}s"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.2f}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        else:
            return f"{num / 1_000_000_000_000:.2f}T"


class start_timer(object):
    """
    タイマー
    """

    def __init__(self):
        pass

    def __enter__(self):
        self.start_time = time.time()
        return self

    def notice(self, *args, iteration="total", **kwargs):
        elapsed_time = time.time() - self.start_time
        total = kwargs.get(iteration, None)
        if total is not None and total > 0:
            kwargs = dict(
                elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time, 3),
                throughput=round(elapsed_time / total, 3),
                iteration=total,
                **kwargs,
            )
        else:
            kwargs = dict(
                elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time, 3),
                **kwargs,
            )
        notice(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def describe_counters(
    counters, caption, output_file, columns=["Item", "Count", "%"], head=20
):
    import pandas as pd

    df = pd.DataFrame.from_dict(counters, orient="index").reset_index()
    df.columns = columns[:2]
    total_count = df[columns[1]].sum()
    df[columns[2]] = (df[columns[1]] / total_count) * 100
    df[columns[2]] = df[columns[2]].round(3)
    adhoc_print(df.head(head), face="")
    adhoc_print(f"{caption} See {output_file}")
