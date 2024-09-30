from datetime import datetime
import sys


def log_msg(header, *ls_statements):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t} {header}] ", end="")
    print(*ls_statements)
    sys.stdout.flush()