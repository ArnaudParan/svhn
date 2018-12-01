import sys

def print_progress_bar(i, total, length=20, message=None):
    if message is None:
        messge = "Progress"
    if i < total:
        sys.stderr.write(f"\r{message} : [{'=' * ((i * length) // total) + '>' + '-' * (length - ((i * length) // total) - 1)}] ({i}/{total}) {i/total:.0%}")
    else:
        sys.stderr.write(f"\r{message} : [{'=' * length}] ({i}/{total}) {i/total:.0%}")
    sys.stderr.flush()
