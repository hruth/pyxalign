import os

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

default_divisor = 32
new_divisor = int(os.environ.get("PYXALIGN_DIVISOR", default_divisor))
if not is_power_of_two(new_divisor):
    raise ValueError(f"The divisor must be a power of 2, but PYXALIGN_DIVISOR was set to {new_divisor}")
else:
    divisor = new_divisor