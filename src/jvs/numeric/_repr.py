"""Bounded diagnostic representations for arbitrary-size exact integers."""


def integer_repr(value: int) -> str:
    """Show small integers in decimal and large ones as explicit abbreviations.

    Avoid decimal conversion of large values and never build a full hex string.
    The 256-bit decimal cutoff is below Python's minimum configurable digit limit.
    """
    bits = value.bit_length()
    if bits <= 256:
        return repr(value)
    magnitude = abs(value)
    leading = magnitude >> (bits - 32)
    trailing = magnitude & 0xFFFFFFFF
    sign = "-" if value < 0 else "+"
    return (
        f"<integer sign={sign}, bits={bits}, "
        f"top32=0x{leading:08x}, bottom32=0x{trailing:08x}, ...>"
    )
