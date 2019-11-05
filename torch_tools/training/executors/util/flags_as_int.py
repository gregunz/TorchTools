def int_to_flags(n):
    bit_list = [int(bit) for bit in bin(n)[2:][::-1]]  # little endian list of bits as integers
    return [i for i, bit in enumerate(bit_list) if bit == 1]  # [2:] to chop off the "0b" part


def flags_to_int(flags):
    if len(flags) == 0:
        return 0
    flags_set = set(flags)
    bitlist = [int(i in flags_set) for i in range(max(flags) + 1)][::-1]
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out
