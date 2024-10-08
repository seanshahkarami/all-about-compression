"""
Author: Sean Shahkarami

This is some code I wrote to help me understand how the DEFLATE algorithm works. I'm using Python's
builtin gzip module to work against known, valid data.

The goal here isn't the most performant or clean code but simply something which works and I was able
to use to learn how DEFLATE works.

Some interesting takeaways...

The LZ77 and Huffman coding are in some sense the easiest part of DEFLATE to understand.

I initially went in focusing on writing a compressor, and while that taught me quite a bit
about LZ77 and Huffman codes (and how easy it was to implement these in Python!), most of
the work seems to be in squeezing every bit of data out we can by repeated application of
these ideas.

What surprised me the most is learning that the literal / length and distance code trees
themselves are put through a round of runlength + Huffman coding!

Additionally, the use of length and distance coding tables along with extra bits as a kind
of variable length protocol is pretty slick. This is a really neat idea and something worth
keeping in mind for the future.
"""

from collections import deque
import gzip
from random import randbytes, randint
from io import BytesIO


class BitReader:
    """
    Helper object to read from bit stream since DEFLATE operates on variable length bit codes.
    """

    def __init__(self, data):
        self.reader = BytesIO(data)
        self.bits = 0
        self.numbits = 0

    def morebits(self):
        r = self.reader.read(1)
        if len(r) != 1:
            raise IOError("unable to read data")
        self.bits |= r[0] << self.numbits
        self.numbits += 8

    def readbits(self, n):
        while self.numbits < n:
            self.morebits()
        mask = (1 << n) - 1
        r = self.bits & mask
        self.bits >>= n
        self.numbits -= n
        return r

    def readbytes(self, n):
        self.bits = 0
        self.numbits = 0
        return self.reader.read(n)


def test_bit_reader():
    reader = BitReader(bytes([0b10000001, 0b10101111]))
    assert reader.readbits(3) == 0b001
    assert reader.readbits(5) == 0b10000

    assert reader.readbits(4) == 0b1111
    assert reader.readbits(4) == 0b1010

    reader = BitReader(bytes([0b10000001, 0b10101111]))
    assert reader.readbytes(2) == bytes([0b10000001, 0b10101111])

    reader = BitReader(bytes([0b10000001, 0b10101111]))
    reader.readbits(3)
    assert reader.readbytes(1) == bytes([0b10101111])

    reader = BitReader(bytes([0b11111110, 0b00100111]))
    assert reader.readbits(4) == 14
    assert reader.readbits(4) == 15
    assert reader.readbits(2) == 3
    assert reader.readbits(2) == 1
    assert reader.readbits(2) == 2
    assert reader.readbits(2) == 0


def build_code_tree(pairs):
    """
    This function builds a code tree from a list of (code length, symbol) pairs.
    """
    nonzero_pairs = [(cl, i) for (cl, i) in pairs if cl > 0]
    items = deque(sorted(nonzero_pairs))
    return find_codes("", items)


def find_codes(path, items):
    """
    This is the actual function which build_code_tree uses to find codes matching the code length.

    I know the DEFLATE RFC provides a more slick way to compute this but I found the easiest conceptual
    way to do this to simply "walk a binary tree" and greedily emit codes of the right length.
    """
    if len(items) == 0:
        return None

    length, value = items[0]

    if len(path) > length:
        raise ValueError("unable to construct code")

    if len(path) == length:
        items.popleft()
        # truncate tree here and return code
        return (value, None, None, path)

    left = find_codes(path + "0", items)
    right = find_codes(path + "1", items)
    return (None, left, right, None)


def build_fixed_literal_lengths_tree():
    """
    The DEFLATE RFC defines this as a general purpose code length for the literal / length alphabet.
    """
    fixed_code = []

    for i in range(0, 144):
        fixed_code.append((8, i))

    for i in range(144, 256):
        fixed_code.append((9, i))

    for i in range(256, 280):
        fixed_code.append((7, i))

    for i in range(280, 288):
        fixed_code.append((8, i))

    return build_code_tree(fixed_code)


def build_fixed_distance_tree():
    """
    The DEFLATE RFC defines this as a general purpose code length for the distance alphabet.
    """
    fixed_code = []

    for i in range(0, 32):
        fixed_code.append((5, i))

    return build_code_tree(fixed_code)


fixed_literal_length_tree = build_fixed_literal_lengths_tree()
fixed_distance_tree = build_fixed_distance_tree()


# Now we have quite a few constants defined in the DEFLATE RFC. These seem to be based on statistical
# observations about how frequently these LZ77 copy operations appear.
length_base = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    19,
    23,
    27,
    31,
    35,
    43,
    51,
    59,
    67,
    83,
    99,
    115,
    131,
    163,
    195,
    227,
    258,
]


length_extra_bits = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    0,
]

distance_base = [
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    25,
    33,
    49,
    65,
    97,
    129,
    193,
    257,
    385,
    513,
    769,
    1025,
    1537,
    2049,
    3073,
    4097,
    6145,
    8193,
    12289,
    16385,
    24577,
]

distance_extra_bits = [
    0,
    0,
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
]


def decompress(gzdata):
    """
    This is the main entry point to DEFLATE decompression. Note, we only read a single block here but it's
    obvious how to extend this to multiple blocks.
    """
    _, compressed_data = read_header(gzdata)

    output = bytearray()

    reader = BitReader(compressed_data)

    # read block header
    while True:
        block_final = reader.readbits(1)
        block_type = reader.readbits(2)

        if block_type == 0:
            decode_uncompressed_block(reader, output)
        elif block_type == 1:
            decode_fixed_compressed_block(reader, output)
        elif block_type == 2:
            decode_dynamic_compressed_block(reader, output)
        else:
            raise NotImplementedError(f"Not implemented block type {block_type}")

        if block_final == 1:
            break

    return bytes(output)


def decode_uncompressed_block(reader, output):
    len_bytes = reader.readbytes(2)
    nlen_bytes = reader.readbytes(2)

    block_len = int.from_bytes(len_bytes, "little")
    block_nlen = int.from_bytes(nlen_bytes, "little")
    assert block_len + block_nlen == 0xFFFF

    output.extend(reader.readbytes(block_len))


def decode_fixed_compressed_block(reader, output):
    decode_compressed_block(
        reader, output, fixed_literal_length_tree, fixed_distance_tree
    )


def decode_dynamic_compressed_block(reader, output):
    # read code lengths
    num_ll_codes = reader.readbits(5) + 257
    num_dist_codes = reader.readbits(5) + 1
    num_code_lengths = reader.readbits(4) + 4

    # read the runlength code length tree
    cl_tree = decode_cl_tree(reader, num_code_lengths)

    # read the literal / length and distance code lengths
    # NOTE The DEFLATE RFC seems to say ll and dist lengths can exist back to back but
    # I found doing two reads passes all test cases I have.
    ll_lengths = decode_huffman_tree(reader, num_ll_codes, cl_tree)
    dist_lengths = decode_huffman_tree(reader, num_dist_codes, cl_tree)

    # build literal / length and distance code trees
    ll_tree = build_code_tree(ll_lengths)
    dist_tree = build_code_tree(dist_lengths)

    decode_compressed_block(reader, output, ll_tree, dist_tree)


def read_header(data):
    """
    Reads gzip header and returns compressed data with DEFLATE.
    """

    # check identifier gzip 1f8b
    assert data[0] == 0x1F
    assert data[1] == 0x8B

    # compression should be DEFLATE
    assert data[2] == 8

    # flags should all be zero
    assert data[3] == 0

    # extra flags should be 2 (best compression)
    assert data[8] == 2

    # os should be unknown (python behavior)
    assert data[9] == 255

    return {"mtime": "wow"}, data[10:]


def next_code(reader: BitReader, tree):
    """
    Walks a code tree bit by bit to find the next code.
    """
    node = tree

    while True:
        bit = reader.readbits(1)
        if bit == 0:
            node = node[1]
        elif bit == 1:
            node = node[2]
        if node[0] is not None:
            return node[0]


def decode_length(reader, code):
    """
    Decode a length using the length + extra bits table from RFC
    """
    index = code - 257
    extra_bits = reader.readbits(length_extra_bits[index])
    return length_base[index] + extra_bits


def decode_distance(reader, code):
    """
    Decode a length using the distance + extra bits table from RFC
    """
    index = code
    extra_bits = reader.readbits(distance_extra_bits[index])
    return distance_base[index] + extra_bits


# code length tree data is also ordered by statistical appearance of runlength code length codes...
cl_code_order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]


def decode_cl_tree(reader, num_codes):
    # read and build code length tree
    cl_code_lengths = []

    for i in range(num_codes):
        numbits = reader.readbits(3)
        if numbits == 0:
            continue
        cl_code_lengths.append((numbits, cl_code_order[i]))

    return build_code_tree(cl_code_lengths)


def decode_compressed_block(reader, output, literal_length_tree, distance_tree):
    """
    Decode compressed block using literal / length codes and distance codes.
    """

    while True:
        code = next_code(reader, literal_length_tree)

        # emit literal codes immediately
        if code < 256:
            output.append(code)
            continue

        # code 256 indicates end of block
        if code == 256:
            return

        # otherwise, finish decoding length from extra bits
        length = decode_length(reader, code)

        # next code will be distance code + extra bits
        code = next_code(reader, distance_tree)
        distance = decode_distance(reader, code)

        # copy previous data using lz77 algorithm and length / distance we just decoded
        start = len(output) - distance
        for i in range(length):
            output.append(output[start + i])


def decode_huffman_tree(reader, num_codes, cl_tree):
    """
    Decode dynamic Huffman tree using runlength codes specified in DEFLATE RFC.
    """
    code_lengths = []

    while len(code_lengths) < num_codes:
        code = next_code(reader, cl_tree)
        if code < 16:
            code_lengths.append(code)
        elif code == 16:
            n = 3 + reader.readbits(2)
            appendn(code_lengths, code_lengths[-1], n)
        elif code == 17:
            n = 3 + reader.readbits(3)
            appendn(code_lengths, 0, n)
        elif code == 18:
            n = 11 + reader.readbits(7)
            appendn(code_lengths, 0, n)
        else:
            raise ValueError(f"unexpected cl code {code}")

    assert len(code_lengths) == num_codes

    return [(cl, i) for i, cl in enumerate(code_lengths)]


def appendn(arr, val, n):
    for _ in range(n):
        arr.append(val)


def test_decompress():
    examples = [
        ("empty", b""),
        ("0", b"0"),
        ("01", b"01"),
        ("0s", b"00000000000000000000000000000"),
        ("xys", b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyy"),
        ("1-10 1s", b"12345678901111111111111111111"),
        ("2 ax", b"axax"),
        ("3 ax", b"axaxax"),
        ("rep 100", b"axaxax" * 100),
        ("rep 1000", b"axaxax" * 1000),
    ]

    for name, example in examples:
        try:
            assert decompress(gzip.compress(example)) == example
        except AssertionError:
            print("FAIL", name)


def fuzz_decompress():
    for _ in range(1000):
        example = randbytes(randint(0, 4096))
        assert decompress(gzip.compress(example)) == example


def main():
    test_bit_reader()
    test_decompress()
    fuzz_decompress()


if __name__ == "__main__":
    main()
