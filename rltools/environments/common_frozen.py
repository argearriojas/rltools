import numpy as np


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4


MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "4x4empty": [
        "FFFF",
        "FSFF",
        "FFGF",
        "FFFF"
    ],
    "5x5empty": [
        "FFFFF",
        "FSFFF",
        "FFFFF",
        "FFFGF",
        "FFFFF"
    ],
    "6x6empty": [
        "FFFFFF",
        "FSFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFGF",
        "FFFFFF"
    ],
    "8x8empty": [
        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "5x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "10x10empty": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "8x8candy": [

        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFCFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF",
    ],
    "10x10candy": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFCFFFFFFF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2-nails": [
        "FFFFFFFFFF",
        "FFFFNFNFNF",
        "FFSFFCFCFF",
        "FFFFNFNFNF",
        "FFFFFFFCFF",
        "FFCFFFNFNF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "11x11gradient": [
        "22222223432",
        "21112234543",
        "21012345654",
        "21112234543",
        "22222123432",
        "22221S12322",
        "22222122222",
        "33333333333",
        "44444444444",
        "55555555555",
        "66666666666",
    ],
    "11x11gradient-x2": [
        "98489444444",
        "84248445754",
        "42024447974",
        "84244445754",
        "98444244444",
        "44442S24444",
        "44444244444",
        "55555555555",
        "66666666666",
        "77777777777",
        "88888888888",
    ],
    "11x11zigzag": [
        "FFFFFFFFFFF",
        "FSFFFFFFFFF",
        "FFFFFFFFFFF",
        "WWWWWWWFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFWWWWWWW",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "15x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "16x16empty": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candy": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candyx2": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFCFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFCFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "5x17empty": [
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
    ],
    "5x24empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "7x32empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFSFFFFFFFFFFFFFFFFFFFFFFGFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "15x45": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "17x17center": [
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFGFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
    ],
    "5x15zigzag": [
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
    ],
    "8x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "11x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "23x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
    ],
    "15x15mixed": [
        'FFFFFFFSFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFWWWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'WWWWFFFWWWWWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFWWWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFGFFFFFFF',
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] not in '#H':
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


def evolv_state():
    pass
