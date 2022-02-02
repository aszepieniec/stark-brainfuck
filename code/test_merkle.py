from binascii import hexlify
from salted_merkle import SaltedMerkle
from merkle import Merkle
from os import urandom


def test_salted_merkle():
    n = 64
    elements = [list(l) for l in zip([urandom(int(urandom(1)[0]))
                                      for i in range(n)], [urandom(int(urandom(1)[0])) for i in range(n)])]

    tree = SaltedMerkle(elements)
    root = tree.root()

    # opening any leaf should work
    for i in range(n):
        salt, path = tree.open(i)
        assert(SaltedMerkle.verify(root, i, salt, path, elements[i]))

    # opening non-leafs should not work
    for i in range(n):
        salt, path = tree.open(i)
        assert(False == SaltedMerkle.verify(root, i, salt, path, urandom(51)))

    # opening wrong leafs should not work
    for i in range(n):
        salt, path = tree.open(i)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == SaltedMerkle.verify(root, i, salt, path, elements[j]))

    # opening leafs with the wrong index should not work
    for i in range(n):
        salt, path = tree.open(i)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == SaltedMerkle.verify(root, j, salt, path, elements[i]))

    # opening leafs to a false root should not work
    for i in range(n):
        salt, path = tree.open(i)
        assert(False == SaltedMerkle.verify(
            urandom(32), i, salt, path, elements[i]))

    # opening leafs with even one falsehood in the path should not work
    for i in range(n):
        salt, path = tree.open(i)
        for j in range(len(path)):
            fake_path = path[0:j] + [urandom(32)] + path[j+1:]
            assert(False == SaltedMerkle.verify(
                root, i, salt, fake_path, elements[i]))

    # opening leafs with false salt should not work
    for i in range(n):
        salt, path = tree.open(i)
        assert(False == SaltedMerkle.verify(
            root, i, urandom(32), path, elements[i]))


def test_merkle():
    n = 64
    elements = [list(l) for l in zip([urandom(int(urandom(1)[0]))
                                      for i in range(n)], [urandom(int(urandom(1)[0])) for i in range(n)])]

    tree = Merkle(elements)
    root = tree.root()

    # opening any leaf should work
    for i in range(n):
        path = tree.open(i)
        assert(Merkle.verify(root, i, path, elements[i]))

    # opening non-leafs should not work
    for i in range(n):
        path = tree.open(i)
        assert(False == Merkle.verify(root, i, path, urandom(51)))

    # opening wrong leafs should not work
    for i in range(n):
        path = tree.open(i)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == Merkle.verify(root, i, path, elements[j]))

    # opening leafs with the wrong index should not work
    for i in range(n):
        path = tree.open(i)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == Merkle.verify(root, j, path, elements[i]))

    # opening leafs to a false root should not work
    for i in range(n):
        path = tree.open(i)
        assert(False == Merkle.verify(
            urandom(32), i, path, elements[i]))

    # opening leafs with even one falsehood in the path should not work
    for i in range(n):
        path = tree.open(i)
        for j in range(len(path)):
            fake_path = path[0:j] + [urandom(32)] + path[j+1:]
            assert(False == Merkle.verify(
                root, i, fake_path, elements[i]))
