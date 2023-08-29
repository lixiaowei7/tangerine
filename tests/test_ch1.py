from watermelon.chapter_1 import version_space


def test_version_space_induce():
    s = set(version_space.induce())
    assert s == {(3, 4, 4), (3, 4, 7), (3, 7, 4)}
