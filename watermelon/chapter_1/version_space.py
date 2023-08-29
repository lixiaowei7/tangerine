# 青绿、乌黑
# 蜷缩、硬挺、稍蜷
# 浊响、清脆、沉闷
sample = [(2, 4, 4, 1), (1, 4, 4, 1), (2, 2, 2, 0), (1, 1, 1, 0)]


def gen_hypothesis_space():
    ret = []
    for color in (2, 1, 3):
        for pedicle in (4, 2, 1, 7):
            for stroke in (4, 2, 1, 7):
                yield color, pedicle, stroke

    yield 0, 0, 0


def induce():
    ret = []
    for h_c, h_p, h_s in gen_hypothesis_space():
        out = False
        for s_c, s_p, s_s, sign in sample:
            # 与正例不一致的假设
            if not (h_c & s_c and h_p & s_p and h_s & s_s) and sign:
                out = True
                break
            # 与反例一致的假设
            if h_c & s_c and h_p & s_p and h_s & s_s and not sign:
                out = True
                break
        if out:
            continue
        ret.append((h_c, h_p, h_s))

    return ret
