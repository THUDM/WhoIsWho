import copy
from .util import *
from .is_chinese import is_chinese_name
import itertools

black_list = []
funcs = [
    match_name_one,
    match_name_two,
    match_name_three,
    match_name_four,
    match_name_five,
    match_name_six,
    match_name_seven,
]


def dryRun(names):
    if len(names) <= 1:
        return True

    len_of_most_complex_names, most_complex_names = len(
        next(iter(names)).split()), set()

    for name in names:
        length = len(name.split())
        if length == len_of_most_complex_names:
            most_complex_names.add(name)
        elif length > len_of_most_complex_names:
            len_of_most_complex_names = length
            most_complex_names = set((name, ))

    ## 验证 most_complex_names
    for a, b in itertools.combinations(most_complex_names, 2):
        if not may_be_duplicates_partial(a, b, True):
            return False

    ## 验证剩余的name
    for name in names:
        if name in most_complex_names:
            continue
        passed = False
        for complex_name in most_complex_names:
            if may_be_duplicates_partial(name, complex_name, True):
                passed = True
                break
        if not passed:
            return False
    return True


##  'b u mcneely'   'Betty U. Mcneely'


def match_name(funcs, name, names, loose=False):
    '''
		此方法的局限性在于：
		name = "J tang"
		names = [
			"Jie tang",
			"Jian Tang",
			"JiYao tang"
		]

		return:  {"J tang":true, "Jie tang": true },  {"Jian Tang", "JiYao tang"}  dryRun函数导致
		看清楚自己的使用场景进行names的更改使用
	'''

    if name in black_list:
        return [], []

    pt = set()
    pt.add(name)
    name_l = cleaning_name(name)
    token_l = cleaning_name(tokenize_name(name))
    for dname in names:
        if dname in black_list or dname in pt:
            continue
        dname_l = cleaning_name(dname)
        is_match = False
        for f in funcs:
            if f(name_l, dname_l, loose):
                pt.add(dname)
                is_match = True
                break
        if not is_match:
            dname_token_l = cleaning_name(tokenize_name(dname))
            for f in funcs:
                if f(token_l, dname_token_l, loose):
                    pt.add(dname)
                    break
        else:
            if not dryRun([cleaning_name(a) for a in pt]) and not dryRun(
                [cleaning_name(tokenize_name(a)) for a in pt]):
                pt.remove(dname)

    last_f = set()
    for dname in names:
        if dname not in pt:
            last_f.add(dname)

    return pt, last_f
