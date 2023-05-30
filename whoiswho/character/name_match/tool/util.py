import collections
import copy
from .is_chinese import is_chinese_name, cleaning_name
from .token import tokenize_name


def name2dict(name):
    name_dict = collections.defaultdict(int)
    for word in name.split():
        name_dict[word] += 1
    return name_dict


def same_name(a, b):
    return name2dict(a.replace('.', ' ')) == name2dict(b.replace('.', ' '))


def is_abbr_word(w):
    return w.endswith(".") or len(w) == 1


def split_abbr_full(name):
    abbr_part, full_part = [], []
    for word in name.split():
        if is_abbr_word(word):
            abbr_part.append(word)
        else:
            full_part.append(word)
    return abbr_part, full_part


def may_be_duplicates_partial(a, b, loose=False):
    ret = same_name(a, b) or is_abbr_of(a, b, True) or is_abbr_of(b, a, True)
    if loose:
        ret = ret or is_abbr_of(a, b, True, True) or is_abbr_of(
            b, a, True, True)
    return ret


def get_first_chars(name):
    if isinstance(name, list):
        name = ' '.join(name)
    first_chars = collections.defaultdict(int)
    for word in name.split():
        first_chars[word[0]] += 1
    return first_chars


def is_abbr_of(a, b, partial=False, loose=False):
    if same_name(a, b):
        return True

    abbr_part_a, full_part_a = split_abbr_full(a)
    abbr_part_b, full_part_b = split_abbr_full(b)
    common_words = set(abbr_part_a + full_part_a).intersection(set(abbr_part_b + full_part_b))

    for word in common_words:
        if len(word) < 2:
            continue
        if word in abbr_part_a:
            abbr_part_a.remove(word)
        if word in full_part_a:
            full_part_a.remove(word)
        if word in abbr_part_b:
            abbr_part_b.remove(word)
        if word in full_part_b:
            full_part_b.remove(word)

    if loose:
        first_chars_a, first_chars_b = get_first_chars(' '.join(abbr_part_a + full_part_a)), get_first_chars(
            ' '.join(abbr_part_b + full_part_b))
        if not (set(first_chars_b.keys()).issubset(first_chars_a.keys()) or set(first_chars_a.keys()).issubset(
                first_chars_b.keys())):
            return False

        if len(full_part_a) != 0 and len(full_part_b) != 0:
            for word_a in full_part_a:
                prefix = word_a if len(word_a) < 4 else word_a[:3]
                suffix = word_a if len(word_a) < 4 else word_a[-3:]
                match = False
                for word_b in full_part_b:
                    if word_b.startswith(prefix) or word_b.endswith(suffix):
                        match = True
                        break
                if not match: return False

        return True

    abbr_part_b_copy, full_part_b_copy = copy.copy(abbr_part_b), copy.copy(full_part_b)

    for word_a in full_part_a:
        if partial:
            include = False
            for word_b in full_part_b:
                if word_b.startswith(word_a):
                    full_part_b_copy.remove(word_b)
                    include = True
                    break
            if not include:
                return False
        else:
            if word_a in full_part_b:
                full_part_b_copy.remove(word_a)
            else:
                return False
        full_part_b = full_part_b_copy

    for word_a in abbr_part_a:
        startswith = False
        for word_b in abbr_part_b:
            if word_b.startswith(word_a):
                abbr_part_b_copy.remove(word_b)
                startswith = True
                break
        for word_b in full_part_b:
            if word_b.startswith(word_a):
                full_part_b_copy.remove(word_b)
                startswith = True
                break
        if not startswith:
            return False
        abbr_part_b, full_part_b = abbr_part_b_copy, full_part_b_copy

    return True


def has_middle_name(name):
    abbr, full = split_abbr_full(name)
    return len(abbr) == 1 and len(full) == 2


def remove_middle_name(name):
    new_name = []
    for word in name.split():
        if is_abbr_word(word): continue
        new_name.append(word)
    return ' '.join(new_name)


# match func
def match_name_one(main_name, dupl_name, loose=False):
    if same_name(main_name, dupl_name):
        return True
    _, full_a = split_abbr_full(main_name)
    _, full_b = split_abbr_full(dupl_name)

    if len(full_a) != 0 and len("".join(full_a)) == len("".join(full_b)):
        if len(full_a) > len(full_b):
            s = "".join(full_a)
            m = True
            for word_b in full_b:
                if s.find(word_b) == -1:
                    m = False
                    break
            if m:
                return True
        else:
            s = "".join(full_b)
            m = True
            for word_a in full_a:
                if s.find(word_a) == -1:
                    m = False
                    break
            if m:
                return True
    return False


def match_name_two(main_name, dupl_name, loose=False):
    main_is_chinese, dupl_is_chinese = is_chinese_name(main_name), is_chinese_name(dupl_name)
    full_a, full_b = [], []
    if main_is_chinese and dupl_is_chinese:
        abbr_a, full_a = split_abbr_full(main_name)
        abbr_b, full_b = split_abbr_full(dupl_name)
        if len(abbr_a) == 0 and len(abbr_b) == 0:
            return False
    if (main_is_chinese and len(main_name.split()) < 2) or (dupl_is_chinese and len(dupl_name.split()) < 2):
        return False
    if loose:
        if (main_is_chinese or dupl_is_chinese) and (len(full_a) == 0 or len(full_b) == 0):
            return False

        return is_abbr_of(main_name, dupl_name, loose=True)
    return is_abbr_of(main_name, dupl_name, loose=False) and get_first_chars(main_name) == get_first_chars(dupl_name)


# 非中文
def match_name_three(main_name, dupl_name, loose=False):
    main_is_chinese, dupl_is_chinese = is_chinese_name(main_name), is_chinese_name(dupl_name)
    if main_is_chinese and dupl_is_chinese:
        return False
    return is_abbr_of(main_name, dupl_name)


# 非中文
def match_name_four(main_name, dupl_name, loose=False):
    main_is_chinese, dupl_is_chinese = is_chinese_name(main_name), is_chinese_name(dupl_name)
    if main_is_chinese and dupl_is_chinese:
        return False
    return is_abbr_of(main_name, dupl_name, True)


def match_name_five(main_name, dupl_name, loose=False):
    if len(main_name) == 0 or len(dupl_name) == 0:
        return False
    main_words, dupl_words = main_name.split(), dupl_name.split()
    if len(main_words) < 3 or len(dupl_words) < 3:
        return False
    if not ("".join(main_words[:-1]) == "".join(dupl_words[:-1])):
        return False
    return main_words[-1] == dupl_words[-1][:-1] or dupl_words[-1] == main_words[-1][:-1]


# 非中文
def match_name_six(main_name, dupl_name, loose=False):
    main_is_chinese, dupl_is_chinese = is_chinese_name(main_name), is_chinese_name(dupl_name)
    if main_is_chinese and dupl_is_chinese:
        return False
    if has_middle_name(main_name) == has_middle_name(dupl_name):
        return False
    if has_middle_name(main_name):
        main_name = remove_middle_name(main_name)
    if has_middle_name(dupl_name):
        dupl_name = remove_middle_name(dupl_name)
    main_name, dupl_name = main_name.replace(" ", ""), dupl_name.replace(" ", "")
    if main_name != dupl_name and (main_name.startswith(dupl_name) or dupl_name.startswith(main_name) and abs(
            len(main_name) - len(dupl_name)) < 3):
        return True
    return False


# 中文拼音
def match_name_seven(main_name, dupl_name, loose=False):
    main_is_chinese, dupl_is_chinese = is_chinese_name(main_name), is_chinese_name(dupl_name)
    if not main_is_chinese or not dupl_is_chinese:
        return False
    main_words, dupl_words = main_name.split(), dupl_name.split()
    if len(main_words) < 3 or len(dupl_words) < 3:
        return False
    abbr_a, full_a = split_abbr_full(main_name)
    abbr_b, full_b = split_abbr_full(dupl_name)
    if len(abbr_a) != 0 or len(abbr_b) != 0:
        return False
    if len(full_a) > len(full_b):
        for b in full_b:
            if b not in full_a:
                return False
    else:
        for a in full_a:
            if a not in full_b:
                return False
    return True
