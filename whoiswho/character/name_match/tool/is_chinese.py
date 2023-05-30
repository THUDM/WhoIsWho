import json
from .const import *
from unidecode import unidecode
import pypinyin

# pypinyin.load_single_dict({ord('还'): 'hái,huán'})
# pypinyin.load_phrases_dict({'周晟': [['zhou'], ['sheng']]})  # 增加 "桔子" 词组
pinyin_special_case = {'周晟': 'zhou sheng', '胡英': 'hu ying', '郭强': 'guo qiang'}


# 不带声调的(style=pypinyin.NORMAL)
def pinyin(word):
    s = ''
    # p = pypinyin.pinyin(word, style=pypinyin.NORMAL)
    if word in pinyin_special_case:
        return pinyin_special_case[word]
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i) + ' '
    return ' '.join(s.split())


def cleaning_name(name):
    # s = pinyin('叶林')
    en_name = ''.join([i if ord(i) < 128 else ' ' for i in name])
    en_name = ' '.join(en_name.split())
    cn_name = ''.join([i if ord(i) > 128 else ' ' for i in name])
    cn_name = ' '.join(cn_name.split())
    if cn_name in pinyin_special_case.keys():  # 对某些特殊情况，直接返回其拼音
        name = pinyin_special_case[cn_name]
    elif len(en_name) < 3 and len(cn_name) > 1:  # 当作者名为 中文 时返回其拼音
        name = pinyin(cn_name)
    elif len(en_name) > 3 and len(cn_name) > 1:
        name = en_name
    name = unidecode(name)
    name = name.lower()
    new_name = ""
    for a in name:
        if a.isalpha():
            new_name += a
        else:
            new_name = new_name.strip()
            new_name += " "
    return new_name.strip()


def list_matching(list_ref, token_ref):
    matches = []
    for tok in token_ref:
        if tok in list_ref:
            matches.append(tok)
    return matches


def hash_matching(hash_ref, token_ref):
    matches = []
    for tok in token_ref:
        if tok in hash_ref:
            matches.append(tok)
    return matches


def list_excluding(list_ref1, list_ref2):
    unique_for_1 = []
    for ele in list_ref1:
        if ele not in list_ref2:
            unique_for_1.append(ele)
    return unique_for_1


def is_chinese_name(name):
    name = cleaning_name(name)
    tokens = name.split()
    full_name = [tok for tok in tokens if len(tok) > 1]
    full_size = len(full_name)

    ch_name_matches = hash_matching(chinese_name, full_name)
    ch_token_matches = hash_matching(chinese_token, full_name)

    if full_size == 2 or full_size == 1:
        if len(ch_name_matches) >= 1:
            return True

        if len(ch_token_matches) >= 1:
            ban_name_matches = list_matching(ban_list_name, full_name)
            if len(ban_name_matches) >= 1:
                return True
    elif full_size >= 3:
        ch_token_matches_ex = list_excluding(ch_token_matches, ch_name_matches)
        cname, ctkoen = len(ch_name_matches), len(ch_token_matches_ex)

        if cname >= 1 or len(ch_token_matches) >= 1:
            cname += len(list_matching(ban_list_name, full_name))
            ctkoen += len(list_matching(ban_list_token, full_name))
        if cname + ctkoen >= full_size - 1:
            return True

    return False


if __name__ == "__main__":

    names = [
        "Ernest Jordan",
        "K. MORIBE",
        "D. Jakominich",
        "William H. Nailon",
        "P. B. Littlewood",
        "A. Kuroiwa",
        "Jose Pereira",
        "Buzz Aldrin",
        "M. Till-berg",
        "E.c.c. Tsang",
        "E. A. Uliana",
        "Shankar Sa Y",
        "KAIPING HAN",
        "Xiaotao Wu",
        "Anneke A. Sohoone",
        "Harry Dankowicz",
        "Gebreselassie Baraki",
        "Yufeng Xin",
        "Mass-market Dynamics",
        "Ph. Mathieu",
        "Robert A. Granat",
        "Hafez Hadinejad-mahram",
        "H. De Hoop",
        "Mark L. Manwaring",
        "Andrew L. Goldberg",
        "Julian Brad Eld",
        "Bruce A. Rosenblatt",
        "Mitchell D. Theys",
        "Olaf E. Flippo",
        "Elisabeth Umkehrer",
        "Balasubramanian Sethuraman",
        "BARRY K. WITHERSPOON",
        "Natalia Jimeno",
        "Zhen Song",
        "Edmund Pierzchala",
        "Halina Przymusinska",
        "Jae-Hoon Kim",
        "Jonathan M. Borwein",
        "Victor M. Kureichick",
        "P Ludvigsen",
        "Mahir Hassan",
        "Na Li",
    ]
    for name in names:
        print(name, "==================", is_chinese_name(name))
