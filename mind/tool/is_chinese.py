import json
from .const import *
from unidecode import unidecode
import pinyin


def get_pin_yin(name):
    return pinyin.get(name, delimiter=" ", format="strip")


def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def cleaning_name(name):
    if is_chinese(name):
        name = get_pin_yin(name)
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