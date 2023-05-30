from .match_name import match_name, funcs, findMain


def MatchName(name, name_alias, name2clean, loose=False):
    return match_name(funcs, name, name_alias, name2clean, loose)


def FindMain(name, name_alias, loose=False):
    return findMain(funcs, name, name_alias, loose)
