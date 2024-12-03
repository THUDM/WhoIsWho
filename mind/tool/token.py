import re
from unidecode import unidecode

STOPWORDS = {'jr', 'iii', 'dr', 'mr', 'junior'}

NICKNAME_DICT = {
    "al": "albert",
    "andy": "andrew",
    "tony": "anthony",
    "art": "arthur",
    "arty": "arthur",
    "bernie": "bernard",
    "bern": "bernard",
    "charlie": "charles",
    "chuck": "charles",
    "danny": "daniel",
    "dan": "daniel",
    "don": "donald",
    "ed": "edward",
    "eddie": "edward",
    "gene": "eugene",
    "fran": "francis",
    "freddy": "frederick",
    "fred": "frederick",
    "hank": "henry",
    "irv": "irving",
    "jimmy": "james",
    "jim": "james",
    "joe": "joseph",
    "jacky": "john",
    "jack": "john",
    "jeff": "jeffrey",
    "ken": "kenneth",
    "larry": "lawrence",
    "leo": "leonard",
    "matt": "matthew",
    "mike": "michael",
    "nate": "nathan",
    "nat": "nathan",
    "nick": "nicholas",
    "pat": "patrick",
    "pete": "peter",
    "ray": "raymond",
    "dick": "richard",
    "rick": "richard",
    "bob: bobby: rob": "robert",
    "ron: ronny": "ronald",
    "russ": "russell",
    "sam: sammy": "samuel",
    "steve": "stephan",
    "stu": "stuart",
    "teddy": "theodore",
    "ted": "theodore",
    "tom": "thomas",
    "thom": "thomas",
    "tommy": "thomas",
    "timmy": "timothy",
    "tim": "timothy",
    "walt": "walter",
    "wally": "walter",
    "bill": "william",
    "billy": "william",
    "will": "william",
    "willy": "william",
    "mandy": "amanda",
    "cathy": "catherine",
    "cath": "catherine",
    "chris": "christopher",
    "chrissy": "christine",
    "cindy: cynth": "cynthia",
    "debbie": "deborah",
    "deb": "deborah",
    "betty": "elizabeth",
    "beth": "elizabeth",
    "liz": "elizabeth",
    "bess": "elizabeth",
    "flo": "florence",
    "francie": "frances",
    "fran": "frances",
    "jan": "janet",
    "kate": "katherine",
    "kathy": "katherine",
    "jan": "janice",
    "nan": "nancy",
    "pam": "pamela",
    "pat": "patricia",
    "bobbie": "roberta",
    "sophie": "sophia",
    "sue": "susan",
    "suzie": "susan",
    "terry": "teresa",
    "val": "valerie",
    "ronnie": "veronica",
    "vonna": "yvonne",
    "peggy": "margaret",
    "ted": "edward",
    "sally": "sarah",
    "harry": "henry",
}


def tokenize_name(name):
    splitted_name = []
    for word in name.split():
        if len(word) == 2 and word.count('.') == 0 and word.isupper():
            word = ' '.join(word)
        splitted_name.append(word)
    name = ' '.join(splitted_name).replace("'", '').replace("’", '')
    name = re.sub('[^\w.]', ' ', name).lower()
    name = unidecode(name)
    splitted_name = []
    for word in name.split():
        if word.replace('.', '') in STOPWORDS: continue
        if word in NICKNAME_DICT: word = NICKNAME_DICT[word]
        if word.count('.') > 1: word = ' '.join(word.split('.'))
        splitted_name.append(word)
    name = ' '.join(splitted_name)
    name = re.sub(' +', ' ', name.encode('ascii', 'ignore').decode('ascii'))
    return name


if __name__ == "__main__":
    names = ['m ćwiok', 'm. ćwiok']
    for name in names:
        two_name = tokenize_name(name)
        print("origin: ", name, " transformer: ", two_name)