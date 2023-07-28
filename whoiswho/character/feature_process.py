import json
import random
import pickle
import math
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, Manager
import numpy as np
import copy
from pyjarowinkler import distance
from time import time
from time import strftime
from time import localtime
from operator import itemgetter
import math
import re
from unidecode import unidecode
from .name_match.tool.interface import MatchName
from collections import Counter
from operator import itemgetter
from tqdm import tqdm
import sys,os
from whoiswho.config import paper_idf_path

np.set_printoptions(suppress=True)
num_processing = int(multiprocessing.cpu_count() / 2)

class featureGeneration:
    def __init__(self):
        self.__loadEssential()

    def __loadEssential(self):

        data_dir = os.path.abspath(paper_idf_path)
        print("tf-idf file path:",data_dir)
        with open(data_dir + '/name_uniq_dict.json', "r") as file:
            self.name_uniq_dict = json.load(file)
        with open(data_dir + '/venue_idf.json', "r") as file:
            self.ven_tfidf = json.load(file)
        with open(data_dir + '/new_org_idf.json', "r") as file:
            self.org_tfidf = json.load(file)
        with open(data_dir + '/title_idf.json', "r") as file:
            self.title_tfidf = json.load(file)

        self.STOPWORDS = {'jr', 'iii', 'dr', 'mr', 'junior'}

        self.NICKNAME_DICT = {
            "al"             : "albert",
            "andy"           : "andrew",
            "tony"           : "anthony",
            "art"            : "arthur",
            "arty"           : "arthur",
            "bernie"         : "bernard",
            "bern"           : "bernard",
            "charlie"        : "charles",
            "chuck"          : "charles",
            "danny"          : "daniel",
            "dan"            : "daniel",
            "don"            : "donald",
            "ed"             : "edward",
            "eddie"          : "edward",
            "gene"           : "eugene",
            "fran"           : "francis",
            "freddy"         : "frederick",
            "fred"           : "frederick",
            "hank"           : "henry",
            "irv"            : "irving",
            "jimmy"          : "james",
            "jim"            : "james",
            "joe"            : "joseph",
            "jacky"          : "john",
            "jack"           : "john",
            "jeff"           : "jeffrey",
            "ken"            : "kenneth",
            "larry"          : "lawrence",
            "leo"            : "leonard",
            "matt"           : "matthew",
            "mike"           : "michael",
            "nate"           : "nathan",
            "nat"            : "nathan",
            "nick"           : "nicholas",
            "pat"            : "patrick",
            "pete"           : "peter",
            "ray"            : "raymond",
            "dick"           : "richard",
            "rick"           : "richard",
            "bob: bobby: rob": "robert",
            "ron: ronny"     : "ronald",
            "russ"           : "russell",
            "sam: sammy"     : "samuel",
            "steve"          : "stephan",
            "stu"            : "stuart",
            "teddy"          : "theodore",
            "ted"            : "theodore",
            "tom"            : "thomas",
            "thom"           : "thomas",
            "tommy"          : "thomas",
            "timmy"          : "timothy",
            "tim"            : "timothy",
            "walt"           : "walter",
            "wally"          : "walter",
            "bill"           : "william",
            "billy"          : "william",
            "will"           : "william",
            "willy"          : "william",
            "mandy"          : "amanda",
            "cathy"          : "catherine",
            "cath"           : "catherine",
            "chris"          : "christopher",
            "chrissy"        : "christine",
            "cindy: cynth"   : "cynthia",
            "debbie"         : "deborah",
            "deb"            : "deborah",
            "betty"          : "elizabeth",
            "beth"           : "elizabeth",
            "liz"            : "elizabeth",
            "bess"           : "elizabeth",
            "flo"            : "florence",
            "francie"        : "frances",
            "fran"           : "frances",
            "jan"            : "janet",
            "kate"           : "katherine",
            "kathy"          : "katherine",
            "jan"            : "janice",
            "nan"            : "nancy",
            "pam"            : "pamela",
            "pat"            : "patricia",
            "bobbie"         : "roberta",
            "sophie"         : "sophia",
            "sue"            : "susan",
            "suzie"          : "susan",
            "terry"          : "teresa",
            "val"            : "valerie",
            "ronnie"         : "veronica",
            "vonna"          : "yvonne",
            "peggy"          : "margaret",
            "ted"            : "edward",
            "sally"          : "sarah",
            "harry"          : "henry",
        }

    def tokenize_name(self, name):
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
            if word.replace('.', '') in self.STOPWORDS: continue
            if word in self.NICKNAME_DICT: word = self.NICKNAME_DICT[word]
            if word.count('.') > 1: word = ' '.join(word.split('.'))
            splitted_name.append(word)
        name = ' '.join(splitted_name)
        name = re.sub(' +', ' ', name.encode('ascii', 'ignore').decode('ascii'))
        return name

    def clean_name(self, name):
        name = unidecode(name)
        name = name.lower()
        new_name = ""
        for a in name:
            # print("1:", a)
            if a.isalpha():
                new_name += a
            else:
                new_name = new_name.strip()
                new_name += " "
            # print("2:", new_name)
        return new_name.strip()

    def get_name_uniq(self, name_c):
        name_rareness = 0.0
        if name_c:
            name_c = name_c.lower().split()
            for seg in name_c:
                s = self.name_uniq_dict.get(seg.strip(" "), 10)
                name_rareness += s
        return name_rareness

    # def get_coauthor_info(self, paper_attr, author_paper_attr_list)

    def process_data(self, total_ins):
        res = []
        # total_ratio = []
        # count = 0
        for each in tqdm(total_ins):
            # count += 1
            index, ins_res = self.atomic_process(each)
            res.append(ins_res)
            # print(count)
            # total_ratio.append(nor_ratio)
        return res

    def multi_process_data(self, total_ins):
        function = self.atomic_process
        print('num_processing:', num_processing)
        print('Please be patient while it is being processed...')
        pool = multiprocessing.Pool(num_processing)
        multi_res = pool.map(function, total_ins)  #multi_res：（ index, (res, coauthor_ratio) ）
        pool.close()
        pool.join()
        print('finish multiprocessing!')
        sorted_res = sorted(multi_res, key=itemgetter(0))
        re_res = []
        for each in sorted_res:
            _index, feas = each
            re_res.append(feas)  #(res, coauthor_ratio)
        return re_res

    def normalize(self, total_ratio):
        normal_ratio = 0.0
        total_ratio = np.array(total_ratio)
        # maxs = max(total_ratio)
        # mins = min(total_ratio)
        sort_ratios = sorted(total_ratio, reverse=True)
        if sort_ratios[0] != sort_ratios[-1]:
            normal_ratio = round((sort_ratios[0] - sort_ratios[1]) / (sort_ratios[0] - sort_ratios[-1] + 1e-8), 6)
        # print(sort_ratios)
        # print(normal_ratio)
        # exit()
        return normal_ratio
        # if(maxs)

    def atomic_process(self, each_ins):
        res = []
        total_ratio = []
        # print(len(each_ins))
        # exit()
        index, data = each_ins
        for ins in data:  #ins: (unassAttr, candiAttrList)
            feature, ratios = self.process_ranking_feature(ins)
            res.append(feature)
            total_ratio.append(ratios)

        coauthor_ratio = self.normalize(total_ratio)
        return index, (res, coauthor_ratio)

    def process_ranking_feature(self, ins):
        '''

        Args:
            ins: tuple(`paper_info`, List[`paper_info`]),
            List[`paper_info`] the characteristics of all the papers of an individual in the profile.
            paper_info: List[(set(coauthor name), org_str, venue, keywords_str, title)]

        Returns:

        '''
        paper_attr, author_paper_attr_list = ins
        features = []
        name2clean = {}
        paper_names, paper_org, paper_venue, paper_keywords, paper_title = paper_attr
        paper_names = list(paper_names)[:50]

        # Process names
        for each in paper_names:
            tmp_clean = self.clean_name(each)
            if each not in name2clean:
                name2clean[each] = tmp_clean

        candiauthor2int = defaultdict(int)
        candiorgs = []
        candivenues = []
        candititles = []
        candikeywords = []
        candiyears = []
        author_papers = len(author_paper_attr_list)

        filter_author_names = []
        # print("papers:", len(author_paper_attr_list))
        for each in author_paper_attr_list:
            each_names, each_org, each_venue, each_keywords, each_title = each
            # collect name
            each_names = list(each_names)[:50]
            for each in each_names:
                tmp_clean = self.clean_name(each)
                candiauthor2int[tmp_clean] += 1
                if each not in name2clean:
                    name2clean[each] = tmp_clean

            filter_author_names.append(each_names)
            # collect org
            if each_org != "":
                candiorgs.append(each_org)

            # collect venue
            if each_venue != "":
                candivenues.append(each_venue)

            # collect keywords
            # if(len(each_keywords) != 0):
            if each_keywords != "":
                candikeywords.append(each_keywords)

            # collect title
            if each_title != "":
                candititles.append(each_title)

        paper_coauthor_tfidf_ratio = .0
        authorkeys = list(candiauthor2int.keys())
        if (len(paper_names) == 0) or (len(authorkeys) == 0):
            # features.extend([-999] * 12)
            features.extend([0.0] * 4)
        else:
            coauthors = set()
            for each_names in filter_author_names:
                # each_names, each_org, each_venue, each_keywords, each_title = each
                each_coauthors = MatchName(paper_names, each_names, name2clean, True)
                coauthors = coauthors | each_coauthors
            coauthor_tfidf = 0.0
            counted_coauthor_tfidf = 0.0
            for each in coauthors:
                name_score = self.get_name_uniq(each)
                coauthor_tfidf += name_score
                counted_coauthor_tfidf += candiauthor2int.get(each, 1) * name_score

            paper_tfidf = 0.0
            for each in paper_names:
                name_score = self.get_name_uniq(name2clean[each])
                paper_tfidf += name_score

            author_tfidf = 0.0
            for each, count in candiauthor2int.items():
                name_score = self.get_name_uniq(each)
                author_tfidf += name_score * count

            paper_coauthor_tfidf_ratio = round(coauthor_tfidf / (paper_tfidf + 1e-8), 6)
            author_coauthor_tfidf_ratio = round(counted_coauthor_tfidf / (author_tfidf + 1e-8), 6)

            features.extend(
                [coauthor_tfidf, paper_coauthor_tfidf_ratio, counted_coauthor_tfidf, author_coauthor_tfidf_ratio])

        # other fearues 8
        features.extend(self.other_features(paper_org, candiorgs, 14.37))

        features.extend(self.other_features(paper_venue, candivenues, 10.42))

        features.extend(self.other_features(paper_title, candititles, 14.79))

        features.extend(self.other_features(paper_keywords, candikeywords))

        # features.extend(self.year_features(paper_year, candiyears))
        # 4 + 8 + 8 + 8 + 8 = 36
        # print(len(features))
        # exit()
        return features, paper_coauthor_tfidf_ratio

    def other_features(self, paper_attr, author_attr_list, default_value=1):
        feature_list = []
        paper_attr = ' '.join(re.sub(r'[\W_]', ' ', paper_attr).split())
        # print(author_attr_list)
        author_attr_list = [" ".join(re.sub(r'[\W_]', ' ', item).split()) for item in author_attr_list]
        candi_string = ' '.join(author_attr_list)
        if (paper_attr.strip() != "") and (candi_string.strip() != ""):
            paper_attr_list = paper_attr.strip().lower().split()
            paper_attr_set = set(paper_attr_list)
            jaro_scores = []
            card_scores = []
            count_total = 0
            count_common = 0
            for item in author_attr_list:
                if (item != ""):
                    count_total += 1
                    jaros = distance.get_jaro_distance(paper_attr, item)
                    item_set = set(item.split())
                    cards = len(item_set & paper_attr_set) / len(paper_attr_set | item_set)
                    if (jaros == 1.) or (cards == 1.):
                        count_common += 1
                    jaro_scores.append(jaros)
                    card_scores.append(cards)
            # print('111:', jaro_scores)
            if (len(jaro_scores) == 0):
                jaro_scores = [0.]
            max_jaro_score = np.max(jaro_scores)
            min_jaro_score = np.min(jaro_scores)
            mean_jaro_score = np.mean(jaro_scores)

            if (len(card_scores) == 0):
                card_scores = [0.]
            max_card_score = np.max(card_scores)
            min_card_score = np.min(card_scores)
            mean_card_score = np.mean(card_scores)

            word_count_paper = defaultdict(int)
            for word in paper_attr_list:
                word_count_paper[word] += 1

            word_count_author = defaultdict(int)
            author_list = candi_string.strip().lower().split()
            for word in author_list:
                word_count_author[word] += 1

            inter_words = set(word_count_paper.keys()) & set(word_count_author.keys())
            score_paper = 0.0
            score_author = 0.0
            for inter in inter_words:
                score = self.org_tfidf.get(inter, default_value)
                score_paper += score * word_count_paper[inter]
                score_author += score * word_count_author[inter]

            total_score_paper = 0
            for word, count in word_count_paper.items():
                score = self.org_tfidf.get(word, default_value)
                total_score_paper += score * count

            total_score_author = 0
            for word, count in word_count_author.items():
                score = self.org_tfidf.get(word, default_value)
                total_score_author += score * count

            word_paper_ratio = round(score_paper / (total_score_paper + 1e-8), 6)
            word_author_ratio = round(score_author / (total_score_author + 1e-8), 6)

            feature_list.extend(
                [max_jaro_score, mean_jaro_score, max_card_score, mean_card_score,
                 score_paper, word_paper_ratio, score_author, word_author_ratio])
        else:
            feature_list.extend([0.0] * 8)
        return feature_list
