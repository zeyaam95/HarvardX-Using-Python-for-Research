import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

bookdir = "./Books"


def word_count(text):
    text = text.lower()
    skips = [".", '"', "'", ";", ",", ":"]
    for ch in skips:
        text.replace("ch", "")
    word_counts = {}
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
        return word_counts


def word_cf(text):
    text = text.lower()
    skips = [".", '"', "'", ";", ",", ":"]
    for ch in skips:
        text.replace("ch", "")
        word_counts = Counter(text.split(" "))
        return word_counts


def read_book(path):
    with open(path, "r", encoding='utf-8') as current_f:
        text = current_f.read()
        text = text.replace('\n', "").replace('\r', "")
    return text


def word_stats(word_counts):
    unique = len(word_counts)
    counts = word_counts.values()
    return (unique, counts)


def word_count_distribution(text):
    words_count = word_cf(text)
    count_distribution = Counter(words_count.values())
    return count_distribution


def more_frequent(distribution):
    counts = list(distribution.keys())
    frequency_of_counts = list(distribution.values())
    summ = np.cumsum(frequency_of_counts)
    #tot = sum(distribution.values())
    #lis = []
    # for e in summ:
    #    lis.append(e)
    #del lis[0]
    # lis.append(0)
    maxi = max(summ)
    count = 0
    distri = {}
    for val in counts:
        distri[val] = 1 - (summ[count] / maxi)
        count += 1
    print(summ, '\n', distri)
    return distri


table = pd.DataFrame(columns=("Language", "Author", "Title", "Length", "Unique"))
title_num = 1
for lang in os.listdir(bookdir):
    for auth in os.listdir(bookdir + "/" + lang):
        for title in os.listdir(bookdir + "/" + lang + "/" + auth):
            inputfile = bookdir + "/" + lang + "/" + auth + "/" + title
            inputfile = inputfile.encode('utf-8')
            print(inputfile)
            text = read_book(inputfile)
            (uniq, count) = word_stats(word_cf(text))
            table.loc[title_num] = lang, auth.capitalize(), title.replace(".txt", ""), sum(count), uniq
            title_num += 1

print(more_frequent(word_count_distribution(text)))
print('\n\n\n\n\n', word_count_distribution(text))
