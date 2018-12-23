import nltk
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures
import operator
import string
import math

measures = BigramAssocMeasures()

l_ru = []
with open("text_ru.txt", 'r', encoding="utf-8") as f:
	for line in f:
		for w in nltk.word_tokenize(line.lower()):
			if w not in string.punctuation:
				l_ru.append(w)

l_en = []
with open("text_en.txt", 'r', encoding="utf-8") as f:
	for line in f:
		for w in nltk.word_tokenize(line.lower()):
			if w not in string.punctuation:
				l_en.append(w)

freq_ru = nltk.FreqDist(l_ru)
sort_fr_ru = freq_ru.most_common()
finder_ru = BigramCollocationFinder.from_words(l_ru)
t_ru = finder_ru.nbest(measures.student_t, 100)

freq_en = nltk.FreqDist(l_en)
sort_fr_en = freq_en.most_common()
finder_en = BigramCollocationFinder.from_words(l_en)
t_en = finder_en.nbest(measures.student_t, 100)

with open("collocations_ru.csv", 'w', encoding="utf-8") as coll:
    for i in t_ru:
        coll.write("{}; {}; {}\n".format("t", i[0]+" "+i[1], round(finder_ru.score_ngram(measures.student_t, i[0], i[1]),2)))
    for m in t_ru:
        coll.write("{}; {}; {}\n".format("chi^2", m[0]+" "+m[1], round(finder_ru.score_ngram(measures.chi_sq, m[0], m[1]),2)))
    for n in t_ru:
        coll.write("{}; {}; {}\n".format("log-likelihood", n[0]+" "+n[1], round(finder_ru.score_ngram(measures.likelihood_ratio, n[0], n[1]),2)))
    for q in t_ru:
        coll.write("{}; {}; {}\n".format("pmi", q[0]+" "+q[1], round(finder_ru.score_ngram(measures.pmi, q[0], q[1]),2)))

with open("collocations_en.csv", 'w', encoding="utf-8") as coll:
    for i in t_en:
        coll.write("{}; {}; {}\n".format("t", i[0]+" "+i[1], round(finder_en.score_ngram(measures.student_t, i[0], i[1]),2)))
    for m in t_en:
        coll.write("{}; {}; {}\n".format("chi^2", m[0]+" "+m[1], round(finder_en.score_ngram(measures.chi_sq, m[0], m[1]),2)))
    for n in t_en:
        coll.write("{}; {}; {}\n".format("log-likelihood", n[0]+" "+n[1], round(finder_en.score_ngram(measures.likelihood_ratio, n[0], n[1]),2)))
    for q in t_en:
        coll.write("{}; {}; {}\n".format("pmi", q[0]+" "+q[1], round(finder_en.score_ngram(measures.pmi, q[0], q[1]),2)))
