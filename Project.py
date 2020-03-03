import json
import os
import subprocess
# subprocess.call('dir', shell=True)
# subprocess.call(['cmd', '/c', 'dir'])\
import numpy as np
# import matplotlib.pyplot as plt
# from igraph import *
# import cairo
# from collections import defaultdict
# import networkx as nx
#-------------------------------------------------------------------------------------------
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer
Stemmer = PorterStemmer()
from nltk.stem.lancaster import  LancasterStemmer
Stemmer2 = LancasterStemmer()
stopwords = set(stopwords.words())
#-------------------------------------------------------------------------------------------
import pandas as pd
import timeit
#-------------------------------------------------------------------------------------------
start = timeit.default_timer()
#-------------------------------------------------------------------------------------------
from stanfordcorenlp import StanfordCoreNLP
# from nltk.parse.corenlp import CoreNLPParser
nlp = StanfordCoreNLP(os.path.join(os.getcwd(),'stanford-corenlp-full-2018-10-05'))
#-------------------------------------------------------------------------------------------
#sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
#print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# props = {'annotators': 'tokenize,depparse'}
# print(nlp.annotate(sentence, properties=props))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', nlp.dependency_parse(sentence))
#-------------------------------------------------------------------------------------------




class Data:
    def __init__(self, csvFile, ColName):
        self.numOfArticles=2
        my_csv = pd.read_csv(csvFile, nrows=self.numOfArticles)
        self.DataSet = tuple(my_csv[ColName])
        self.ProcessedDict = []
        self.SSplitDataSet = []
        self.cleanSSplitDataSet = []
        self.allWords = []

    def makeDictList(self):

        for article in self.SSplitDataSet:
            processedDict = {}
            wordsInArticle = []
            for sentence in article:

                processedSentence = self.processedSentence(sentence)
                #print(processedSentence)
                for word in processedSentence:
                    wordsInArticle.append(word)   # for all WORDS!
                    if processedSentence.index(word)+1 < len(processedSentence):
                        nextWord = processedSentence[processedSentence.index(word)+1]
                        key = tuple([word,nextWord])
                        if key in processedDict.keys():
                            processedDict[key][0] += 1
                            processedDict[key][1].append(article.index(sentence))
                        else:
                            processedDict.update({key: [1,[article.index(sentence)]]})
            self.allWords.append(set(wordsInArticle))
            self.ProcessedDict.append(processedDict)
    def processedSentence(self,sentence):

        sentence = sentence.split()
        newSentence = []
        for word in sentence:
            word = word.lower()
            if word not in stopwords:
                    if len(word) != 2:
                        word = Stemmer.stem(word)
                        newSentence.append(word)
        return newSentence

    def SentenceSplit(self):
        props = {'annotators': 'tokenize,ssplit', 'outputFormat': 'json'}
        ProcessedDataSet = list(self.DataSet)
        for article in ProcessedDataSet:

            article = json.loads(nlp.annotate(article, properties=props))
            sentenceList = []
            for indexOfSentence in range(len(article['sentences'])):
                sentence = ''
                for indexOfTokens in range(len(article['sentences'][indexOfSentence]['tokens'])):
                    sentence +=' '+ article['sentences'][indexOfSentence]['tokens'][indexOfTokens]['word']
                sentenceList.append(sentence)
            self.SSplitDataSet.append(sentenceList)
    def cleanSentenceSplit(self):
        props = {'annotators': 'tokenize,ssplit', 'outputFormat': 'json'}
        ProcessedDataSet = list(self.DataSet)
        for article in ProcessedDataSet:

            article = json.loads(nlp.annotate(article, properties=props))
            sentenceList = []
            for indexOfSentence in range(len(article['sentences'])):
                sentence = []
                for indexOfTokens in range(len(article['sentences'][indexOfSentence]['tokens'])):
                    word = article['sentences'][indexOfSentence]['tokens'][indexOfTokens]['word']
                    word = word.lower()
                    if word not in stopwords:
                        if len(word) != 2:
                            word = Stemmer.stem(word)
                            sentence.append(word)
                sentenceList.append(sentence)
            self.cleanSSplitDataSet.append(sentenceList)
#------------------------------------------------------------------------------------------------------------------
def build_index(allWords):
    wordsList = allWords

    return {words: [index] for index, words in enumerate(wordsList)}

def build_transition_matrix(processedDict, wordsIndex, numOfSentences):
    matrix = np.zeros((len(wordsIndex), len(wordsIndex)))

    for word in wordsIndex:

        for secondWord in wordsIndex:

            if word == secondWord:
                matrix[wordsIndex[word][0]][wordsIndex[secondWord][0]] = 0
            else:
                key = tuple([word,secondWord])
                if key in processedDict:
                    matrix[wordsIndex[word][0]][wordsIndex[secondWord][0]] = (processedDict[key][0]/numOfSentences)
                else:
                    matrix[wordsIndex[word][0]][wordsIndex[secondWord][0]] = 0
    return matrix
def page_rank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A))

    while True:
        new_P = (0.8*np.dot(P,A.T)+0.2)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P
def updateResult(words,results):
    index = 0
    for key in words.keys():
        words[key].append(results[index])
        index += 1
    return words

def loadData(FileProcess):
    print("LOADING DATA......")
    FileProcess.SentenceSplit()
    FileProcess.makeDictList()
    FileProcess.cleanSentenceSplit()
    for num in range(FileProcess.numOfArticles):
        wordsIndex=build_index(FileProcess.allWords[num])
        M = build_transition_matrix(FileProcess.ProcessedDict[num], wordsIndex, len(FileProcess.SSplitDataSet[num]))
        result = page_rank(M)
        wordsIndex = updateResult(wordsIndex, result)
        print(wordsIndex)
        myvalue=result.argmax()
        bestword=([word for (word,value) in wordsIndex.items() if value[0]==myvalue])
        print(bestword)
        maxindex = 0
        max = 0
        for sIndex in range(len(FileProcess.cleanSSplitDataSet[num])):
            sum = 0
            for word in FileProcess.cleanSSplitDataSet[num][sIndex]:
                sum += wordsIndex[word][1]
                # if len(FileProcess.cleanSSplitDataSet[num][sIndex])>0:
                sum /= len(FileProcess.cleanSSplitDataSet[num][sIndex])
            if sum >= max:
                max = sum
                maxindex = sIndex
        print("article number: ", num, " Title: ", FileProcess.SSplitDataSet[num][maxindex])

#-------------------------------------------------------------------------------------------
FileProcess = Data('articles1.csv', 'content')

FileProcess.SentenceSplit()
FileProcess.makeDictList()
FileProcess.cleanSentenceSplit()
# loadData(FileProcess)
#
# #=========================================================================================================
print(FileProcess.SSplitDataSet[0])
print(FileProcess.ProcessedDict[0])
print(FileProcess.allWords[0])
print(FileProcess.cleanSSplitDataSet[0])

wordsIndex = build_index(FileProcess.allWords[0])
print(wordsIndex)


M=build_transition_matrix(FileProcess.ProcessedDict[0], wordsIndex, len(FileProcess.SSplitDataSet[0]))
result=page_rank(M)


wordsIndex = updateResult(wordsIndex, result)
print(wordsIndex)
maxindex=0
max=0
for sIndex in range(len(FileProcess.cleanSSplitDataSet[0])):
    sum=0

    for word in FileProcess.cleanSSplitDataSet[0][sIndex]:
        sum +=wordsIndex[word][1]
    sum /= len(FileProcess.cleanSSplitDataSet[0][sIndex])
    if sum > max:
        max = sum
        maxindex = sIndex
print("index: ", maxindex," maxSum: ",max)
print(FileProcess.SSplitDataSet[0][maxindex])


# ==================================================================================================================


stop = timeit.default_timer()
print('Time:', stop - start)
