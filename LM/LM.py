#!/usr/bin/python
#coding=utf-8
import datetime

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode='r')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        sentenceCount = 0
        wordCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\n'):
                self.sentences.append(sen)
                sentenceCount += 1
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1].decode('utf-8')
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        print(self.name + ".conll contains " + str(sentenceCount) + " sentences")
        print(self.name + ".conll contains " + str(wordCount) + " words")

class linear_model:
    def __init__(self):
        self.model = dict()
        self.tags = dict()
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
    
    def create_feature_with_tag(self, sentence, pos, tag):
        word_count = len(sentence.word)
        wi = sentence.word[pos]
        pos_word_len = len(sentence.word[pos])
        if(pos == 0):
            wim1 = "START"
            cim1m1 = "T"
        else:
            wim1 = sentence.word[pos-1]
            cim1m1 = sentence.wordchars[pos-1][len(sentence.word[pos-1])-1]
        if(pos == word_count - 1):
            wip1 = "END"
            cip10 = "E"
        else:
            wip1 = sentence.word[pos + 1]
            cip10 = sentence.wordchars[pos + 1][0]
        cim1 = sentence.wordchars[pos][pos_word_len - 1]
        ci0 = sentence.wordchars[pos][0]
        f = []
        f.append("02:" + str(tag) + "*" + wi)
        f.append("03:" + str(tag) + "*" + wim1)
        f.append("04:" + str(tag) + "*" + wip1)
        f.append("05:" + str(tag) + "*" + cim1m1)
        f.append("06:" + str(tag) + "*" + cip10)
        f.append("07:" + str(tag) + "*" + ci0)
        f.append("08:" + str(tag) + "*" + cim1)
        for i in range(1, pos_word_len - 2):
            cik = sentence.wordchars[pos][i]
            f.append("09:" + str(tag) + "*" + cik)
            f.append("10:" + str(tag) + "*" + ci0 + "*" + cik)
            f.append("11:" + str(tag) + "*" + cim1 + "*" + cik)
            cikp1 = sentence.wordchars[pos][i + 1]
            if(cik == cikp1):
                f.append("13:" + str(tag) + "*" + cik + "*" + "consecutive")
        if(pos_word_len == 1):
            f.append("12:" + str(tag) + "*" + wi + "*" + cim1m1 + "*" + cip10)
        for i in range(0, pos_word_len - 1):
            if(i >= 4):
                break
            f.append("14:" + str(tag) + "*" + sentence.word[pos][0:(i + 1)])
            f.append("14:" + str(tag) + "*" + sentence.word[pos][-(i + 1)::])
        return f

    def create_feature_space(self):
        for s in self.train.sentences:
            for p in range(0, len(s.word)):
                f = self.create_feature_with_tag(s, p, s.tag[p])
                for feature in f:
                    self.model[feature] = 0
                if(s.tag[p] in self.tags):
                    self.tags[s.tag[p]] += 1
                else:
                    self.tags[s.tag[p]] = 0
        print("the total number of features is " + str(len(self.model)))
        print("the total number of tags is " + str(len(self.tags)))

    def dot(self,f):
        score = 0
        for i in f:
            if(i in self.model):
                score += self.model[i] 
        return score

    def max_tag(self, sentence, pos):
        maxnum = -1e10
        tempnum = 0
        tag = "NULL"
        for t in self.tags:
            fv = self.create_feature_with_tag(sentence, pos, t)
            tempnum = self.dot(fv)
            if(tempnum > (maxnum + 1e-10)):
                maxnum = tempnum
                tag = t
        return tag

    def online_training(self):
        max_train_precision = 0
        max_dev_precision = 0
        for iterator in range(0, 20):
            print("iterator " + str(iterator))
            wordCount = 0
            for s in self.train.sentences:
                for p in range(0, len(s.word)):
                    max_tag = self.max_tag(s, p)
                    correcttag = s.tag[p]
                    if(max_tag != correcttag):
                        fmaxtag = self.create_feature_with_tag(s, p, max_tag)
                        fcorrecttag = self.create_feature_with_tag(s, p, correcttag)
                        for i in fmaxtag:
                            if(i in self.model):
                                self.model[i] -= 1
                        for i in fcorrecttag:
                            if(i in self.model):
                                self.model[i] += 1
            train_iterator, train_c, train_count, train_precision = self.evaluate(self.train, iterator)
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
            self.save_model(iterator)
            if(train_precision > (max_train_precision + 1e-10)):
                max_train_precision = train_precision
                max_train_iterator = train_iterator
                max_train_c = train_c
                max_train_count = train_count
            if(dev_precision > (max_dev_precision + 1e-10)):
                max_dev_precision = dev_precision
                max_dev_iterator = dev_iterator
                max_dev_c = dev_c
                max_dev_count  = dev_count
        print("Conclusion:")
        print("\t"+self.train.name + " iterator: " + str(max_train_iterator) + "\t" + str(max_train_c) + " / " + str(max_train_count) + " = " + str(max_train_precision))
        print("\t"+self.dev.name + " iterator: " + str(max_dev_iterator) + "\t" + str(max_dev_c) + " / " + str(max_dev_count) + " = " + str(max_dev_precision))

    def save_model(self, iterator):
        fmodel = open("linearmodel.lm" + str(iterator), mode='w')
        for key in self.model:
            fmodel.write(key.encode('utf-8') + "\t" + str(self.model[key]) + '\n')
        fmodel.close()

    def evaluate(self, dataset, iterator):
       c = 0
       count = 0
       fout = open(dataset.name + ".out" + str(iterator), mode='w')
       for s in dataset.sentences:
           for p in range(0, len(s.word)):
               count += 1
               max_tag = self.max_tag(s, p)
               correcttag = s.tag[p]
               fout.write(s.word[p].encode('utf-8') + '\t' + str(max_tag) + '\t' + str(correcttag) + '\n')
               if(max_tag != correcttag):
                   pass
               else:
                   c += 1
       print(dataset.name + "\tprecision is " + str(c) + " / " + str(count) + " = " + str(1.0 * c/count))
       fout.close()
       return iterator, c, count, 1.0 * c/count


################################ main #####################################
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    lm = linear_model()
    lm.create_feature_space()
    lm.online_training()
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
