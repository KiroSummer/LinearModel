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
        self.total_word_count = 0
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\n'):
                self.sentences.append(sen)
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
        self.total_word_count = wordCount
        print(self.name + ".conll contains " + str(len(self.sentences)) + " sentences")
        print(self.name + ".conll contains " + str(self.total_word_count) +" words")

class linear_model:
    def __init__(self):
        self.feature = dict()
        self.tags = dict()
        self.matrix_v = []
        self.matrix_update_times =[]
        self.matrix_model = []
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
    
    def create_feature(self, sentence, pos):
        word_count = len(sentence.word)
        wi = sentence.word[pos]
        pos_word_len = len(sentence.word[pos])
        if(pos == 0):
            wim1 = "START"
            cim1m1 = "T"
        else:
            wim1 = sentence.word[pos-1]
            cim1m1 = sentence.wordchars[pos-1][len(sentence.word[pos-1])-1]
        if(pos == word_count-1):
            wip1 = "END"
            cip10 = "E"
        else:
            wip1 = sentence.word[pos+1]
            cip10 = sentence.wordchars[pos+1][0]
        cim1 = sentence.wordchars[pos][pos_word_len - 1]
        ci0 = sentence.wordchars[pos][0]
        f = []
        f.append("02:" + wi)
        f.append("03:" + wim1)
        f.append("04:" + wip1)
        f.append("05:" + cim1m1)
        f.append("06:" + cip10)
        f.append("07:" + ci0)
        f.append("08:" + cim1)
        for i in range(1, pos_word_len - 2):
            cik = sentence.wordchars[pos][i]
            f.append("09:" + cik)
            f.append("10:" + ci0 + "*" + cik)
            f.append("11:" + cim1 + "*" + cik)
            cikp1 = sentence.wordchars[pos][i + 1]
            if(cik == cikp1):
                f.append("12:" + cik + "*" + "consecutive")
        if(pos_word_len == 1):
            f.append("13:" + wi + "*" + cim1m1 + "*" + cip10)
        for i in range(0, pos_word_len - 1):
            if(i >= 4):
                break
            f.append("14:" + sentence.word[pos][0:(i + 1)])
            f.append("15:" + sentence.word[pos][-(i + 1)::])
        return f

    def create_feature_space(self):
        feature_index = 0
        tag_index = 0
        for s in self.train.sentences:
            for p in range(0, len(s.word)):
                f = self.create_feature(s, p)
                for feature in f:
                    if (feature in self.feature):
                        pass
                    else:
                        self.feature[feature] = feature_index
                        feature_index += 1
                if(s.tag[p] in self.tags):
                    pass
                else:
                    self.tags[s.tag[p]] = tag_index
                    tag_index += 1
        self.matrix_model = [[0 for col in range(len(self.tags))] for row in range(len(self.feature))]
        self.matrix_v = [[0 for col in range(len(self.tags))] for row in range(len(self.feature))]
        self.matrix_update_times = [[0 for col in range(len(self.tags))] for row in range(len(self.feature))]
        print("the total number of features is "+str(len(self.feature)))
        print("the total number of tags is "+str(len(self.tags)))

    def dot(self, f, tag):
        score = 0
        tag_index = self.tags[tag]
        for feature in f:
            if(feature in self.feature):
                feature_index = self.feature[feature]
                score += self.matrix_model[feature_index][tag_index]
        return score

    def max_tag(self, sentence, pos):
        maxnum = -1e10
        tempnum = 0
        tag = "NULL"
        for t in self.tags:
            fv = self.create_feature(sentence, pos)
            tempnum = self.dot(fv, t)
            if(tempnum > (maxnum + 1e-10)):
                maxnum = tempnum
                tag = t
        return tag

    def dot_v(self, f, tag):
        score = 0
        tag_index = self.tags[tag]
        for feature in f:
            if(feature in self.feature):
                feature_index = self.feature[feature]
                score += self.matrix_v[feature_index][tag_index]
        return score

    def max_tag_v(self, sentence, pos):
        maxnum = -1e10
        tempnum = 0
        tag = "NULL"
        for t in self.tags:
            fv = self.create_feature(sentence, pos)
            tempnum = self.dot_v(fv, t)
            if(tempnum > (maxnum + 1e-10)):
                maxnum = tempnum
                tag = t
        return tag

    def online_training(self):
        max_train_precision = 0
        max_dev_precision = 0
        update_times = 0
        word_count = self.train.total_word_count
        for iterator in range(0, 20):
            print("iterator " + str(iterator))
            times = 0
            for s in self.train.sentences:
                for p in range(0, len(s.word)):
                    times += 1
                    max_tag = self.max_tag(s, p)
                    correcttag = s.tag[p]
                    if(max_tag != correcttag):
                        update_times += 1
                        fmaxtag = self.create_feature(s, p)
                        fcorrecttag = self.create_feature(s, p)
                        maxtag_index = self.tags[max_tag]
                        correcttag_index = self.tags[correcttag]
                        for i in fmaxtag:
                            if(i in self.feature):
                                feature_index = self.feature[i]
                                last_v_value = self.matrix_model[feature_index][maxtag_index]
                                self.matrix_model[feature_index][maxtag_index] -= 1
                                last_update_times = self.matrix_update_times[feature_index][maxtag_index]    #上一次更新所在的次数
                                current_update_times = update_times    #本次更新所在的次数
                                self.matrix_update_times[feature_index][maxtag_index] = update_times
                                self.matrix_v[feature_index][maxtag_index] += (current_update_times - last_update_times -1 )*last_v_value + self.matrix_model[feature_index][maxtag_index]
                        for i in fcorrecttag:
                            if(i in self.feature):
                                feature_index = self.feature[i]
                                last_v_value = self.matrix_model[feature_index][maxtag_index]
                                self.matrix_model[feature_index][correcttag_index] += 1
                                last_update_times = self.matrix_update_times[feature_index][maxtag_index]    #上一次更新所在的次数
                                current_update_times = update_times    #本次更新所在的次数
                                self.matrix_update_times[feature_index][maxtag_index] = update_times
                                self.matrix_v[feature_index][maxtag_index] += (current_update_times - last_update_times - 1)*last_v_value + self.matrix_model[feature_index][maxtag_index]
            #本次迭代完成
            current_update_times = update_times    #本次更新所在的次数
            for row in range(len(self.matrix_v)):
                for col in range(len(self.matrix_v[row])):
                    last_v_value = self.matrix_model[row][col]
                    last_update_times = self.matrix_update_times[row][col]    #上一次更新所在的次数
                    if(current_update_times != last_update_times):
                        self.matrix_update_times[row][col] = current_update_times
                        self.matrix_v[row][col] += (current_update_times - last_update_times - 1)*last_v_value + self.matrix_model[row][col]
                    
            self.save_model(iterator)
            train_iterator, train_c, train_count, train_precision = self.evaluate(self.train, iterator)
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
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
        fmodel = open("linearmodel.lm"+str(iterator), mode='w')
        for feature in self.feature:
            feature_index = self.feature[feature]
            for tag in self.tags:
                tag_index = self.tags[tag]
                if(self.matrix_model[feature_index][tag_index] != 0):
                    entire_feature = feature.split(':')[0] + ":" + tag + "*" + feature.split(':')[1]
                    fmodel.write(entire_feature.encode('utf-8') + '\t' + str(self.matrix_model[feature_index][tag_index]) + '\n')
        fmodel.close()

    def evaluate(self, dataset, iterator):
       c = 0
       count = 0
       fout = open(dataset.name + ".out" + str(iterator), mode='w')
       for s in dataset.sentences:
           for p in range(0, len(s.word)):
               count += 1
               max_tag = self.max_tag_v(s, p)
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
    print("executing time is "+str((endtime-starttime).seconds)+" s")
