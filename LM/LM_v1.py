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
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):
                sentenceCount += 1
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
        print(self.name + ".conll contains " + str(self.total_word_count) + " words")

class linear_model:
    def __init__(self):
        self.feature = dict()
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.tags = dict()
        self.tags_length = 0
        self.v = []
        self.update_times =[]
        self.w = []
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        #self.train.open_file("./data/train.conll")
        #self.train.open_file("zy.txt")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        #self.dev.open_file("./data/dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
    
    def create_feature(self, sentence, pos):
        word_count = len(sentence.word)
        wi = sentence.word[pos]
        pos_word_len = len(sentence.word[pos])
        if(pos == 0):
            wi_left_word = "$$"
            wi_left_word_last_c = "$"
        else:
            wi_left_word = sentence.word[pos-1]
            wi_left_word_last_c = sentence.wordchars[pos-1][len(sentence.word[pos-1])-1]
        if(pos == word_count-1):
            wi_right_word = "##"
            wi_right_word_first_c = "#"
        else:
            wi_right_word = sentence.word[pos+1]
            wi_right_word_first_c = sentence.wordchars[pos+1][0]
        wi_last_c = sentence.wordchars[pos][pos_word_len - 1]
        wi_first_c = sentence.wordchars[pos][0]
        f = []
        f.append("02:" + wi)
        f.append("03:" + wi_left_word)
        f.append("04:" + wi_right_word)
        f.append("05:" + wi + '*' + wi_left_word_last_c)
        f.append("06:" + wi + '*' + wi_right_word_first_c)
        f.append("07:" + wi_first_c)
        f.append("08:" + wi_last_c)
        for i in range(1, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            f.append("09:" + wi_kth_c)
            f.append("10:" + wi_first_c + "*" + wi_kth_c)
            f.append("11:" + wi_last_c + "*" + wi_kth_c)
        for i in range(0, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            wi_kth_next_c = sentence.wordchars[pos][i + 1]
            if(wi_kth_c == wi_kth_next_c):
                f.append("13:" + wi_kth_c + "*" + "consecutive")
        if(pos_word_len == 1):
            f.append("12:" + wi + "*" + wi_left_word_last_c + "*" + wi_right_word_first_c)
        for i in range(0, pos_word_len):
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
        self.feature_length = len(self.feature)
        self.tags_length = len(self.tags)
        self.w = [0]*(self.feature_length * self.tags_length)
        self.v = [0]*(self.feature_length * self.tags_length)
        self.update_times = [0]*(self.feature_length * self.tags_length)
        self.feature_keys = list(self.feature.keys())
        self.feature_values = list(self.feature.values())
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tags_length))
        #输出特征
        #zy = open("zy_feature.txt", mode = 'w')
        #for f in self.feature:
            #zy.write(f.encode('utf-8')+'\n')

    def dot(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.w[offset + f]
        return score

    def get_feature_id(self, fv):
        fv_id = []
        for feature in fv:
            if(feature in self.feature):
                fv_id.append(self.feature[feature])
        return fv_id;
	
    def max_tag(self, sentence, pos):
        maxscore = -1
        tempscore = 0
        tag = "NULL"
        fv = self.create_feature(sentence, pos)
        fv_id = self.get_feature_id(fv)
        for t in self.tags:
            tempscore = self.dot(fv_id, self.feature_length * self.tags[t])
            if(tempscore >= maxscore):
                maxscore = tempscore
                tag = t
        return tag

    def dot_v(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.v[offset + f]
        return score

    def max_tag_v(self, sentence, pos):
        maxscore = -1
        tempscore = 0
        tag = "NULL"
        fv = self.create_feature(sentence, pos)
        fv_id = self.get_feature_id(fv)
        for t in self.tags:
            tempscore = self.dot_v(fv_id, self.feature_length * self.tags[t])
            if(tempscore >= maxscore):
                maxscore = tempscore
                tag = t
        return tag

    def update_v(self, index, update_times, last_w_value):		
        last_update_times = self.update_times[index]    #上一次更新所在的次数
        current_update_times = update_times    #本次更新所在的次数
        self.update_times[index] = update_times
        self.v[index] += (current_update_times - last_update_times -1 )*last_w_value + self.w[index]

    def update_weight(self, s, p, max_tag, correct_tag, update_times):
        f = self.create_feature(s, p)
        f_id = self.get_feature_id(f)
        maxtag_id = self.tags[max_tag]
        correcttag_id = self.tags[correct_tag]
        for i in f_id:
            offset = self.feature_length * maxtag_id
            index = offset + i
            last_w_value = self.w[index]    #更新前的权重
            self.w[index] -= 1
            self.update_v(index, update_times, last_w_value)
        for i in f_id:
            offset = self.feature_length * correcttag_id
            index = offset + i
            last_w_value = self.w[index]    #更新前的权重
            self.w[index] += 1
            self.update_v(index, update_times, last_w_value)

    def online_training(self):
        max_train_precision = 0.0
        max_dev_precision = 0.0
        update_times = 0
        word_count = self.train.total_word_count
        for iterator in range(0, 20):
            print("iterator " + str(iterator))
            for s in self.train.sentences:
                for p in range(0, len(s.word)):
                    max_tag = self.max_tag(s, p)
                    correct_tag = s.tag[p]
                    if(max_tag != correct_tag):
                        update_times += 1
                        self.update_weight(s, p, max_tag, correct_tag, update_times)
            #本次迭代完成
            current_update_times = update_times    #本次更新所在的次数
            for i in range(len(self.v)):
                last_w_value = self.w[i]
                last_update_times = self.update_times[i]    #上一次更新所在的次数
                if(current_update_times != last_update_times):
                    self.update_times[i] = current_update_times
                    self.v[i] += (current_update_times - last_update_times - 1)*last_w_value + self.w[i]
                    
            self.save_model(iterator)
            #进行评估
            #train_iterator, train_c, train_count, train_precision = self.evaluate(self.train, iterator)
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
            #保存概率最大的情况
            #if(train_precision > (max_train_precision + 1e-10)):
                #max_train_precision = train_precision
                #max_train_iterator = train_iterator
                #max_train_c = train_c
                #max_train_count = train_count
            if(dev_precision > (max_dev_precision + 1e-10)):
                max_dev_precision = dev_precision
                max_dev_iterator = dev_iterator
                max_dev_c = dev_c
                max_dev_count  = dev_count
        print("Conclusion:")
        #print("\t"+self.train.name + " iterator: "+str(max_train_iterator)+"\t"+str(max_train_c)+" / "+str(max_train_count) + " = " +str(max_train_precision))
        print("\t"+self.dev.name + " iterator: "+str(max_dev_iterator)+"\t"+str(max_dev_c)+" / "+str(max_dev_count) + " = " +str(max_dev_precision))

    def save_model(self, iterator):
        fmodel = open("linearmodel.lm"+str(iterator), mode='w')
        for feature_id in self.feature_values:
            feature = self.feature_keys[feature_id]
            left_feature = feature.split(':')[0] + ':'
            right_feature = '*' + feature.split(':')[1]
            for tag in self.tags:
                tag_id = self.tags[tag]
                entire_feature = left_feature + tag + right_feature
                w = self.w[tag_id * self.feature_length + feature_id]
                if(w != 0):
                    fmodel.write(entire_feature.encode('utf-8') + '\t' + str(w) + '\n')
        fmodel.close()

    def evaluate(self, dataset, iterator):
       c = 0
       count = 0
       fout = open(dataset.name+".out" + str(iterator), mode='w')
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
