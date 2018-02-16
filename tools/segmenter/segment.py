#!/usr/bin/python
# coding=utf8
# 
# Author: Wang Pidong
# Date: 2012-02-08
# Description: rewriting the lowjk-segmenter into Python
#              Now we only need three files to do Chinese word segmentation:
#              (1) this python script; (2) a Maximum Entropy model file (text format, GBK encoding);
#              (3) the Chinese lexicon (in GBK encoding).
#              The input is from the standard input (UTF-8 encoding Chinese sentences).
#              The output is printed to the standard output.

import os
import sys
import re
import math
import types
import time
import threading
from multiprocessing import Pool
from optparse import OptionParser

### setup the arguments
default_MODEL_FILE='tools/segmenter/ctbModel'
default_LEX='tools/segmenter/lex.GB.txt'
parser = OptionParser()
parser.add_option("-m", "--model", action="store", dest="model",
                  default=default_MODEL_FILE, help="set the Chinese segmentation model file (Zhang Le's maxent format)")
parser.add_option("-l", "--lexicon", action="store", dest="lexicon",
                  default=default_LEX, help="set the Chinese lexicon to assist the segmentation process")
parser.add_option("-n", "--number-threads", action="store", type="int", dest="number_threads",
                  default=1, help="set the number of threads")
#parser.add_option("-c", "--Chinese", action="store_true", dest="is_chinese", 
#                  default=False, help="whether the language is Chinese")

(options, args) = parser.parse_args()



####################### Class of Maxent ####################################################
exp = math.exp
DBM_MAX = 1.0E+304
verbose = 1
class ItemMap: #{{{
    """
    doc test string {{{
    >>> m = ItemMap()
    >>> len(m)
    0
    >>> m.add('a')
    0
    >>> m.add('a')
    0
    >>> m.add('b')
    1
    >>> len(m)
    2
    >>> m[0]
    'a'
    >>> m[1]
    'b'
    >>> m[2]
    >>>
    }}}
    """
    def __init__(self):
        self.index = [] # store item
        self.dict  = {} # map item to index

    def __str__(self):
        # TODO: avoid using += for speed
        s = '[ItemMap]\n'
        for i in xrange(len(self.index)):
            s += '%d: %s\n' % (i,self.index[i])
        return s

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if i < 0 or i >= len(self.index):
            return None
        else:
            return self.index[i]
    
    def id(self, item):
        try:
            return self.dict[item]
        except KeyError:
            return None

    def add(self, item):
        if self.dict.has_key(item):
            return self.dict[item]
        else:
            i = len(self.index)
            self.dict[item] = i
            self.index.append(item)
            return i
        #}}}

class Maxent:
    def __init__(self, model_file_name):
        # the row format of paramater array:
        # pred_id  (outcome_id1, param1),..., (outcome_idn, paramn)
        # each pred_id may have multiply outcomes and corresponding params
        self.params = None
        self.pred_map = self.outcome_map = None
        self.model_file_name = model_file_name
    def __str__(self):
        if self.params is None:
            return 'Empty Model (Python Version)'
        n = 0
        for i in xrange(len(self.params)):
            n += len(self.params[i])

        return"""Conditional Maximum Entropy Model (Python Version)
        Number of context predicates  : %d
        Number of outcome             : %d
        Number of paramaters(features): %d""" \
        % (len(self.pred_map), len(self.outcome_map),n)
    def start(self):
        self.load(self.model_file_name)
    def predict_features(self, str_features):
        return self.predict(str_features.split(" "))
    def check_modeltype(self, model):
        return 0
    def load(self, model, param = ''):
        """Load a ME model from model file previously saved by Maxent Trainer.

           param is optional if the parameter file is not default .param 
        """
        binary = self.check_modeltype(model)
        if binary:
            raise "binary format not supported yet"
            # load_model_bin(model);
        else:
            self.load_model_txt(model)
    def load_model_txt(self, model, encoding='gbk'):
        #print 'loading txt model from %s' % (model)
        self.pred_map = ItemMap()
        self.outcome_map = ItemMap()
        self.params = []
        f = open(model)
        # skip header comments
        line = f.readline()[:-1]
        if 'txt,maxent' not in line:
            raise """This is pure python version of maxent module, only txt maxent model can be accepted"""
        while line == '' or line[0] == '#':
            line = f.readline()[:-1]

        # read context predicates
        count_features = int(line)
        for i in range(count_features):
            line = f.readline()   
            line = line.decode(encoding, 'ignore')
            line = line[:-1]
            #print i, line.encode('utf8')
            self.pred_map.add(line)

        # read outcomes 
        line = f.readline()[:-1]
        count = int(line)
        for i in range(count):
            line = f.readline()
            line = line[:-1]
            self.outcome_map.add(line)
        # read parameters
        #count = len(self.pred_map)
        count = count_features
        assert count > 0
        fid = 0
        for i in range(count):
            line = f.readline()
            line = line.split()
            params = []
            for i in range(1, len(line)):
                oid = int(line[i])
                params.append((oid, fid))
                fid += 1
            self.params.append(params)
        # load theta
        theta = []
        n_theta = int(f.readline())
        theta = [0] * n_theta
        for i in range(n_theta):
            theta[i] = float(f.readline())
        # put theta into params
        # TODO: possible speed up?
        for param in self.params:
            for i in range(len(param)):
                param[i] = (param[i][0], theta[param[i][1]])
    def eval(self, context):
        """Evaluates given context and return a outcome distribution.

        i.e. Pr(outcome | context)
        context is a list of string names of the contextual predicates
        contextual predicates which are not seen during training are
        simply dropped off.
        eval() return a list of all possible outcomes with their probabilities:
        [(outcome1, prob1), (outcome2, prob2),...] 
        the return list is sorted on their probabilities in descendant order.
        """
        assert type(context) == types.ListType or type(context) == types.TupleType
        n_outcome = len(self.outcome_map)
        probs = [0.0] * n_outcome
        #outcome_sum = zeros(len(self.outcome_map), float)
        for c in context:
            pid = self.pred_map.id(c)
            if pid is not None:
                param = self.params[pid]
                for j in range(len(param)):
                    probs[param[j][0]] += param[j][1]
        sum = 0.0
        for i in range(n_outcome):
            try:
                probs[i] = exp(probs[i])
            except OverflowError:
                probs[i] = DBL_MAX
            sum += probs[i]
        for i in range(n_outcome):
            probs[i] /= sum
        #TODO: optimize here exp(outcome_sum,outcome_sum) does not work
        outcomes = []
        for i in range(n_outcome):
            outcomes.append((self.outcome_map[i], probs[i]))
        outcomes.sort(lambda x,y: -cmp(x[1], y[1]))
        return outcomes
    def predict(self, context):
        """Evaluates given context and return the most possible outcome y

           This function is a thin wrapper for  eval().
        """
        return self.eval(context)[0][0]
####################### end of Class Maxent ################################################
################################### time ###################################################
class Time_it:
    def start(self):
        self.s = time.time()
    def stop_restart(self):
        self.e = time.time()
        r = "%.3f seconds" % (self.e-self.s)
        self.s = time.time()
        return r
    def stop(self):
        self.e = time.time()
        return "%.3f seconds" % (self.e-self.s)
class CHINESE_WORD_SEGMENTER:
    segmenter = None
    timer = Time_it()
    def __init__(self, model_file_name='', lexFile=""):
        if len(model_file_name)==0:
            print >>sys.stderr, 'ERROR: the segmenter must need a model file name when it is constructed.'
            sys.exit(-1)
        print >>sys.stderr, 'SEGMENTER: start to load the model file', model_file_name, '......'
        self.segmenter = Maxent(model_file_name)
        self.segmenter.start()
        print >>sys.stderr, 'SEGMENTER: loading is completed!'
        if len(lexFile)>0:
            print >>sys.stderr, 'SEGMENTER: start to load the Chinese lexicon:', lexFile
            self.m_dict = True;
            # set up dict list if dict present
            self.m_lex=set()
            for line in open(lexFile):
                line=line.decode('gbk')
                line=line.strip()
                self.m_lex.add(line)
            print >>sys.stderr, 'SEGMENTER: loading is completed!'
        en_enum = \
        (':=&\/?@0123456789' + \
        'abcdefghijklmnopqrstuvwxyz' + \
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
        'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ' + \
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ').decode('utf8')
        en_seq = '[%s][\\s%s]*[%s]' % (en_enum, en_enum, en_enum)
        self.m_pattern_en_seq=re.compile(en_seq, re.UNICODE)
    def segment(self, input):
        # precleaning
        input = self.m_preclean.sub(' ', input)
        ######################################################
        # call segmenter to segment one Chinese sentence
        # generate features
        #self.timer.start()
        feature_list = self.FeatureGenerator(input)
        #for features in feature_list:
        #    print >>sys.stderr, ' '.join(features).encode('utf8')
        # call maxent of segmenter 
        predictor=[]
        for features in feature_list:
            r = self.segmenter.eval(features)
            #print r
            if len(r)!=4:
                print >>sys.stderr, "ERROR: the scores returned by the maxent are wrong"
                sys.exit(-1)
            ps=pe=pb=pm=-1.0
            tag2prob={}
            for l,p in r:
                tag2prob[l]=math.log(p)
            predictor.append(tag2prob)
        # decoding
        segmented_sentence=self.Segmenter(predictor, input)
        # correct English words
        corrected_segmented_sentence=self.correct_en_seg(input, segmented_sentence) 
        #print >>sys.stderr, "SEGMENTER: it costs", self.timer.stop()
        return corrected_segmented_sentence
    def remove_whitespace(self, input):
        return self.m_pattern_whitespace.sub('', input) 
    def correct_en_seg(self, right_lines, wrong_lines):
        r = ""    
        for correct_line, wrong_line in zip(right_lines.split("\n"), wrong_lines.split("\n")):
            correct_line_nospace=self.remove_whitespace(correct_line)
            wrong_line_nospace=self.remove_whitespace(wrong_line)
            #print >>sys.stderr, 'DEBUG001:', correct_line_nospace.encode('utf8')
            #print >>sys.stderr, 'DEBUG002:', wrong_line_nospace.encode('utf8')
            if not (correct_line_nospace == wrong_line_nospace):
                print >>sys.stderr, "Warning: correct_en_seg: correct_line is different from wrong_line after whitespaces are removed, so we directly use the wrong_line instead"
                print >>sys.stderr, "correct_line:", correct_line.encode('utf8')
                print >>sys.stderr, "wrong_line  :", wrong_line.encode('utf8')
                return wrong_line
            correct_segs = self.m_pattern_en_seq.findall(correct_line)
            wrong_segs = self.m_pattern_en_seq.findall(wrong_line)
            if not len(correct_segs) == len(wrong_segs):
                print >>sys.stderr, "Warning: correct_en_seg: the number of the English segments are different, so we directly use the wrong_line instead"
                print >>sys.stderr, "correct_line:", correct_line.encode('utf8')
                print >>sys.stderr, "wrong_line  :", wrong_line.encode('utf8')
                return wrong_line
            for correct_seg, wrong_seg in zip(correct_segs, wrong_segs):
                correct_seg_nospace=self.remove_whitespace(correct_seg)
                wrong_seg_nospace=self.remove_whitespace(wrong_seg)
                if not correct_seg_nospace == wrong_seg_nospace:
                    print >>sys.stderr, "Warning: correct_en_seg: correct_seg does not match wrong_seg"
                    print >>sys.stderr, "correct_seg:", correct_seg.encode('utf8')
                    print >>sys.stderr, "wrong_seg  :", wrong_seg.encode('utf8')
                    return wrong_line
                if correct_seg != wrong_seg:
                    wrong_line = wrong_line.replace(wrong_seg, correct_seg)
            r += wrong_line
        return r 
    def FeatureGenerator(self, input_sentence):
        output_feature_list=[]
        # produce features for training file
        input=input_sentence
        inputLen = len(input)+1
        tag = [None]*inputLen
        clChar = [None]*inputLen;
        words_list = input.split()
        wordNum = 0
        eventCount = 0
        for i in range(len(words_list)):
            word = words_list[i];
            # generate clChar[], and some tags
            j=0
            while j<len(word):
                left=True;
                if j == 0: # left Boundary
                    left = True;
                else:
                    left = False;
                currentChar=word[j];
                clChar[wordNum] = currentChar;
                
                if j == 0: # left Boundary
                    left = True;
                
                # convert numbers to '0'
                # full-width Chinese digits or ascii digits
                if  (currentChar>='０'.decode('utf8') and currentChar<='９'.decode('utf8')) or (currentChar>='0' and currentChar<='9'):
                    clChar[wordNum] = "0"; # all digits rep as 0
                j += 1
                
                # determinating the class tag
                if j == len(word):
                # right boundary
                    if left:
                        tag[wordNum] = "s";
                    else:
                        tag[wordNum] = "e";
                else:
                # not right boundary
                    if left:
                        tag[wordNum] = "b";
                    else:
                        tag[wordNum] = "m";
                if clChar[wordNum] in self.m_unicode2ascii:
                    clChar[wordNum]=self.m_unicode2ascii[clChar[wordNum]]
                wordNum += 1
        
        c0=None;
        c1=None;
        c2=None;
        c_1=None;
        c_2=None;
        numberSymbol=None;

        for k in range(0, wordNum): # write each word and encoding info out
            c0 = clChar[k];
            # find c_2 c_1 c0 c1 c2
            if k - 2 >= 0:
                c_2 = clChar[k - 2];
                c_1 = clChar[k - 1];
            elif k - 1 >=0:
                c_2 = "_";
                c_1 = clChar[k - 1];
            else:
                c_1 = "_";
                c_2 = "_"; # denote empty
            if k + 2 < wordNum:
                c1 = clChar[k + 1];
                c2 = clChar[k + 2];
            elif k + 1 < wordNum:
                c1 = clChar[k + 1];
                c2 = "_";
            else:
                c1 = "_";
                c2 = "_";
            # number and punctuation related features
            numberSymbol_list=[]
            for c in [c_2, c_1, c0, c1, c2]:
                if c in self.m_number:
                    numberSymbol = "1";
                elif c in self.m_date:
                    numberSymbol = "2";
                elif c in self.m_alphabet:
                    numberSymbol = "3";
                else:
                    numberSymbol = "4";
                numberSymbol_list.append(numberSymbol)
            numberSymbol=''.join(numberSymbol_list)
            # obtain feature context
            feature_list=[tag[k], "a="+c0, "b="+c_2, "c="+c_1, "d="+c1, "e="+c2, "f="+c_1+c0, "g="+c0+c1, "h="+c_2+c_1, "i="+c1+c2, "j="+c_1+c1]
            if c0 in self.m_punctuation:
                feature_list.append("o=1")
            feature_list.append("p="+numberSymbol)
            # obtain dictionary feats if dict provided
            # Arabic digits ignored
            if c0!="0" and self.m_dict:
                seqLen = 0;
                scan = k;
                backBest = 0
                frontBest = 0;
                minLen = 0;
                repeat = "";

                # back search
                while seqLen < 5 and scan > -1:
                    repeat = clChar[scan]+repeat
                    scan -= 1
                    seqLen += 1
                    if seqLen > minLen:
                        if repeat in self.m_lex:
                            backBest = seqLen;
                scan = k;
                seqLen = 0;
                repeat = "";
                while seqLen < 5 and scan < wordNum:
                    repeat=repeat+clChar[scan];
                    scan += 1
                    seqLen += 1
                    if seqLen > minLen:
                        if repeat in self.m_lex:
                            frontBest = seqLen;
            
                centerBest = 0;
                if (k != 0) and (k != wordNum - 1):
                    centre = clChar[k - 1] + clChar[k] + clChar[k + 1];
                    if centre in self.m_lex:
                        centerBest = 3;
                    if k > 1:
                        if (clChar[k - 2] + centre) in self.m_lex:
                            centerBest = 4;
                    if k < wordNum - 2:
                        if (centre + clChar[k + 2]) in self.m_lex:
                            centerBest = 4;
                # obtain relevant dict features
                if frontBest > 0 or backBest > 0 or centerBest > 0:
                    if frontBest == 1 and backBest == 1 and centerBest == 0:
                        feature_list.extend(["r="+c_1+"_s", "s="+c0+"_s", "t="+c1+"_s"])
                    elif frontBest > backBest and frontBest > centerBest:
                        feature_list.extend(["r="+c_1+"_b", "s="+c0+"_b", "t="+c1+"_b"])
                    elif backBest >= centerBest:
                        feature_list.extend(["r="+c_1+"_e", "s="+c0+"_e", "t="+c1+"_e"])
                    else:
                        feature_list.extend(["r="+c_1+"_m", "s="+c0+"_m", "t="+c1+"_m"])
            output_feature_list.append(feature_list);
            eventCount += 1
        return output_feature_list
    def Segmenter(self, predictor, input_sentence):
        if len(input_sentence.strip())!=0:
            input_sentence = self.remove_whitespace(input_sentence)
        else:
            return ""
        length=len(input_sentence);
        w=0;
        clChar = input_sentence
        words = [None]*(length + 1);
        best = [0.0]*(length + 1);
        bestLength = [0]*(length + 1);
        tag = [None]*length;
        best[0] = 0.0;
        bestLength[0] = 0;
        # apply segmentation testing algorithm
        for i in range(0, length+1):
            firstTime = True;
            for j in range(i-1, -1, -1):
                w = i - j;
                if w > 20: # word of length 20 is deemed not valid here
                    break;
                text = ""
                for m in range(j, i):
                    text = text + clChar[m];
                left = False;
                right = False;
                # obtain tag of this word
                for n in range(j, i):
                    if n == j:# left Boundary
                        left = True;
                    if n == i - 1: # last char, right boundary
                        right = True;
                    if right:
                        if left:
                            tag[n] = "s";
                        else:
                            tag[n] = "e";
                    else:
                    # not right
                        if left:
                            tag[n] = "b";
                        else:
                            tag[n] = "m";
                    left = False;
                    right = False;
                # evaluate log probability of this text string
                prob = best[i - w];
                for k in range(j, i): # write each word and encoding info out
                    # eval returns log prob
                    #print >>sys.stderr, 'DEBUG6:',k,tag[k]
                    stringProb = predictor[k][tag[k]];
                    prob = prob + stringProb;
                    if (not firstTime) and (prob < best[i]):
                        break
                if (prob >= best[i]) or firstTime: # firstTime=> no data in yet
                    best[i] = prob;
                    words[i] = text;
                    bestLength[i] = w;
                firstTime = False;
        # write out correct segmentation
        word_list=[]
        i = length;
        while i > 0:
            word_list.insert(0, words[i])
            i = i - bestLength[i];
        return ' '.join(word_list)

    # set up punctuation list
    m_punctuation=set("， 、 。 ． · ； ： ？ ！ ： … ¨ ， 、 ． · ； ： ？ ！ │ ─ │ ─ │ ＿ │ ─ （ ） （ ） ｛ ｝ ｛ ｝ 〔 〕 〔 〕 【 】 【 】 《 》 《 》 〈 〉 ∧ ∨ 「 」 「 」 『 』 『 』 （ ） ％ ｛ ｝ 〔 〕 ‘ ’ “ ” ＂ ＂ ｀ ＇ ＃ ＆ ~ , : ; ! % > < / \" ' [ ] { } \\ - . ) (".decode('utf8').split())
    m_number=set("１ ２ ３ ４ ５ ６ ７ ８ ９ ０ 1 2 3 4 5 6 7 8 9 0 ○ 〇 一 二 三 四 五 六 七 八 九 十 零 百 千 万 亿 两 廿 卅 ".decode('utf8').split())
    # setup m_unicode2ascii, while extracting features we always convert some unicode characters to ascii characters, e.g., '，' to ','
    m_unicode2ascii={
        "\xa1\xb4".decode('gbk'):"<",
        "\xa3\xbc".decode('gbk'):"<",
        "\xa1\xb5".decode('gbk'):">",
        "\xa3\xbe".decode('gbk'):">",
        "\xa3\xa1".decode('gbk'):"!",
        "\xa3\xa3".decode('gbk'):"#",
        "\xa3\xa5".decode('gbk'):"%",
        "\xa3\xa6".decode('gbk'):"&",
        "\xa3\xa8".decode('gbk'):"(",
        "\xa3\xa9".decode('gbk'):")",
        "\xa3\xaa".decode('gbk'):"*",
        "\xa3\xab".decode('gbk'):"+",
        "\xa3\xac".decode('gbk'):",",
        "～".decode('utf8'):",",
        "~".decode('utf8'):",",
        "\xa3\xad".decode('gbk'):"-",
        "━".decode('utf8'):"-",
        "\xa3\xae".decode('gbk'):".",
        "·".decode('utf8'):".",
        "・".decode('utf8'):".",
        "•".decode('utf8'):".",
        "\xa3\xaf".decode('gbk'):"/",
        "\xa3\xb0".decode('gbk'):"0",
        "\xa3\xb1".decode('gbk'):"1",
        "\xa3\xb2".decode('gbk'):"2",
        "\xa3\xb3".decode('gbk'):"3",
        "\xa3\xb4".decode('gbk'):"4",
        "\xa3\xb5".decode('gbk'):"5",
        "\xa3\xb6".decode('gbk'):"6",
        "\xa3\xb7".decode('gbk'):"7",
        "\xa3\xb8".decode('gbk'):"8",
        "\xa3\xb9".decode('gbk'):"9",
        "\xa3\xba".decode('gbk'):":",
        "\xa3\xbb".decode('gbk'):";",
        "\xa3\xbd".decode('gbk'):"=",
        "\xa3\xbf".decode('gbk'):"?",
        "\xa3\xc0".decode('gbk'):"@",
        "\xa3\xc1".decode('gbk'):"A",
        "\xa3\xc2".decode('gbk'):"B",
        "\xa3\xc3".decode('gbk'):"C",
        "\xa3\xc4".decode('gbk'):"D",
        "\xa3\xc5".decode('gbk'):"E",
        "\xa3\xc6".decode('gbk'):"F",
        "\xa3\xc7".decode('gbk'):"G",
        "\xa3\xc8".decode('gbk'):"H",
        "\xa3\xc9".decode('gbk'):"I",
        "\xa3\xca".decode('gbk'):"J",
        "\xa3\xcb".decode('gbk'):"K",
        "\xa3\xcc".decode('gbk'):"L",
        "\xa3\xcd".decode('gbk'):"M",
        "\xa3\xce".decode('gbk'):"N",
        "\xa3\xcf".decode('gbk'):"O",
        "\xa3\xd0".decode('gbk'):"P",
        "\xa3\xd1".decode('gbk'):"Q",
        "\xa3\xd2".decode('gbk'):"R",
        "\xa3\xd3".decode('gbk'):"S",
        "\xa3\xd4".decode('gbk'):"T",
        "\xa3\xd5".decode('gbk'):"U",
        "\xa3\xd6".decode('gbk'):"V",
        "\xa3\xd7".decode('gbk'):"W",
        "\xa3\xd8".decode('gbk'):"X",
        "\xa3\xd9".decode('gbk'):"Y",
        "\xa3\xda".decode('gbk'):"Z",
        "\xa3\xdb".decode('gbk'):"[",
        "\xa3\xdc".decode('gbk'):"\\",
        "\xa3\xdd".decode('gbk'):"]",
        "\xa3\xde".decode('gbk'):"^",
        "\xa3\xdf".decode('gbk'):"_",
        "\xa3\xe1".decode('gbk'):"a",
        "\xa3\xe2".decode('gbk'):"b",
        "\xa3\xe3".decode('gbk'):"c",
        "\xa3\xe4".decode('gbk'):"d",
        "\xa3\xe5".decode('gbk'):"e",
        "\xa3\xe6".decode('gbk'):"f",
        "\xa3\xe7".decode('gbk'):"g",
        "\xa3\xe8".decode('gbk'):"h",
        "\xa3\xe9".decode('gbk'):"i",
        "\xa3\xea".decode('gbk'):"j",
        "\xa3\xeb".decode('gbk'):"k",
        "\xa3\xec".decode('gbk'):"l",
        "\xa3\xed".decode('gbk'):"m",
        "\xa3\xee".decode('gbk'):"n",
        "\xa3\xef".decode('gbk'):"o",
        "\xa3\xf0".decode('gbk'):"p",
        "\xa3\xf1".decode('gbk'):"q",
        "\xa3\xf2".decode('gbk'):"r",
        "\xa3\xf3".decode('gbk'):"s",
        "\xa3\xf4".decode('gbk'):"t",
        "\xa3\xf5".decode('gbk'):"u",
        "\xa3\xf6".decode('gbk'):"v",
        "\xa3\xf7".decode('gbk'):"w",
        "\xa3\xf8".decode('gbk'):"x",
        "\xa3\xf9".decode('gbk'):"y",
        "\xa3\xfa".decode('gbk'):"z",
        "\xa3\xfb".decode('gbk'):"{",
        "\xa3\xfc".decode('gbk'):"|",
        "\xa3\xfd".decode('gbk'):"}",
    }
    # set up date type word list
    m_date=set("年 月 日".decode('utf8').split())
    # set up alphabet list
    m_alphabet=set("Ａ Ｂ Ｃ Ｄ Ｅ Ｆ Ｇ Ｈ Ｉ Ｊ Ｋ Ｌ Ｍ Ｎ Ｏ Ｐ Ｑ Ｒ Ｓ Ｔ Ｕ Ｖ Ｗ Ｘ Ｙ Ｚ ａ ｂ ｃ ｄ ｅ ｆ ｇ ｈ ｉ ｊ ｋ ｌ ｍ ｎ ｏ ｐ ｑ ｒ ｓ ｔ ｕ ｖ ｗ ｘ ｙ ｚ a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".decode('utf8').split())
    m_lex=None
    m_dict=False
    m_pattern_en_seq=None
    m_pattern_whitespace=re.compile(r'\s', re.UNICODE)
    m_preclean=re.compile(r'\s+', re.UNICODE)  
      
# main
if __name__ == '__main__':
    MODEL_FILE=options.model
    LEX=options.lexicon
    segmenter=CHINESE_WORD_SEGMENTER(MODEL_FILE, LEX)
    #print (segmenter.segment('什么时候提车啊，不是说星期四吗，怎么又改时间了啊?'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('中华人民共和国'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('之前就是订的2号，然后被虎航卑劣地推到明天了嘛'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('名:粱卫诠， 学manual, 星期六早上'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('Lenovo 即将推出高端 ThinkPad T61p'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('这个老板堪称女中豪杰，对我们手下的非常好，我们也心甘情愿为她工作，甚至是晚上和周末加班，这在美国人里还不多见，她也算是我工作过的最好的老板。'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('八点我就起来了类　然后听到喜气洋洋的消息,我家小苏同学悲痛欲绝耶  然后怀着幸灾乐祸的心情在看明朝破事　哇~~笑死'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('不在啊。outside now'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('··· ··· 不知道过了多久，浑浑噩噩的我终于欣喜的发现：地声正一点点的减弱，大地的抖动正一点点的变缓，而头上的烟花也不在闪烁。'.decode('utf8'))).encode('utf8')
    #print (segmenter.segment('国际汽联官方网在听证会后发表声明说：“沃达丰•迈凯轮•梅塞德斯车队拥有法拉利的'.decode('utf8'))).encode('utf8')
    number_threads=options.number_threads
    timer = Time_it()
    ## wangpd: in order to use the multiprocessing.Pool()'s map(function_name, arg_list) function,
    ##         where function_name can not be a member function of a class
    def pickle_function(one):
        return segmenter.segment(one)
    if number_threads==1:
        timer.start()
        # start to wait for standard input
        print >>sys.stderr, 'SEGMENTER: ready for input from standard input'
        print >>sys.stderr, 'SEGMENTER: using 1 process'
        infile=sys.stdin
        line=infile.readline()
        while len(line)!=0:
            rline=line.decode('utf8')
            rline=rline.strip()
            if rline=="><EXIT><":
                print >>sys.stderr, 'SEGMENTER: exit.'
                sys.exit(0)
            rline=segmenter.segment(rline)
            print >>sys.stdout, rline.encode('utf8')
            sys.stdout.flush()
            line=infile.readline()
        print >>sys.stderr, "SEGMENTER: it costs", timer.stop()
    else:
        timer.start()
        print >>sys.stderr, 'SEGMENTER: ready for input from standard input'
        print >>sys.stderr, 'SEGMENTER: using', number_threads, 'processes'
        input_buffer_list=[]
        infile=sys.stdin
        line=infile.readline()
        while len(line)!=0:
            rline=line.decode('utf8')
            rline=rline.strip()
            input_buffer_list.append(rline)
            line=infile.readline()
        print >>sys.stderr, 'SEGMENTER: get', len(input_buffer_list), 'sentences for segmenting'
        print >>sys.stderr, 'SEGMENTER: start to segment.....'
        process_pool=Pool(number_threads)
        output_list=process_pool.map(pickle_function, input_buffer_list)
        for line in output_list:
            print >>sys.stdout, line.encode('utf8')
        sys.stdout.flush()        
        print >>sys.stderr, "SEGMENTER: it costs", timer.stop()



