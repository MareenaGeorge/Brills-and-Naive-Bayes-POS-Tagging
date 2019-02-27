

from collections import Counter
import sys

class BrillsPosTagging:
    currentTags = []
    correctTags = []
    Tags = set()
    countWordTag = {}
    PrevCurrTagCount = {}
    CountTags = {}
    def errorCalculate(self,sentTag1, sentTag2):
        errortags = 0
        i=0
        while i<(len(sentTag1) - 1):
            if sentTag1[i][1] != sentTag2[i][1]:
                errortags += 1
        return (float(errortags) / float(len(sentTag1)))

    #Function to create POS tagged model based on
    # 1 Brill's transformation based POS Tagging
    # 2 Naïve Bayesian Classification (Bigram) based POS Tagging
    def createPosTagModel(self,corpusfile,sent):
        # Reading the txt file
        cfile = open(corpusfile, 'r')
        data = cfile.read()
        word_tags = data.split()
        prev_Tag = None
        wrongTags = 0
        for word_tag in word_tags:
            word = word_tag.split("_")[0]
            tag = word_tag.split("_")[1]
            if len(self.Tags) < 50:
                self.Tags.add(tag)
            if word in self.countWordTag:
                self.countWordTag[word][tag] = self.countWordTag[word].get(tag, 0) + 1
            else:
                self.countWordTag[word] = {}
                self.countWordTag[word][tag] = 1
            if prev_Tag != None:
                self.PrevCurrTagCount[(prev_Tag, tag)] = self.PrevCurrTagCount.get((prev_Tag, tag), 0) + 1
            prev_Tag = tag
            self.CountTags[tag] = self.CountTags.get(tag, 0) + 1

        # Brill's transformation based POS Tagging
        # Step 1:
        #  Initialize the tags to the most probable one.
        #
        TotalTags = len(word_tags)
        for word_tag in word_tags:
            #get words and tags
            word = word_tag.split('_')[0]
            tag = word_tag.split('_')[1]
            maxWordTag = max(self.countWordTag[word].values())
            if maxWordTag == 0:
                currentTag = "NN"
            else:
                for key in self.countWordTag[word]:
                    if self.countWordTag[word][key] == maxWordTag:
                        currentTag = key
            self.currentTags.append((word, currentTag))
            self.correctTags.append((word, tag))
            if tag != currentTag:
                wrongTags += 1
        error = float(wrongTags) * 100 / float(TotalTags)

        print("After assigning the most probable tags, Initial Error is : %.2f" % error, " %")

        # transformationRules -> ((FROM_TAG, TO_TAG, PREVIOUS_WORD_TAG), SCORE)
        transformationRules = self.getBestInstance().most_common()

        transformationRulesFile = open('transformationRules1.txt', 'w')
        transformationRulesFile.write(("FROM_TAG, TO_TAG, PREVIOUS_WORD_TAG, SCORE \n"))
        for item in transformationRules:
            transformationRulesFile.write((item[0][0] + "," + item[0][1] + "," + item[0][2] + "," + str(item[1]) + "\n"))
        transformationRulesFile.close()

        # Applying Brills and Naive Bayes Tagging to the input sentence.
        sentenceWords = sent.split()
        sentWords = sent.split()
        sentenceWordTags = []
        Original_sentenceWordTags = []
        for word_tag in sentenceWords:
            word = word_tag.split('_')[0]
            tag = word_tag.split('_')[1]
            Original_sentenceWordTags.append((word, tag))
            maxWordTag = max(self.countWordTag[word].values())
            if maxWordTag == 0:
                currentTag = "NN"
            else:
                for key in self.countWordTag[word]:
                    if self.countWordTag[word][key] == maxWordTag:
                        currentTag = key
            sentenceWordTags.append([word, currentTag])

        # Train on rules 5 times or until error is less than 0.1
        i = 0
        #while (i < 5 and self.errorCalculate(sentenceWordTags, Original_sentenceWordTags) > 0.1):
        while(i<5):
            for i in range(1, len(sentenceWordTags)):
                for rule in transformationRules:
                    if rule[0][2] == sentenceWordTags[i - 1][1]:
                        if rule[0][0] == sentenceWordTags[i][1]:
                            sentenceWordTags[i][1] = rule[0][1]
            i += 1

        # Naive Bayes Based POS tagging.
        # Getting Tag transition probabilities counts and Word likelihood probabilities counts.
        NB_Sentence_WT = []
        NB_SentenceT = []
        for word_tag in sentWords:
            word = word_tag.split('_')[0]
            if word in self.countWordTag:
                NB_SentenceT.append(list(self.countWordTag[word].keys()))
            else:
                NB_SentenceT.append(list('NN'))
            NB_Sentence_WT.append([word, None])
        combinations = [[]]
        #print(NB_SentenceT)
        combin = [[]]
        for x in NB_SentenceT:
            combinations = [i + [y] for y in x for i in combinations]
            #print(combinations)

        # Calculating probabilities for all the different possible combinations of POS tags.
        comb_prob = []
        for i, j in enumerate(combinations):
            num = self.countWordTag.get(sentenceWords[0].split('_')[0], 0).get(j[0], 0)
            den = self.CountTags.get(j[0], 0)
            if num == 0 or den == 0:
                comb_prob.append(0)
            else:
                comb_prob.append(float(num) / float(den))
            #print("( " + "\"" + str(eachbigram) + "\"" + " , " + str(bigram_count) + " , " + str(gt_count) + " , " + str(bigram_prob) + " )")

        for i, j in enumerate(combinations):
            totalprob = 1
            t=1
            while(t<len(j)-1):
                num1 = self.countWordTag.get(sentenceWords[t].split('_')[0], 0).get(j[t], 0)
                den1 = self.CountTags.get(j[t], 0)
                num2 = self.PrevCurrTagCount.get((j[t - 1], j[t]), 0)
                den2 = self.CountTags.get(j[t], 0)
                #print(str(num1) + "," + str(den1) + "," + str(num2) + "," + str(den2))
                if num1 == 0 or num2 == 0 or den1 == 0 or den2 == 0:
                    totalprob = 0
                    break
                else:
                    totalprob = totalprob * (float(num1) / float(den1)) * (float(num2) / float(den2))
                t+=1
            comb_prob[i] = comb_prob[i] * totalprob
            #print("( " + str(combinations[i])+ "," + str(num1)+ "," + str(den1)+ "," + str(num2)+ "," + str(den2)+ "," + str(comb_prob[i]) + "," + " , " + str(totalprob))
        
        # Assigning the highest probable tags to the given sentence.
        m = 0
        max_index = 0
        i=0
        while(i< len(combinations)-1):
            if comb_prob[i] > m:
                m = comb_prob[i]
                max_index = i
            i+=1

        for i, j in enumerate(combinations[max_index]):
            NB_Sentence_WT[i][1] = j

        print("Input incomplete POS Tagged Sentence ")
        print()
        print(sent, end="\n\n")
        print(" Brills POS Tagged Sentence \n")
        for i in sentenceWordTags:
            print(i[0] + "_" + i[1], end=" ")
        print(end="\n\n")
        print("Naive Bayes POS Tagged Sentence \n")
        for i in NB_Sentence_WT:
            print(i[0] + "_" + i[1], end=" ")
        print(end="\n\n")

    def getBestInstance(self):
        transformationRules = Counter()
        FromTo_PrevWordsTags = {}
        count = 0
        for fromTag in self.Tags:
            for toTag in self.Tags:
                if fromTag == toTag:
                    continue
                else:
                    FromTo_PrevWordsTags[(fromTag, toTag)] = {T: 0 for T in self.Tags}
                    pos=1
                    while(pos< len(self.currentTags)-1):
                        if self.correctTags[pos][1] == toTag and self.currentTags[pos][1] == fromTag:
                            FromTo_PrevWordsTags[(fromTag, toTag)][self.currentTags[pos - 1][1]] += 1
                        elif self.correctTags[pos][1] == fromTag and self.currentTags[pos][1] == fromTag:
                            FromTo_PrevWordsTags[(fromTag, toTag)][self.currentTags[pos - 1][1]] -= 1
                        pos+=1
                    for prevTag in FromTo_PrevWordsTags[(fromTag, toTag)]:
                        if FromTo_PrevWordsTags[(fromTag, toTag)][prevTag] > 0:
                            count += 1
                            transformationRules[(fromTag, toTag, prevTag)] = FromTo_PrevWordsTags[(fromTag, toTag)][prevTag]
        return transformationRules

if len(sys.argv)<2:
    print("Corpus file should be added as command line..")
    sys.exit()
else:
    sentence = input("Enter sentence\n")
    b = BrillsPosTagging()
    b.createPosTagModel(sys.argv[1],sentence)
