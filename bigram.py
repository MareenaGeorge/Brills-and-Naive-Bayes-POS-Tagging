import sys
import re
from collections import defaultdict


class bigramModel:
    
    unigramcount = {}
    bigramcount = {}
    bigramprobability = {}
    bigramaddoneprobability ={}
    ntokens = 0
    vocabulary = 0
    bigram_goodturing = {}
    cstar ={}
    
    #Function to create bigram model from the txt file
    def createBigramModelFromFile(self,corpusfile):

        #Reading the txt file
        with open(corpusfile,"r") as cfile:
            corpus = cfile.read()
            #By matching any character that is not whitespace
            tokens = re.findall('(\S+)',corpus)
            #print(token)
            self.ntokens = len(tokens)       #N

            #print("Words:" , self.words)
        cfile.close()
        
        #Unigram Probability
        #Probability = C / N
        #C – count (number) of words in a type
        #N – no. of word tokens
        for eachtoken in set(tokens):
            self.unigramcount[eachtoken] = tokens.count(eachtoken)
        self.vocabulary = len(self.unigramcount)    #V

        #print(self.unigramcount)

        #Bigram List
        bigramList = []
        i = 0
        while i < len(tokens) - 1:
            bigram = tokens[i] +" "+ tokens[i+1]
            bigramList.append(bigram)
            i += 1
        #print(bigramList)

        #To compute Bigram counts and Probabilities with no smoothing
        with open("BigramNoSmoothing.txt","w") as nosmoothfile:
            nosmoothfile.write("(W1 W2, C(W1, W2), P(W1, W2) )" +'\n')
            for eachbigram in set(bigramList):
                self.bigramcount[eachbigram] = bigramList.count(eachbigram)
                self.bigramprobability[eachbigram] = bigramList.count(eachbigram)/self.unigramcount[re.findall('(\S+)', eachbigram)[0]]
                nosmoothfile.write("( " + "\"" + str(eachbigram)+ "\""  + " , " + str(self.bigramcount[eachbigram])+ " , " + str(self.bigramprobability[eachbigram]) +" )" +'\n')
        nosmoothfile.close()

        # To compute Bigram counts and Probabilities with Add One smoothing
        with open("BigramAddOneSmoothing.txt", "w") as addonesmoothfile:
            # normalizer = N/(N+V)
            addonenormalizer = self.ntokens / (self.ntokens + self.vocabulary)
            addonesmoothfile.write("(W1 W2, C(W1, W2), C* = (C+1)N/(N+V) , P(W1, W2) )" + '\n')
            for eachbigram in set(bigramList):
                # C* = (C+1)N/(N+V)
                self.cstar[eachbigram] = (self.bigramcount[eachbigram] + 1) * addonenormalizer
                # prob = (C +1)/(N + V)
                #self.bigramaddoneprobability[eachbigram] = (bigramList.count(eachbigram) + 1) / (self.unigramcount[re.findall('(\S+)', eachbigram)[0]]+ self.vocabulary)
                addonesmoothfile.write(
                    "( " + "\"" + str(eachbigram) + "\"" + " , " + str(self.bigramcount[eachbigram]) + " , " + str(self.cstar[eachbigram]) + " , " + str((bigramList.count(eachbigram) + 1) / (self.unigramcount[re.findall('(\S+)', eachbigram)[0]]+ self.vocabulary)) + " )" + '\n')
            #print(addonenormalizer)
        addonesmoothfile.close()

        #To compute  Bigram counts and Probabilities with Good-Turing Discounting based Smoothing
        #grouping a sequence of key-value pairs into a dictionary of lists
        self.bigram_goodturing = defaultdict(list)
        for key in self.bigramcount.keys():
            if self.bigramcount[key] in self.bigram_goodturing.keys():
                self.bigram_goodturing[self.bigramcount[key]].append(key)
            else:
                self.bigram_goodturing[self.bigramcount[key]] = [key]
        #print(self.bigram_goodturing.values())
        with open("BigramGoodTuringSmoothing.txt","w") as goodturingfile:
            goodturingfile.write("( C(W1, W2), Nc , C* = (c+1)*N[c+1]/N[c] , P(W1, W2) )" + '\n')
            for key in sorted(self.bigram_goodturing.keys()):
                goodturingcount =  (key+1)*(len(self.bigram_goodturing[(key+1)])/len(self.bigram_goodturing[key]))
                goodturingfile.write(
                    "( " + "\"" + str(key) + " , " + str(len(self.bigram_goodturing[key])) + " , " + str(goodturingcount) + " , " + str(goodturingcount/self.ntokens) + " )" + '\n')

        goodturingfile.close()

    def computeBigramForSentence(self,bigramList):
        sentenceProb = 1
        print("\nCASE 1: Bigram with no smoothing")
        print("\n( W1 W2, C(W1, W2), P(W1, W2) )")
        for eachbigram in bigramList:
            bigram_count, bigram_prob = 0, 0
            if eachbigram in self.bigramcount.keys():
                bigram_count = self.bigramcount[eachbigram]
            if eachbigram in self.bigramprobability.keys():
                bigram_prob = self.bigramprobability[eachbigram]
            sentenceProb = sentenceProb * bigram_prob
            print("( " + "\"" + str(eachbigram) + "\"" + " , " + str(bigram_count) + " , " + str(bigram_prob) + " )" )
        print("Total bigram probability of the sentence with no smoothingis " + str(sentenceProb) + "\n")


        addonenormalizer = self.ntokens / (self.ntokens + self.vocabulary)
        #print(self.ntokens)
        #print(self.vocabulary)
        #print(addonenormalizer)
        sentenceProb = 1
        sentenceProbignoreUnigram = 1
        print("\nCASE 2: Bigram with Add-one smoothing")
        print("\n( W1 W2, C(W1, W2), C* = (C+1)N/(N+V) , P(W1, W2) )")
        for eachbigram in bigramList:
            bigram_count, bigram_prob ,bigram_probignoreUnigram = 1, 0 ,0
            if eachbigram in self.bigramcount.keys():
                bigram_count += self.bigramcount[eachbigram]
            unigram = re.findall('(\S+)', eachbigram)[0]
            if unigram in self.unigramcount.keys():
                unigram_count = self.unigramcount[unigram]
            bigram_prob = bigram_count / (unigram_count + self.vocabulary)
            bigram_probignoreUnigram = bigram_count / ( self.vocabulary)
            sentenceProb = sentenceProb * bigram_prob
            sentenceProbignoreUnigram = sentenceProbignoreUnigram * bigram_probignoreUnigram
            #print("( "  + str(bigram_count)  + " , " + " , " + str(unigram_count)+ "," + str(self.vocabulary) + "," + str(unigram_count+self.vocabulary) + " , "+ str(bigram_prob) + " , " + str(addonenormalizer) + " , " + str(sentenceProb) + " )")
            print("( " + "\"" + str(eachbigram) + "\"" + " , " + str(bigram_count-1) + " , "+ str(bigram_count*addonenormalizer) + " , " + str(bigram_prob) + " )")
        print("Total bigram probability of the sentence with Add-One smoothing is " + str(sentenceProb) + "\n")
        print("Total bigram probability of the sentence with Add-One smoothing ignoring Unigram is " + str(sentenceProbignoreUnigram) + "\n")
        sentenceProb = 1
        print("\nCASE 3: Bigram with  Good-Turing Discounting based Smoothing")
        print("\n(W1 W2, C(W1, W2), C* = (c+1)*N[c+1]/N[c] , P(W1, W2) )")
        for eachbigram in bigramList:
            bigram_count = 0
            if eachbigram in self.bigramcount.keys():
                bigram_count += self.bigramcount[eachbigram]
            if bigram_count == 0:
                gt_count = len(self.bigram_goodturing[1])
            else:
                gt_count = (bigram_count + 1) * (
                            len(self.bigram_goodturing[(bigram_count + 1)]) / len(self.bigram_goodturing[bigram_count]))
            sentenceProb = sentenceProb * (gt_count / self.ntokens)
            bigram_prob = gt_count/self.ntokens
            print("( " + "\"" + str(eachbigram) + "\"" + " , " + str(bigram_count) + " , " + str(gt_count)+ " , " + str(bigram_prob) + " )")
        print("Total bigram probability of the sentence with Good-Turing Discounting based Smoothing is " + str(sentenceProb) + "\n")

# if Corpus file not specified as command line argument
if len(sys.argv)<2:
    print("Corpus file should be added as command line..")
    sys.exit()
else:
    print("Calculating bigram probabilities..")
    b = bigramModel()
    b.createBigramModelFromFile(sys.argv[1])
    while True:
        print("Enter the sentence or type 'q' to exit")
        sentence = input()
        if sentence == "q":
            break
        tokens = re.findall('(\S+)', sentence)
        #print(tokens)
        #bigramList = b.createBigramList(tokens)
        bigramList = []
        i = 0
        while i < len(tokens) - 1:
            bigram = tokens[i] + " " + tokens[i + 1]
            bigramList.append(bigram)
            i += 1
        b.computeBigramForSentence(bigramList)
    print("Exiting the program...")
