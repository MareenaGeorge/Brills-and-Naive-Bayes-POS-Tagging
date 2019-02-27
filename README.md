#CS630 NLP Homework 2
Structure
	|_1.txt - contains Q1
	|_bigram.py -contains Q2 source code
	|_BrillsAndNBPosTagging.py - contains Q3 source code
        |_Hw2_HandComputedReport.pdf -contains hand computed calculation of Q2 and Q3
        |_Hw2_Output_Report - contains the output of two source code in Q2 and Q3
        |_README.md
Language used: Python 3

Q2: Bigram Probabilities
bigram.py has Q2
 Code to execute the program
 
 		python bigram.py ./HW2_S18_NLP6320-NLPCorpusTreebank2Parts-CorpusA-Unix.txt
         Input Sentence: The president wants to control the board 's control	
 It has three (3) scenarios:
	i. No Smoothing
	ii. Add-one Smoothing
	iii. Good-Turing Discounting based Smoothing
It builds a bigram model based on three scenarios and also computes the total probability for the input sentence.



It creates 3 txt files on Bigram model Generation
	i. BigramNoSmoothing.txt
	ii. BigramAddOneSmoothing.txt
	iii. BigramGoodTuringSmoothing.txt

Q3: Transformation Based POS Tagging
BrillsAndNBPosTagging.py has Q3
 Code to execute the program
 
        python BrillsAndNBPosTagging.py ./HW2_S18_NLP6320_POSTaggedTrainingSet-Unix.txt
        Input Sentence: The_DT president_NN wants_VBZ to_TO control_??? the_DT board_NN 's_POS control_???
It has two parts
    i. Transformation-based POS Tagging
    ii. Naïve Bayesian Classification (Bigram) based POS Tagging
