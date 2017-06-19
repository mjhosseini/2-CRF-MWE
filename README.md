This is the implementation of the double chained CRF used for predicting Multiword Expressions (MWE) and supersenses. 

[UW-CSE at SemEval-2016 Task 10: Detecting multiword expressions and supersenses using double-chained conditional random fields.](https://www.aclweb.org/anthology/S/S16/S16-1143.pdf) Mohammad Javad Hosseini, Noah A. Smith, and Su-In Lee. In Proceedings of the NAACL Workshop on Semantic Evaluations (SemEval 2016), San Diego, CA, June 2016.

We participated at the [SemEval 2016 Task 10: Detecting Minimal Semantic Units and their Meanings (DiMSUM)](http://dimsum16.github.io/). Our submitted models ranked first overall in the competition.

We have implemented a Conditional Random Field and a Double-Chained Conditional Random Field model for joint learning of multiword expressions and supersenses.

The feature extraction is based on [AMALGrAM 2.0](https://github.com/nschneid/pysupersensetagger/blob/master/README.md) (A Machine Analyzer of Lexical Groupings And Meanings) and the dependencies are the same as AMALGrAM 2.0.

### Software

  - Python 2.7
  - Cython (tested on 0.21.1)
  - NLTK 3.0.2+ with the WordNet resource installed

### Running:

After downloading the code, given the above softwares are installed, you can run the code from the scripts folder to replicate the paper's results and/or to test on new data.

### Tagging Scheme
Please refer to dimsum-data-1.5/TAGSET.md for details.

#### Multiword Expressions:

The  annotation  for  MWEs  extends  the  conventional   BIO   scheme to  include  gappy  MWEs  with  one  level  of nesting. Segmentations  are  represented  using  six tags; the lower-case variants indicate that an expression is within another MWE’s gap.

-- O and o: single word expression
-- B and b: the first word of a MWE
-- I and i: a word continuing a MWE

#### Supersenses:

Each  noun  or  verb  expression  is  also  annotated with  a  supersense;   there  are  26  supersenses  for nouns  and  15  for  verbs.   Only  the  first  word  of  a MWE receives a supersense tag.

The input must be sentence and word tokenized and part-of-speech tagged (with the Penn Treebank POS tagset). To obtain automatic POS tags for tokenized text, we recommend the TurboTagger module within [TurboParser](http://www.cs.cmu.edu/~ark/TurboParser/) or the [TweetNLP Tagger](http://www.cs.cmu.edu/~ark/TweetNLP/).

### Data:

The datasets are in the folder dimsum-data-1.5. There is a readme file in the folder explaining the format. Our original submission is in the folder submitted_results.
