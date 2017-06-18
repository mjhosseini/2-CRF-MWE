This is the implementation of the double chained CRF used for predicting MWE and supersenses. 

UW-CSE at SemEval-2016 Task 10: Detecting multiword expressions and supersenses using double-chained conditional random fields. Mohammad Javad Hosseini, Noah A. Smith, and Su-In Lee. In Proceedings of the NAACL Workshop on Semantic Evaluations (SemEval 2016), San Diego, CA, June 2016.

We have implemented a Conditional Random Field and a Double-Chained Conditional Random Field model for joint learning of multiword expressions and supersenses.

The feature extraction is based on AMALGrAM 2.0 (A Machine Analyzer of Lexical Groupings And Meanings)

Dependencies
------------

### Platform

This software has been tested on recent Unix and Mac OS X platforms.
It has *not* been tested on Windows.

### Software

  - Python 2.7
  - Cython (tested on 0.19.1)
  - NLTK 3.0.2+ with the WordNet resource installed

The input must be sentence and word tokenized and part-of-speech tagged (with the Penn Treebank POS tagset). To obtain automatic POS tags for tokenized text, we recommend the TurboTagger module within [TurboParser](http://www.cs.cmu.edu/~ark/TurboParser/) or the [TweetNLP Tagger](http://www.cs.cmu.edu/~ark/TweetNLP/).

### Data

#### Lexicons and Word Clusters

Features in AMALGrAM's tagging model make use of several MWE lists extracted from existing English lexicons, as well as word clusters from a corpus of Yelp reviews. These are available as a separate download at http://www.cs.cmu.edu/~ark/LexSem/.

#### Corpus

The sentences in the annotated dataset that was used to train and evaluate AMALGrAM come from the English Web Treebank (EWTB), which is distributed by LDC. With permission from LDC and Google, the STREUSLE download at http://www.cs.cmu.edu/~ark/LexSem/ includes the source sentences and gold POS tags, but not the parse trees, from EWTB. The parse trees were not used in the lexical semantic annotation or in training the AMALGrAM tagger.

