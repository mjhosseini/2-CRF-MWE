Comprehensive Multiword Expressions (CMWE) Corpus
=================================================

Nathan Schneider, Spencer Onuffer, Nora Kazour, Emily Danchik, Michael T. Mordowanec, 
Henrietta Conrad, and Noah A. Smith 

 - version 1.0 (2014-03-26): 55k words, English web reviews

The annotations can be downloaded at 

  http://www.ark.cs.cmu.edu/LexSem/

This dataset is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) license (see LICENSE).


Synopsis
--------

This dataset provides human annotations of multiword expressions (MWEs) 
for sentences in social web reviews from the English Web Treebank corpus.
55,579 words (3,812 sentences, 723 documents) were annotated.
MWEs are formed by grouping together words into strong (highly idiosyncratic) 
or weak (loosely collocational) expressions according to our
[English annotation guidelines](https://github.com/nschneid/nanni/wiki/MWE-Annotation-Guidelines).
For example,

    I will sum_ it _up~with , it was worth_every_penny !

is annotated as containing 2 strong MWEs (sum_up, worth_every_penny) 
and 1 weak MWE (sum_up~with).

These are comprehensive annotations, i.e., for each sentence, 
the annotator marked *all* expressions deemed MWEs.
Every annotation was reviewed by at least two annotators.
See (Schneider et al., LREC 2014) for details.


Release contents
----------------

This download consists of:

 - README.md

 - LICENSE

 - obfuscated.mwe: The MWE annotations, with sentence context obscured for licensing reasons.

 - deobfuscate.py: Script that refers to a local installation of the 
   [English Web Treebank](http://catalog.ldc.upenn.edu/LDC2012T13) to fill in the rest of every sentence.
   
   To restore surrounding context, run

        python deobfuscate.py obfuscated.mwe /path/to/eng_web_tbk/data > corpus.mwe


File format
-----------

The deobfuscated file corpus.mwe will contain lines such as:

```
ewtb.r.388799.6 I will sum_ it _up~with , it was worth_every_penny !    {"_": [[3, 5], [10, 11, 12]], "words": [["I", "PRP"], ["will", "MD"], ["sum", "VB"], ["it", "PRP"], ["up", "RP"], ["with", "IN"], [",", ","], ["it", "PRP"], ["was", "VBD"], ["worth", "JJ"], ["every", "DT"], ["penny", "NN"], ["!", "."]], "~": [[3, 5, 6]]}
```

Each line contains 3 tab-separated fields:

  1. Sentence ID in the format `ewtb.r.<fileid>.<sentoffset>`

  2. Inline annotation as entered by a human annotator. The `_` character is used to join tokens 
     to form strong MWEs, and the `~` joiner forms weak MWEs. MWEs may contain gaps, in which case 
     the joiner character will be adjacent to a space (as in `sum_ it _up` above). 
     In rare cases a dollar sign suffix like `$1` will be used to index multiple parts of a weak MWE.

  3. JSON object containing the POS-tagged sentence (`"words"`) and 
     multiword groups by 1-based token offsets (`"_"` for strong MWEs, `"~"` for weak MWEs).
     This represents the same information as the inline annotation, but in a more directly 
     computer-readable format.

The ASCIIfied tokenization and gold POS tagging from the Web Treebank are used.


Further information
-------------------

This corpus is described in the paper

  -  Nathan Schneider, Spencer Onuffer, Nora Kazour, Emily Danchik, Michael T. Mordowanec, 
    Henrietta Conrad, and Noah A. Smith (2014). 
    Comprehensive annotation of multiword expressions in a social web corpus. 
    _Proceedings of the 9th Linguistic Resources and Evaluation Conference._

The corpus was used to train a system that identifies multiword expressions in context; 
this is described in

 -  Nathan Schneider, Emily Danchik, Chris Dyer, and Noah A. Smith (2014). 
    Discriminative lexical semantic segmentation with gaps: running the MWE gamut. 
    _Transactions of the Association for Computational Linguistics._

Please cite the former if you use this dataset directly, and the latter if you use 
the identification system (code to be released separately).

Contact [Nathan Schneider](http://nathan.cl) with questions.
