#!/usr/bin/env python2.7
'''
Usage: python deobfuscate.py obfuscated.mwe /path/to/eng_web_tbk/dataAnalyzer > corpus.mwe

Restores singleton words and their POS tags by accessing a local installation 
of the English Web Treebank as released by LDC (http://catalog.ldc.upenn.edu/LDC2012T13).

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-03-24
'''
from __future__ import print_function, division
import os, sys, re, json, fileinput, codecs
from collections import defaultdict, Counter

inFP, ewtbDP = sys.argv[1:]

indata = []
docIDs = set()

for ln in fileinput.input(inFP):
    sentID, anno_, dataJS = ln.strip().split('\t')
    docID = sentID[:sentID.rindex('.')]
    docIDs.add(docID)
    sentOffset = int(sentID[sentID.rindex('.')+1:])
    indata.append((sentID, docID, sentOffset, anno_, json.loads(dataJS)))




reviewsDP = os.path.join(ewtbDP, 'reviews')


tagged_words = {}
for docID in docIDs:
    shortDocID = docID[docID.rindex('.')+1:]

    # load the doc's POS tags from XML
    doc_poses = []
    with codecs.open(os.path.join(reviewsDP, 'xml', shortDocID+'.xml'), 'r', 'utf-8') as xmlF:
        for ln in xmlF:
            m = re.match('^<Feature name="tag">(.+)</Feature>$', ln.strip())
            if m:
                doc_poses.append(m.group(1))

    doc_tagged_words = tagged_words[docID] = []
    
    # load tokenized sentences from source. we need to do this to ensure all the doc's sentences are 
    # present, in order, so they can be matched with POS tags
    with codecs.open(os.path.join(reviewsDP, 'source', 'source_text_ascii_tokenized', shortDocID+'.txt')) as sentF:
        for ln in sentF:
            sent = ln[ln.index('>')+1:].strip()    
            words = sent.split()
            doc_tagged_words.append(list(zip(words, doc_poses)))
            doc_poses = doc_poses[len(words):]
            
    
    assert not doc_poses,('Leftover POSes:',doc_poses)

for sentID, docID, sentOffset, anno_, data in indata:
    mweTokOffsets = set()
    for g in data['_']+data['~']:
        mweTokOffsets.update(g)
    doc_tagged_words = data["words"] = tagged_words[docID][sentOffset-1]
    assert 0 not in mweTokOffsets # they should be 1-based!
    assert (' '+anno_+' ').count(' . ')==len(data["words"])-len(mweTokOffsets),(mweTokOffsets,anno_)
    parts = anno_.strip().split()
    i = 0 # token index, 0-based
    for j in range(len(parts)):
        if parts[j]=='.':
            assert i < len(doc_tagged_words),(i,doc_tagged_words,mweTokOffsets)
            parts[j] = doc_tagged_words[i][0]
            i += 1
        while i+1 in mweTokOffsets: i += 1
    anno = ' '.join(parts)

    print(sentID, anno, json.dumps(data), sep='\t')
