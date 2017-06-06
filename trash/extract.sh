set -eu

python2.7 src/main.py --cutoff 5 --YY tagsets/bio2gNV --defaultY O --debug --extract /Users/hosseini/Desktop/D/research/MWE/data/streusle-2.1/streusle.upos.tags dataset --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json