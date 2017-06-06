import codecs
import sys

#source = '../pysupersensetagger-2.0/dimsum-dataAnalyzer-1.5'
#f1 = codecs.open(source+'/split/dimsum16.train_allT','r','utf-8')
#f2 = codecs.open(source+'/split/dimsum16.train_allT.tags','w','utf-8')

args = sys.argv[1:]
print args

f1 = codecs.open(args[0],'r','utf-8')
f2 = codecs.open(args[1],'w','utf-8')

def dimsum2streusle(l):
    #print "l: ", l
    parts = l.split("\t")
    if (len(parts)==1):
        return l
    elif (len(parts)==9):
        ss = parts[7]
        #print "ss: ",ss
        parts2 = ss.split(".")
        if (len(parts2)==1):
            return l.strip()+"\n"
        pos = parts2[0]
        sst = parts2[1]
        if pos=="n":
            parts[4] += "-"+sst.upper()
            parts[7] = sst.upper()
        elif pos=="v":
            parts[4] += "-"+sst
            parts[7] = sst
        else:
            print "WRONG WAY"
            return None

        ret = ""
        for i in range(len(parts)):
            ret += parts[i] + "\t"
        return ret.strip()+"\n"

    else:
        print "WRONG WAY!", len(parts)
        return None

for l in f1:
    f2.writelines([dimsum2streusle(l)])

