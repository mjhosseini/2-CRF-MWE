import codecs
import sys

#source = '../pysupersensetagger-2.0/dimsum-dataAnalyzer-1.5'
# f2 = codecs.open(source+'/split/dimsum16.train_allT0','w','utf-8')
# f1 = codecs.open(source+'/split/dimsum16.train_allT.tags','r','utf-8')


def dimsum2streusle(l):
    #print "l: ", l
    parts = l.split("\t")
    if (len(parts)==1):
        return l
    elif (len(parts)==9):
        bio_ss = parts[4]
        #print "bio_ss: ",bio_ss
        parts2 = bio_ss.split("-")
        if (len(parts2)==1):
            return l.strip()+"\n"
        bio = parts2[0]
        sst = parts2[1]
        parts[4] = parts2[0]

        if bio.upper()=="I":
            parts[7] = ""
        else:
            if sst.upper() == sst:#This means it is a noun
                parts[7] = "n."+sst.lower()
            else:
                parts[7] = "v."+sst.lower()




        ret = ""
        for i in range(len(parts)):
            ret += parts[i] + "\t"
        return ret.strip()+"\n"

    else:
        print "WRONG WAY!", len(parts)
        return None

def s2d():
    args = sys.argv[1:]
    f1 = codecs.open(args[0],'r','utf-8')
    f2 = codecs.open(args[1],'w','utf-8')

    for l in f1:
        f2.writelines([dimsum2streusle(l)])

if __name__=='__main__':

    args = sys.argv[1:]

    f1 = codecs.open(args[0],'r','utf-8')
    f2 = codecs.open(args[1],'w','utf-8')

    for l in f1:
        f2.writelines([dimsum2streusle(l)])


