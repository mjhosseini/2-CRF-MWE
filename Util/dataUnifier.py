import codecs
source = '../streusle-2.0'
gold1 = codecs.open(source+'/split/forNathan/test.gold.sst','r','utf-8')
gold2 = codecs.open(source+'/split/nathan/test.gold.sst','r','utf-8')
gold2w = codecs.open(source+'/split/nathan/test2.gold.sst','w','utf-8')
pred2 = codecs.open(source+'/split/nathan/test.pred.sst','r','utf-8')
pred2w = codecs.open(source+'/split/nathan/test2.pred.sst','w','utf-8')

ids = []
for line in gold1:
    id = line.split("\t")[0]
    ids.append(id)

g2ls = {}
p2ls = {}
for i in range(len(ids)):
    l1 = gold2.readline()
    l2 = pred2.readline()

    g2ls[l1.split("\t")[0]] = l1
    p2ls[l2.split("\t")[0]] = l2

for i in range(len(ids)):
    gold2w.write(g2ls[ids[i]])
    pred2w.write(p2ls[ids[i]])


