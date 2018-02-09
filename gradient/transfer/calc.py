from transfer_gb import Transfer


LFILE=['hct116.episgt','hek293t.episgt','hela.episgt','hl60.episgt']
LCNT=[4239,4666,8101,2076]
LPATH='../../dataset/'
for i in range(0,4):
	LFILE[i]=LPATH+LFILE[i]

for j in range(0,4):
    for i in range(0,4):
        Transfer(LFILE[i],LCNT[i],LFILE[j],LCNT[j])
    print("")

