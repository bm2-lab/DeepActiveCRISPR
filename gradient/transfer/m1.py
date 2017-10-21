from trans import Transfer


LFILE=['hct116.episgt','hek293t.episgt','hela.episgt','hl60.episgt','label.dat']
LCNT=[4239,4666,8101,2076,19082]

for j in range(0,4):
    for i in range(0,4):
        Transfer(LFILE[i],LCNT[i],LFILE[j],LCNT[j])
    print("")

#for i in range(4,5):
#    Transfer(LFILE[i],LCNT[i],LFILE[i],LCNT[i])