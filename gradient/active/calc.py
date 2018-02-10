from active_gb import ActLearning
from random_gb import RandomLearning
import multiprocessing as mulp

AFILE='hct116.episgt' 
ACNT=4239
BFILE='hek293t.episgt' 
BCNT=4666
CFILE='hela.episgt' 
CCNT=8101
DFILE='hl60.episgt'
DCNT=2076
LPATH='../../dataset/'

queue = mulp.Queue(4)
p1=mulp.Process(target=RandomLearning,args=(LPATH,AFILE,ACNT,16,'c','1'))
p2=mulp.Process(target=ActLearning,args=(LPATH,AFILE,ACNT,1.0/4,16,0,1,'r','1'))
p3=mulp.Process(target=RandomLearning,args=(LPATH,BFILE,BCNT,16,'c','1'))
p4=mulp.Process(target=ActLearning,args=(LPATH,BFILE,BCNT,1.0/4,16,0,1,'r','1'))
p5=mulp.Process(target=RandomLearning,args=(LPATH,CFILE,CCNT,16,'c','1'))
p6=mulp.Process(target=ActLearning,args=(LPATH,CFILE,CCNT,1.0/4,16,0,1,'r','1'))
p7=mulp.Process(target=RandomLearning,args=(LPATH,DFILE,DCNT,16,'c','1'))
p8=mulp.Process(target=ActLearning,args=(LPATH,DFILE,DCNT,1.0/4,16,0,1,'r','1'))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()

queue.close()

