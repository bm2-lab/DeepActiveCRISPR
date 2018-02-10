from active_gb import ActLearning
from random_gb import RandomLearning
import multiprocessing as mulp

#LFILE='hct116.episgt' 
#LCNT=4239
RFILE='hek293t.episgt' 
RCNT=4666
LFILE='hela.episgt' 
LCNT=8101
#LFILE='hl60.episgt'
#LCNT=2076
LPATH='../../dataset/'

queue = mulp.Queue(4)
p1=mulp.Process(target=RandomLearning,args=(LPATH,LFILE,LCNT,16,'c','1'))
p2=mulp.Process(target=ActLearning,args=(LPATH,LFILE,LCNT,1.0/4,16,0,1,'r','1'))
p3=mulp.Process(target=RandomLearning,args=(LPATH,RFILE,RCNT,16,'c','1'))
p4=mulp.Process(target=ActLearning,args=(LPATH,RFILE,RCNT,1.0/4,16,0,1,'r','1'))
#p3=mulp.Process(target=ActLearning,args=('hela.episgt',8101,))
#p4=mulp.Process(target=ActLearning,args=('hl60.episgt',2076,))
p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()
queue.close()

#ActLearning('hl60.episgt',2076)
#RandomLearning('hl60.episgt',2076)
