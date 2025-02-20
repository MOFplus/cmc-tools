import time

from molsys.util.timer import Timer, timer


class TimerDemo:

    def __init__(self):
       self.timer = Timer("Test timer")

    @timer("Time subroutine")
    def wait(self,time_to_sleep):
        time.sleep(time_to_sleep)


    def run(self):
       with self.timer as mt:
           time.sleep(6)
           with mt.fork("sub timer 1") as subtimer1:
               time.sleep(0.001)
           with mt.fork("sub timer 2") as subtimer2:
               time.sleep(2)
               with subtimer2.fork("subsub timer") as subsubtimer:
                   time.sleep(1)
           time.sleep(1)
           with mt.fork("sub timer 3") as subtimer3:
               time.sleep(1)
       
           self.wait(2) 
           self.wait(6) 
 
       with self.timer('Monitor block of code'):
           x = 2 + 2
           time.sleep(4)
       
       self.timer.report()



timer_test = TimerDemo()
timer_test.run()

