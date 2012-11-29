import sys
import time

all_modules = ['linalg', 'decomp', 'prox', 'dictLearn']

modules = []

for s in all_modules:
    try:
        exec ('import test_%s' %s)
        modules.append(s)
    except:
        print "Removing %s" %s
# for debug
simul = False

def usage():
    print "Usage : %s [test-or-group-name]+" %sys.argv[0]
    print '  Run specified test or group of tests (all by default)'
    print '  Available groups and tests are:'
    for m in modules:
        print "%s :" %m
        exec('lstm = test_%s.tests' %m)
        print '  %s' %(' '.join([ lstm[i] for i in xrange(0,len(lstm),2)]))
    print '\nExamples:'
    print '%s linalg' %sys.argv[0]
    print '%s sort calcAAt' %sys.argv[0]
    sys.exit(1)

def run_test(testname,prog):
    print "** Running %s" %testname
    if simul:
        return
    err = prog()
    if err != None:
        print "  ERR = %f" %err

def main(argv):
    tic = time.time()
    lst = []
    for s in argv:
        if s[0] == '-':
            usage()
        lst.append(s)
    if(len(lst) == 0):
        lst = modules
    for testname in lst:
        if testname in modules:
            print "**** %s ****" %testname
            exec('lstm = test_%s.tests' %testname)
            for i in xrange(0,len(lstm),2):
                run_test(lstm[i],lstm[i+1])
            continue
        else:
            found = False
            for m in modules:
                exec('lstm = test_%s.tests' %m)
                for i in xrange(0,len(lstm),2):
                    if (lstm[i] == testname):
                        found = True
                        run_test(lstm[i],lstm[i+1])
                        break
                if found:
                    break

            if(not found):
                print "Test %s not found!" %testname

    tac = time.time()
    print '\nTotal time : %.3fs'  %(tac - tic)

if __name__ == "__main__":
    main(sys.argv[1:])
