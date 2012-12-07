import ihog
import scipy.io
import cPickle

print "reading mat file..."
mat = scipy.io.loadmat("../pd-v7.mat")

sbin = mat['sbin'][0][0]
lambda1 = mat['lambda'][0][0]
k = mat['k'][0][0]
n = mat['n'][0][0]
ny = mat['ny'][0][0]
nx = mat['nx'][0][0]
iters = mat['iters'][0][0]

print "k =", k
print "n =", n
print "ny =", ny
print "nx =", nx

dhog = mat['dhog']
dgray = mat['dgray']

print "creating paired dictionary..."

pd = ihog.PairedDictionary(dgray, dhog, n, k, ny, nx, lambda1, sbin)

print "writing to disk..."
cPickle.dump(pd, open("pd.pkl", "w"))

print "successfully converted"
