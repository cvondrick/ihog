mexobjects=cppLasso.out 

headers=../linalg/misc.h ../linalg/cblas_template.h \
	../linalg/misc.h ../linalg/linalg.h \
       	../decomp/decomp.h 

INCLUDE+=-I../ -I../linalg/ -I../decomp/

all:
	rm -f compile*
	$(MAKE) lib
	$(MAKE) $(mexobjects)
	mv $(LIBNAME) ../build/

lib:
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(NAME) lib.o lib.cpp
	ar -cvr $(LIBNAME) lib.o
	rm lib.o

clean:
	rm -f *.o *.a *.out

%.o : %.cpp $(headers)
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(NAME)$@ $<

%.out: %.cpp $(headers)
ifeq ($(VER),linuxmac)
	$(MAKE) $*.o
	echo "$(CXX) $(CPPLINK) $*.o $(INCLUDELIB) $(LIBS) -lSPAMS -o $@" >> "compile.sh"
	$(CXX) $(CPPLINK) $*.o -lSPAMS $(INCLUDELIB) $(LIBS) -o $@ 
	mv $*.out $*
	mv $* ../build/
	mv compile* ../build/
else
   #TODO
endif

