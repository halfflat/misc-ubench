.PHONY: clean all
.SECONDARY:

topdir:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
srcdir:=$(topdir)
gbench_top:=$(topdir)gbench

OPTFLAGS?=-O3 -march=native
CXXFLAGS+=$(OPTFLAGS) -MMD -MP -std=c++14 -g -pthread
CPPFLAGS+=-I $(srcdir) -isystem $(gbench_top)/include
LDLIBS+=-lbenchmark

all:: qappend

define cxx-compile
$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ -c $<
endef

define cxx-link
$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
endef

# gbenchmark:

gbench_sources:=$(wildcard $(gbench_top)/src/*.cc)
gbench_objects:=$(patsubst %.cc,o/gbench/%.o,$(notdir $(gbench_sources)))
gbench_depends:=$(patsubst %.cc,o/gbench/%.d,$(notdir $(gbench_sources)))

-include $(gbench_depends)

o/gbench/%.o: CPPFLAGS+=-DHAVE_STD_REGEX -DNDEBUG
o/gbench/%.o: $(gbench_top)/src/%.cc
	@mkdir -p o/gbench
	$(cxx-compile)

libbenchmark.a: $(gbench_objects)
	$(AR) r $@ $^

# qappend:

qappend_sources:=$(srcdir)/qbench/qappend.cc
qappend_objects:=$(patsubst %.cc,o/qappend/%.o,$(notdir $(qappend_sources)))
qappend_depends:=$(patsubst %.cc,o/qappend/%.d,$(notdir $(qappend_sources)))

-include $(qappend_depends)

o/qappend/%.o: $(srcdir)/qappend/%.cc
	mkdir -p o/qappend
	$(cxx-compile)

qappend: $(qappend_objects) libbenchmark.a
	$(cxx-link)

clean:
	rm -f $(qappend_objects) $(gbench_objects)

realclean: clean
	rm -f qappend libbenchmark.a $(qappend_depends) $(gbench_depends)
	for dir in o/qappend o/gbench o; do [ -d "$$dir" ] && rmdir "$$dir"; done
