.PHONY: clean all
.SECONDARY:

benches:=comment-regex round-up small-vec-search wrong-stride io-to-str min-interval indirect-sum isqrt
#cu_benches:=cuda-reduce-by-key

topdir:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
srcdir:=$(topdir)
gbench_top:=$(topdir)gbench

OPTFLAGS?=-O3 -march=native
CXXFLAGS+=$(OPTFLAGS) -MMD -MP -std=c++14 -g -pthread
CPPFLAGS+=-isystem $(gbench_top)/include

NVCC?=nvcc
NVCCFLAGS+=-O3 --std=c++14 -arch=sm_60

all:: $(benches)

define cxx-compile
$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ -c $<
endef

define cxx-link
$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
endef

define nvcc-compile
$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -o $@ -c $<
endef

define nvcc-makedepend
$(NVCC) $(NVCCFLAGS) -MM -MF $(patsubst %.o,%.d,$@) $(CPPFLAGS) $<
endef

define nvcc-link
$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
endef


# Arguments: name source-directory
define obj_template
$(1)_cc_sources:=$$(wildcard $(2)/*.cc)
$(1)_cc_objects:=$$(patsubst %.cc,o/$(1)/%.o,$$(notdir $$($(1)_cc_sources)))

$(1)_cu_sources:=$$(wildcard $(2)/*.cu)
$(1)_cu_objects:=$$(patsubst %.cu,o/$(1)/%.o,$$(notdir $$($(1)_cu_sources)))

$(1)_objects:=$$($(1)_cc_objects) $$($(1)_cu_objects)

$(1)_depends:=$$(patsubst %.cc,o/$(1)/%.d,$$(notdir $$($(1)_cc_sources)))
$(1)_depends+=$$(patsubst %.cu,o/$(1)/%.d,$$(notdir $$($(1)_cu_sources)))

clean_objs+=$$($(1)_objects)
clean_deps+=$$($(1)_depends)
clean_dirs+=o/$(1)

-include $$($(1)_depends)
o/$(1)/%.o: $(2)/%.cc
	@mkdir -p o/$(1)
	$$(cxx-compile)

o/$(1)/%.d: $(2)/%.cu
	@mkdir -p o/$(1)
	$$(nvcc-makedepend)

o/$(1)/%.o: $(2)/%.cu
	@mkdir -p o/$(1)
	$$(nvcc-compile)
endef

# gbenchmark:

$(eval $(call obj_template,gbench,$(gbench_top)/src))
o/gbench/%.o: CPPFLAGS+=-DHAVE_STD_REGEX -DNDEBUG

libbenchmark.a: $(gbench_objects)
	$(AR) r $@ $^


# All benchmarks:

#wrong-stride: CPPFLAGS+=-DPAD
wrong-stride: CPPFLAGS+=-DEXPENSIVE
wrong-stride: CXXFLAGS+=-fopenmp

define bench_template
$$(eval $$(call obj_template,$(1),$$(srcdir)/$(1)))
$(1): libbenchmark.a
$(1): $$($(1)_objects) libbenchmark.a
	$$(cxx-link)
endef

define cuda_bench_template
$$(eval $$(call obj_template,$(1),$$(srcdir)/$(1)))
$(1): libbenchmark.a
$(1): $$($(1)_objects) libbenchmark.a
	$$(nvcc-link)
endef

$(foreach b,$(benches),$(eval $(call bench_template,$(b))))
$(foreach b,$(cu_benches),$(eval $(call cuda_bench_template,$(b))))


# Clean up:

clean:
	rm -f $(clean_objs)

realclean: clean
	rm -f $(benches) libbenchmark.a $(clean_deps)
	for dir in $(clean_dirs); do if [ -d "$$dir" ]; then rmdir "$$dir"; fi; done
