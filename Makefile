PYTHON   ?= python3
CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-result
NVCC     ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_86 -Xcompiler -fopenmp
WORKERS  ?= 24

CERT = certification
SRCH = search
CPU_BINS = $(addprefix $(SRCH)/, fast_mertens_enum fast_mertens_midband fast_mertens_deep fast_mertens_deep_dir2 \
           fast_mertens_coset fast_mertens_coset_dir1 fast_mertens_coset_3flip libpath1_eval.so)

.PHONY: all certify lemma7 k2 zero-cache table1 check check-only bounds search gpu zeros clean clean-cache

all: certify

# ---------------------------------------------------------------- rigorous certification (THEOREM.md)
certify: check            # check depends on the generators, so this order also holds under make -j

lemma7:
	cd $(CERT) && $(PYTHON) verify_lemma7.py

k2:                       # all 2,325,572 zeros below 1.3e6; ~45 min on 24 cores, resumable
	cd $(CERT) && $(PYTHON) certify_K2.py $(WORKERS)

table1:                   # 6.2M certified zeros on [1.3e6, 4.3e6]; ~50 min on 16 cores, resumable
	cd $(CERT) && $(PYTHON) verify_table1.py 1300000 4300000 $(WORKERS)

zero-cache:               # certified zeros + zeta'(rho) below T0, cached once; makes `k2` for a new point take minutes
	cd $(CERT) && $(PYTHON) certify_K2.py $(WORKERS) --cache-only

check: lemma7 k2 table1   # combine certificates, print the theorem's conclusion (runs only after all three)
	cd $(CERT) && $(PYTHON) check_theorem.py

check-only:               # re-check existing certificates without regenerating them
	cd $(CERT) && $(PYTHON) check_theorem.py

bounds:                   # error terms only (Saouter-te Riele variant, T1 = 1e7, M = 23.05)
	cd $(CERT) && $(PYTHON) error_bounds.py

# ---------------------------------------------------------------- candidate search (not needed to verify)
search: $(CPU_BINS)

$(SRCH)/libpath1_eval.so: $(SRCH)/path1_eval_fast.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $<

$(SRCH)/%: $(SRCH)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu: $(SRCH)/gpu_mertens_stream

$(SRCH)/gpu_mertens_stream: $(SRCH)/gpu_mertens_stream.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

zeros:                    # 600k Newton-refined zeros used by the search (needs search/zeros6.gz from Odlyzko)
	cd $(SRCH) && $(PYTHON) fast_flint_zeros.py 600000

clean:
	rm -f $(CPU_BINS) $(SRCH)/gpu_mertens_stream
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-cache:              # forces certify_K2 / verify_table1 to recompute from scratch
	rm -rf $(CERT)/k2_chunks $(CERT)/table1_chunks $(CERT)/zero_cache
