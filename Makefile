PY := .venv/bin/python

.PHONY: setup data verify bench eval test all-phase1

setup:  ## install system deps, Ollama, models, python env
	bash scripts/setup_ubuntu.sh

data:  ## download RouteLLM dataset + build labels/splits
	$(PY) scripts/prepare_dataset.py

verify:  ## Phase 1 exit-criteria check
	$(PY) scripts/verify_phase1.py

bench:  ## cold vs warm latency of the router SLM
	$(PY) scripts/benchmark_slm.py

eval:  ## zero-shot SLM routing accuracy on held-out prompts
	$(PY) scripts/eval_slm_routing.py --n 200

test:  ## offline unit tests
	$(PY) -m unittest discover -s tests -v

all-phase1: data verify bench eval
