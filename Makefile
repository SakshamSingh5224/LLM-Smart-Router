PY := .venv/bin/python

.PHONY: setup data verify bench eval train-router eval-router serve-router serve-gateway verify3 test all-phase1 all-phase2 migrate migrate-new migrate-stamp

setup:  ## install system deps, Ollama, models, python env
	bash scripts/setup_ubuntu.sh

data:  ## download RouteLLM dataset + build labels/splits
	$(PY) scripts/prepare_dataset.py

verify:  ## Phase 1 exit-criteria check
	$(PY) scripts/verify_phase1.py

bench:  ## cold vs warm latency of the router SLM
	$(PY) scripts/benchmark_slm.py

eval:  ## zero-shot SLM routing accuracy on held-out prompts (Phase 1 baseline)
	$(PY) scripts/eval_slm_routing.py --n 200

train-router:  ## train + calibrate the routing classifier (Phase 2)
	$(PY) scripts/train_router.py

eval-router:  ## cost-vs-quality curve on the held-out TEST split (Phase 2)
	$(PY) scripts/eval_router_test.py

serve-router:  ## run the standalone router microservice on :8001
	$(PY) -m uvicorn router_service.app:app --host 0.0.0.0 --port 8001

serve-gateway:  ## run the gateway + frontend on :8000
	$(PY) -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000

verify3:  ## Phase 3 exit-criteria check (gateway must already be running)
	$(PY) scripts/verify_phase3.py

test:  ## offline unit tests
	$(PY) -m unittest discover -s tests -v

migrate:  ## apply all pending Alembic migrations to whatever DATABASE_URL points at
	.venv/bin/alembic upgrade head

migrate-new:  ## autogenerate a new migration from model changes (edit before applying!); usage: make migrate-new m="add foo column"
	.venv/bin/alembic revision --autogenerate -m "$(m)"

migrate-stamp:  ## mark an EXISTING database (tables already created by create_all) as already at baseline, with no DDL run
	.venv/bin/alembic stamp head

all-phase1: data verify bench eval

all-phase2: train-router eval-router

.PHONY: mvp3c-test mvp3c-calibrate
mvp3c-test:
	PYTHONPATH=. $(PY) -m scripts.test_mvp3c "$(QUERY)"

mvp3c-calibrate:
	PYTHONPATH=. $(PY) -m scripts.calibrate_mvp3c
