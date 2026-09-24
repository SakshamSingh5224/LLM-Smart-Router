.PHONY: test data train eval-classifier serve clean

test:
	pytest tests/

data:
	python3 scripts/prepare_dataset.py

train:
	python3 scripts/train_classifier.py

eval-classifier:
	python3 scripts/eval_classifier_routing.py

serve:
	uvicorn smartrouter.api:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf results/*.joblib results/*.json results/*.csv
