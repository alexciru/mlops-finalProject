.PHONY: clean data lint format requirements sync_data_down sync_data_up

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = mlops-finalProject
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
lint:
	flake8 mlops_finalproject
	black --check --config pyproject.toml mlops_finalproject


## Format source code with black
format:
	black --config pyproject.toml mlops_finalproject




## Set up python interpreter environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) mlops_finalproject/data/make_dataset.py  data/raw/German/  data/processed/


# locally
train: 
	$(PYTHON_INTERPRETER) mlops_finalproject/models/train_model.py



predict:
    $(PYTHON_INTERPRETER) python mlops_finalproject/models/predict_model.py trained_model_32img_20230118_1729.pt data\raw\German\Test


#cloud - run training
train_cloud_cpu: 
	gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=config_cpu.yaml



train_cloud_gpu: 
	gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=config_gpu.yaml


# deploy pytorch server locally
pytorch_server_building:
	docker build -f inference.dockerfile --tag=europe-west1-docker.pkg.dev/mlops-finalproject/inference-mnet/serve-mnet .


pytorch_server_run:
	docker run -p 8080:8080 --name=local_mnet_inference99 europe-west1-docker.pkg.dev/mlops-finalproject/inference-mnet/serve-mnet


# Create bucket and deploy
deploy:
	gcloud auth configure-docker europe-west1-docker.pkg.dev
	docker push europe-west1-docker.pkg.dev/mlops-finalproject/inference-mnet/serve-mnet
	gcloud beta ai-platform models create demo_mnet --region=europe-west1  --enable-logging  --enable-console-logging



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
