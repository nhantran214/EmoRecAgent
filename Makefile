.PHONY: install install-dev test data absa-targets absa absa-retry absa-preview warmup-absa absa-benchmark absa-quality absa-quality-compare sample-gold compare neo4j load-kg experiment experiment-paper lightgcn-paper xsimgcl-paper core-paper tisasrec-paper agent4rec-setup agent4rec-setup-py39 agent4rec-paper agent4rec-paper-debug agent4rec-paper-py39 ablations build-hgt-graph train-hgt experiment-hgt clean clean-absa

# Use the active interpreter (e.g. conda activate ERA) or override: make test PYTHON=...
PYTHON ?= python3
export PYTHONPATH := src
LOG_DIR ?= logs

install:
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"

test: install-dev
	$(PYTHON) -m pytest -q

data:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/build_dataset.py --config configs/default.yaml --log-dir $(LOG_DIR)

absa-targets:
	$(PYTHON) scripts/export_absa_targets.py --config configs/default.yaml --log-dir $(LOG_DIR)

absa:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/run_absa.py --config configs/default.yaml --log-dir $(LOG_DIR)

absa-retry:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/run_absa.py --config configs/default.yaml --log-dir $(LOG_DIR) --retry-errors

absa-preview:
	$(PYTHON) scripts/preview_absa.py --config configs/default.yaml

warmup-absa:
	$(PYTHON) scripts/warmup_absa_checkpoint.py --config configs/default.yaml

absa-benchmark:
	$(PYTHON) scripts/benchmark_absa_latency.py --config configs/default.yaml

absa-quality-compare:
	$(PYTHON) scripts/compare_absa_quality.py --config configs/default.yaml

clean-absa:
	rm -f data/processed/Beauty_and_Personal_Care/absa_cache.sqlite
	rm -f data/processed/Beauty_and_Personal_Care/absa_cache.cache_manifest.json
	rm -f data/processed/Beauty_and_Personal_Care/absa_cache.sqlite-journal
	rm -f $(LOG_DIR)/absa_errors.jsonl

sample-gold:
	$(PYTHON) scripts/sample_absa_gold.py --config configs/default.yaml

absa-quality:
	$(PYTHON) scripts/eval_absa_quality.py --config configs/default.yaml

compare:
	$(PYTHON) scripts/compare_results.py \
		--a $(A) --b $(B) --metric $(METRIC)

neo4j:
	bash scripts/setup_docker_neo4j.sh
	$(PYTHON) scripts/verify_neo4j.py

load-kg: neo4j
	$(PYTHON) scripts/load_kg.py --config configs/default.yaml

# HGT pipeline (ABSA cache is read-only; do NOT run clean-absa before this).
build-hgt-graph:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/build_hgt_graph.py --config configs/default.yaml --log-dir $(LOG_DIR)

train-hgt:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/train_hgt.py --config configs/default.yaml --log-dir $(LOG_DIR)

experiment-hgt:
	@mkdir -p $(LOG_DIR) results
	$(PYTHON) scripts/run_experiment.py \
		--config configs/default.yaml \
		--method emorecagent_hgt \
		--split data/processed/Beauty_and_Personal_Care \
		--out results/emorecagent_hgt.json \
		--log-dir $(LOG_DIR)

experiment:
	@mkdir -p $(LOG_DIR) results
	$(PYTHON) scripts/run_experiment.py \
		--config configs/default.yaml \
		--method svd \
		--split data/processed/Beauty_and_Personal_Care \
		--out results/svd.json \
		--log-dir $(LOG_DIR)

# Protocol B — paper-aligned baseline comparison (user_batch, full catalog, macro user-mean).
# METHOD=svd|itemknn|popularity (default: svd)
experiment-paper:
	@mkdir -p $(LOG_DIR) results/paper
	$(PYTHON) scripts/run_experiment.py \
		--config configs/paper_baseline.yaml \
		--method $(or $(METHOD),svd) \
		--split data/processed/Beauty_and_Personal_Care \
		--out results/paper/$(or $(METHOD),svd).json \
		--log-dir $(LOG_DIR)

# Protocol B — LightGCN paper-aligned eval (user_batch, full catalog, macro user-mean @20).
lightgcn-paper:
	@mkdir -p baseline/LightGCN-PyTorch/logs results/paper
	$(PYTHON) baseline/LightGCN-PyTorch/amazon/run_experiment.py \
		--config baseline/LightGCN-PyTorch/configs/paper_lightgcn.yaml \
		--out results/paper/lightgcn.json

# Protocol B — XSimGCL paper-aligned eval (user_batch, full catalog, macro user-mean @20).
xsimgcl-paper:
	@mkdir -p baseline/SimGCL-MixGCF/logs results/paper
	$(PYTHON) baseline/SimGCL-MixGCF/amazon/run_experiment.py \
		--config baseline/SimGCL-MixGCF/configs/paper_xsimgcl.yaml \
		--out results/paper/xsimgcl.json

# Protocol B — CORE paper-aligned eval (user_batch, full catalog, macro user-mean @20).
core-paper:
	@mkdir -p baseline/CORE/logs results/paper
	$(PYTHON) baseline/CORE/amazon/run_experiment.py \
		--config baseline/CORE/configs/paper_core.yaml \
		--out results/paper/core.json

# Protocol B — TiSASRec paper-aligned eval (user_batch, full catalog, macro user-mean @20).
tisasrec-paper:
	@mkdir -p baseline/TiSASRec.pytorch/logs results/paper
	$(PYTHON) baseline/TiSASRec.pytorch/amazon/run_experiment.py \
		--config baseline/TiSASRec.pytorch/configs/paper_tisasrec.yaml \
		--out results/paper/tisasrec.json

# Protocol B — Agent4Rec full pipeline (train/valid/test CF + paper-scale simulation + comparison eval).
# ERA (Python 3.11+): pip install -e ".[dev]" then make agent4rec-setup
agent4rec-setup:
	$(PYTHON) -m pip install -r baseline/Agent4Rec/requirements-amazon.txt
	cd baseline/Agent4Rec/recommenders && $(PYTHON) setup.py build_ext --inplace

# A4R-baseline (Python 3.9): no root emorecagent install — see py39/README.md
agent4rec-setup-py39:
	$(PYTHON) -m pip install -r baseline/Agent4Rec/py39/requirements-torch-cu128.txt \
		--index-url https://download.pytorch.org/whl/cu128
	$(PYTHON) -m pip install -r baseline/Agent4Rec/py39/requirements.txt
	cd baseline/Agent4Rec/recommenders && $(PYTHON) setup.py build_ext --inplace

agent4rec-paper: agent4rec-setup
	@mkdir -p baseline/Agent4Rec/logs results/paper baseline/Agent4Rec/runs
	PYTHONUNBUFFERED=1 $(PYTHON) baseline/Agent4Rec/amazon/run_experiment.py \
		--config baseline/Agent4Rec/configs/paper_agent4rec.yaml \
		--out results/paper/agent4rec.json

agent4rec-paper-debug: agent4rec-setup
	@mkdir -p baseline/Agent4Rec/logs results/paper baseline/Agent4Rec/runs
	PYTHONUNBUFFERED=1 $(PYTHON) baseline/Agent4Rec/amazon/run_experiment.py \
		--config baseline/Agent4Rec/configs/paper_agent4rec_debug.yaml \
		--out results/paper/agent4rec_debug.json

agent4rec-paper-py39: agent4rec-setup-py39
	@mkdir -p baseline/Agent4Rec/logs results/paper baseline/Agent4Rec/runs
	PYTHONUNBUFFERED=1 $(PYTHON) baseline/Agent4Rec/amazon/run_experiment.py \
		--config baseline/Agent4Rec/configs/paper_agent4rec.yaml \
		--out results/paper/agent4rec.json

# Protocol B — AgentCF full pipeline (RecBole train + native eval + comparison eval).
# Requires conda env with recbole (e.g. A4R-baseline). Comparison eval needs ERA 3.11+ / emorecagent.
agentcf-setup:
	$(PYTHON) -m pip install -r baseline/AgentCF/requirements-amazon.txt

agentcf-paper: agentcf-setup
	@mkdir -p baseline/AgentCF/logs results/paper baseline/AgentCF/checkpoints
	PYTHONUNBUFFERED=1 $(PYTHON) baseline/AgentCF/amazon/run_experiment.py \
		--config baseline/AgentCF/configs/paper_agentcf.yaml \
		--out results/paper/agentcf.json

agentcf-paper-debug: agentcf-setup
	@mkdir -p baseline/AgentCF/logs results/paper baseline/AgentCF/checkpoints
	PYTHONUNBUFFERED=1 $(PYTHON) baseline/AgentCF/amazon/run_experiment.py \
		--config baseline/AgentCF/configs/paper_agentcf_debug.yaml \
		--out results/paper/agentcf_debug.json

ablations:
	@mkdir -p $(LOG_DIR) results/ablations
	@for cfg in configs/ablations/*.yaml; do \
		name=$$(basename $$cfg .yaml); \
		echo "[ablation] $$name"; \
		$(PYTHON) scripts/run_experiment.py \
			--config $$cfg \
			--method emorecagent \
			--split data/processed/Beauty_and_Personal_Care \
			--out results/ablations/$$name.json \
			--log-dir $(LOG_DIR); \
	done

clean: clean-absa
	rm -rf .pytest_cache htmlcov .coverage results/*.json
