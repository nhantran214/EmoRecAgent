.PHONY: install install-dev test data absa-targets absa absa-classical absa-retry absa-preview warmup-absa absa-benchmark absa-quality absa-quality-compare sample-gold compare neo4j tgi tgi-absa load-kg experiment experiment-paper lightgcn-paper xsimgcl-paper core-paper tisasrec-paper agent4rec-setup agent4rec-setup-py39 agent4rec-paper agent4rec-paper-debug agent4rec-paper-py39 ablations check-emorecagent-prereqs check-emorecagent-eval-prereqs check-emorecagent-stage1-test-prereqs train-emorecagent precompute-tu-emorecagent train-align-emorecagent build-cross-user-lookup-emorecagent test-emorecagent experiment-emorecagent experiment-emorecagent-stage1-baseline compare-emorecagent-stage2 compare-retrieval compare-batch-parity clean clean-absa

# Category — same method/hyperparameters; only data + artifact paths change.
# Supported: Beauty_and_Personal_Care (default), Sports_and_Outdoors, Toys_and_Games,
#            Yelp (review track), Yelp_AC (AC-TSR no-review / RecBole track)
# Example: make data CATEGORY=Sports_and_Outdoors
#          make data CATEGORY=Yelp      # review + ABSA track
#          make data CATEGORY=Yelp_AC   # ID-only AC-TSR; skips ABSA
# Caches are isolated per category under data/processed/$(CATEGORY)/ — never shared.
# Do not reuse data/processed/Yelp for Yelp_AC (or vice versa).
CATEGORY ?= Beauty_and_Personal_Care
ABSA_WORKERS ?= 32
SPLIT_DIR := data/processed/$(CATEGORY)

ifeq ($(CATEGORY),Beauty_and_Personal_Care)
  CONFIG := configs/default.yaml
  ALIGN_CONFIG := configs/emorecagent_align.yaml
  BASELINE_CONFIG := configs/emorecagent_stage1_baseline.yaml
  RESULTS_DIR := results
  LOG_DIR ?= logs
else
  CONFIG := configs/categories/$(CATEGORY).yaml
  ALIGN_CONFIG := configs/categories/$(CATEGORY)_emorecagent_align.yaml
  BASELINE_CONFIG := configs/categories/$(CATEGORY)_emorecagent_stage1_baseline.yaml
  RESULTS_DIR := results/$(CATEGORY)
  LOG_DIR ?= logs/$(CATEGORY)
endif

# Use the active interpreter (e.g. conda activate ERA) or override: make test PYTHON=...
PYTHON ?= python3
export PYTHONPATH := src
PROGRESS_INTERVAL ?= 25

install:
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"

test: install-dev
	$(PYTHON) -m pytest -q

data:
	@mkdir -p $(LOG_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(CONFIG) SPLIT_DIR=$(SPLIT_DIR)"
	$(PYTHON) scripts/build_dataset.py --config $(CONFIG) --log-dir $(LOG_DIR)

absa-targets:
	$(PYTHON) scripts/export_absa_targets.py --config $(CONFIG) --log-dir $(LOG_DIR)

absa:
	@mkdir -p $(LOG_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(CONFIG) ABSA_WORKERS=$(ABSA_WORKERS)"
	$(PYTHON) scripts/run_absa.py --config $(CONFIG) --log-dir $(LOG_DIR) --workers $(ABSA_WORKERS)

absa-classical:
	@mkdir -p $(LOG_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(CONFIG) backend=classical device=cuda batch_size=$(or $(BATCH_SIZE),32)"
	$(PYTHON) scripts/run_absa.py --config $(CONFIG) --log-dir $(LOG_DIR) --classical-only --device cuda --batch-size $(or $(BATCH_SIZE),32)

absa-retry:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/run_absa.py --config $(CONFIG) --log-dir $(LOG_DIR) --workers $(ABSA_WORKERS) --retry-errors

absa-preview:
	$(PYTHON) scripts/preview_absa.py --config $(CONFIG)

warmup-absa:
	$(PYTHON) scripts/warmup_absa_checkpoint.py --config $(CONFIG)

absa-benchmark:
	$(PYTHON) scripts/benchmark_absa_latency.py --config $(CONFIG)

absa-quality-compare:
	$(PYTHON) scripts/compare_absa_quality.py --config $(CONFIG)

clean-absa:
	rm -f $(SPLIT_DIR)/absa_cache.sqlite
	rm -f $(SPLIT_DIR)/absa_cache.cache_manifest.json
	rm -f $(SPLIT_DIR)/absa_cache.sqlite-journal
	rm -f $(LOG_DIR)/absa_errors.jsonl

sample-gold:
	$(PYTHON) scripts/sample_absa_gold.py --config $(CONFIG)

absa-quality:
	$(PYTHON) scripts/eval_absa_quality.py --config $(CONFIG)

compare:
	$(PYTHON) scripts/compare_results.py \
		--a $(A) --b $(B) --metric $(METRIC)

neo4j:
	bash scripts/setup_docker_neo4j.sh
	$(PYTHON) scripts/verify_neo4j.py

tgi:
	bash scripts/setup_tgi.sh

tgi-absa:
	bash scripts/setup_tgi_absa.sh

ensure-tgi:
	bash scripts/ensure_tgi.sh

load-kg: neo4j
	$(PYTHON) scripts/load_kg.py --config $(CONFIG)

# TiSASRec + agent Stage-2 rerank (paper method). Paths follow CATEGORY.
# Prerequisites (read-only): make data && make absa (skip absa when CATEGORY=Yelp_AC /
#   absa.enabled=false). Pipeline does not modify those outputs.
# Writes only under data/processed/$(CATEGORY)/tisasrec_align/ and $(RESULTS_DIR)/.
# Yelp_AC Stage-2: preference_source=item_metadata, cross_user_mode=id_only (no ABSA).
check-emorecagent-prereqs:
	$(PYTHON) scripts/check_emorecagent_prereqs.py --config $(CONFIG)

check-emorecagent-eval-prereqs:
	$(PYTHON) scripts/check_emorecagent_prereqs.py --config $(ALIGN_CONFIG) --eval

check-emorecagent-stage1-test-prereqs:
	$(PYTHON) scripts/check_emorecagent_prereqs.py --config $(CONFIG) --stage1-test

train-emorecagent: check-emorecagent-prereqs
	@mkdir -p $(LOG_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(CONFIG)"
	@echo "Progress -> stdout and $$(ls -t $(LOG_DIR)/train_tisasrec_stage1_*.log 2>/dev/null | head -1 || echo $(LOG_DIR)/train_tisasrec_stage1_*.log) (tail -f)"
	PYTHONUNBUFFERED=1 $(PYTHON) scripts/train_tisasrec_stage1.py --config $(CONFIG) --log-dir $(LOG_DIR)

precompute-tu-emorecagent: check-emorecagent-prereqs
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/precompute_tu_cache.py --config $(CONFIG) --split $(or $(SPLIT),train) --log-dir $(LOG_DIR) $(if $(NO_LLM),--no-llm,) $(if $(MAX_ROWS),--max-rows $(MAX_ROWS),)

train-align-emorecagent:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) scripts/train_alignment_stage2.py --config $(CONFIG) --log-dir $(LOG_DIR) $(if $(filter 0,$(USE_HASH_ENCODER)),,--use-hash-encoder)

build-cross-user-lookup-emorecagent: check-emorecagent-prereqs
	$(PYTHON) scripts/build_cross_user_lookup.py --config $(CONFIG)

# Stage 1 TiSASRec test-split eval (stdout + log file). Requires make train-emorecagent only.
test-emorecagent: check-emorecagent-stage1-test-prereqs
	@mkdir -p $(LOG_DIR) $(RESULTS_DIR)
	@echo "CATEGORY=$(CATEGORY) -> $(RESULTS_DIR)/emorecagent_stage1_test.json"
	@echo "Progress -> stdout and $(LOG_DIR)/test_emorecagent_latest.log (tail -f)"
	PYTHONUNBUFFERED=1 $(PYTHON) scripts/eval_tisasrec_stage1_test.py \
		--config $(CONFIG) \
		--log-dir $(LOG_DIR) \
		--log-file $(LOG_DIR)/test_emorecagent_latest.log \
		--out $(RESULTS_DIR)/emorecagent_stage1_test.json \
		$(if $(MAX_PAIRS),--max-pairs $(MAX_PAIRS),)

compare-retrieval:
	$(PYTHON) scripts/compare_retrieval_baselines.py --config $(CONFIG)

compare-batch-parity:
	@mkdir -p $(LOG_DIR)
	@echo "Progress -> stdout and $(LOG_DIR)/compare_batch_parity_latest.log (tail -f)"
	$(PYTHON) scripts/compare_batch_parity.py \
		--max-test-rows $(or $(MAX_TEST_ROWS),100) \
		--skip-speed-gate \
		--progress-interval $(PROGRESS_INTERVAL) \
		--log-file $(LOG_DIR)/compare_batch_parity_latest.log \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(REQUEST_TIMEOUT),--request-timeout $(REQUEST_TIMEOUT),)

experiment-emorecagent: check-emorecagent-eval-prereqs
	@mkdir -p $(LOG_DIR) $(RESULTS_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(ALIGN_CONFIG) SPLIT=$(SPLIT_DIR)"
	@echo "Progress -> stdout and $(LOG_DIR)/experiment_emorecagent_latest.log (tail -f)"
	@echo "Checkpoint -> $(RESULTS_DIR)/emorecagent_align.checkpoint.*.jsonl (auto-resume)"
	@echo "Requires: train-emorecagent, precompute-tu-emorecagent SPLIT=test, build-cross-user-lookup-emorecagent"
	$(PYTHON) scripts/run_experiment.py \
		--config $(ALIGN_CONFIG) \
		--method emorecagent_align \
		--split $(SPLIT_DIR) \
		--out $(RESULTS_DIR)/emorecagent_align.json \
		--eval-pass full \
		--no-sampled-eval \
		--progress-interval $(PROGRESS_INTERVAL) \
		--log-file $(LOG_DIR)/experiment_emorecagent_latest.log \
		$(if $(MAX_TEST_ROWS),--max-test-rows $(MAX_TEST_ROWS),)

experiment-emorecagent-stage1-baseline: check-emorecagent-eval-prereqs
	@mkdir -p $(LOG_DIR) $(RESULTS_DIR)
	@echo "CATEGORY=$(CATEGORY) CONFIG=$(BASELINE_CONFIG) SPLIT=$(SPLIT_DIR)"
	$(PYTHON) scripts/run_experiment.py \
		--config $(BASELINE_CONFIG) \
		--method emorecagent_align \
		--split $(SPLIT_DIR) \
		--out $(RESULTS_DIR)/emorecagent_stage1_baseline.json \
		--eval-pass full \
		--no-sampled-eval \
		--progress-interval $(PROGRESS_INTERVAL) \
		--log-file $(LOG_DIR)/experiment_emorecagent_stage1_baseline.log \
		$(if $(MAX_TEST_ROWS),--max-test-rows $(MAX_TEST_ROWS),)

compare-emorecagent-stage2:
	$(PYTHON) scripts/compare_emorecagent_stage2.py \
		--baseline $(RESULTS_DIR)/emorecagent_stage1_baseline.json \
		--fused $(RESULTS_DIR)/emorecagent_align.json

experiment:
	@mkdir -p $(LOG_DIR) $(RESULTS_DIR)
	$(PYTHON) scripts/run_experiment.py \
		--config $(CONFIG) \
		--method svd \
		--split $(SPLIT_DIR) \
		--out $(RESULTS_DIR)/svd.json \
		--log-dir $(LOG_DIR)

# Protocol B — paper-aligned baseline comparison (user_batch, full catalog, macro user-mean).
# METHOD=svd|itemknn|popularity (default: svd)
experiment-paper:
	@mkdir -p $(LOG_DIR) $(RESULTS_DIR)/paper
	$(PYTHON) scripts/run_experiment.py \
		--config configs/paper_baseline.yaml \
		--method $(or $(METHOD),svd) \
		--split $(SPLIT_DIR) \
		--out $(RESULTS_DIR)/paper/$(or $(METHOD),svd).json \
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
