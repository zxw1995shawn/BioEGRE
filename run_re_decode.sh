export SAVE_DIR=./output/chemprot-128-32-50-0.0-BioELECTRA
# export SPLIT="2"
# export DATA_DIR=../datasets/RE/${DATA}/${SPLIT}
# export ENTITY=${DATA}-${SPLIT}
export DATA="chemprot"
export DATA_DIR=../datasets/RE/${DATA}/

export MAX_LENGTH=128
export BATCH_SIZE=16
export NUM_EPOCHS=50
export SAVE_STEPS=1000
export SEED=1
export CLASSIFIER_DROPOUT=0.0

export ENTITY=${DATA}-${MAX_LENGTH}-${BATCH_SIZE}-${NUM_EPOCHS}-${CLASSIFIER_DROPOUT}
export MODEL=BioELECTRA-GPNN-alldata
# export MODEL=BioBERT

export output_dir=${SAVE_DIR}/${ENTITY}-${MODEL}

python ./scripts/re_eval.py --output_path=${SAVE_DIR}/test_results.txt --answer_path=${DATA_DIR}/test.tsv