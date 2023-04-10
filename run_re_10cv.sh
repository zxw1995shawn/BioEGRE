for SPLIT in 1 2 3 4 5 6 7 8 9 10
do
  export SAVE_DIR=/tmp/zhengxw/output
  export DATA="ADE"
  export CLASSIFIER_DROPOUT=0.0
  export DATA_DIR=/tmp/zhengxw/${DATA}/${SPLIT}

  export MAX_LENGTH=128
  export BATCH_SIZE=16
  export NUM_EPOCHS=20
  export SAVE_STEPS=1000
  export SEED=1
  export NUM_CHOSN_NEIGHBORS=32
  export NUM_GPNN_OUTPUT_NODE=4
  export USE_CLS=False
  export NUM_GPNN_LAYERS=0-0

  export ENTITY=${DATA}-${SPLIT}-${MAX_LENGTH}-${BATCH_SIZE}-${NUM_EPOCHS}

  export MODEL=BioELECTRA
  echo "***** " $DATA " train-eval " $SPLIT " Start *****"
  python bioelectra_linking_CLS_with_entities_with_marked_word_with_multiple_gpnn_layers_for_binary_classification.py \
    --task_name gad \
    --config_name ./bioelectra-base-discriminator-pubmed/config.json \
    --data_dir ${DATA_DIR} \
    --num_chosn_neighbors ${NUM_CHOSN_NEIGHBORS}\
    --num_GPNN_output_node ${NUM_GPNN_OUTPUT_NODE}\
    --model_name_or_path ./bioelectra-base-discriminator-pubmed \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --output_dir ${SAVE_DIR}/${ENTITY}-${MODEL} \
    --overwrite_output_dir \
    --classifier_dropout ${CLASSIFIER_DROPOUT} \
    --logging_dir ./logs/${ENTITY}-${MODEL} \
    --logging_steps 20\
    --dataloader_pin_memory True\
    --dataloader_num_worker 8
done
echo "***** " $DATA " train-eval Done *****"