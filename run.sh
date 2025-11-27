  python SoilNet/train-soilnet-dendrite.py \
    --train_csv soiling_dataset/soiling_labels_4class_train.csv \
    --test_csv  soiling_dataset/soiling_labels_4class_test.csv \
    --out_dir   runs/mobilenetv2_soiling \
    --epochs 60 --batch_size 64