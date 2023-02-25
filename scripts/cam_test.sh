#!/usr/bin/env bash
# cd "`dirname "$0"`"
CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml
CKPT=pretrained_models/fast_421_res152_256x192.pth
OUTDIR=${4:-"./examples/res"}
# cd "/Users/ianwu/Desktop/CS281/AlphaPose"

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --outdir ${OUTDIR} \
    --detector "yolo"  --save_img --save_video \
    --webcam 1 \
    --gpus -1

