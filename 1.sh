python val.py \
  --weights checkpoints/ITSDT-15k.pt \
  --data data.yaml \
  --dataset-root datasets/ITSDT-15K_retina \
  --device 0 \
  --imgsz 512  \
  --batch 4 \
  --name IRDST-H_val

python export.py \
  --weights checkpoints/ITSDT-15k.pt \
  --output checkpoints/ITSDT-15k.onnx \
  --imgsz 512 \
  --with-motion-stream

python video_onnx_stream.py \
  --weights checkpoints/ITSDT-15k.onnx \
  --source datasets/ITSDT-15K_all/ITSDT-15K/images/38 \
  --conf 0.25 \
  --name test_ITSDT-15k_stream_38


python motion_map.py \
  --source-root datasets/test/1/ir_周四022603_113902_11044_16195_25.0/images \
  --output-root datasets/test/1/ir_周四022603_113902_11044_16195_25.0/image \
  --recursive
