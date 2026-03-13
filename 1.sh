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
  --source-root datasets/DAUB-R_retina/images \
  --output-root datasets/DAUB-R_retina/image_onnx \
  --recursive \
  --mode onnx
