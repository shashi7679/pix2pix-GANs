FILE=$sat2map

echo "Note: available models are edges2shoes, sat2map, map2sat, facades_label2photo, and day2night"
echo "Specified [$FILE]"

mkdir -p ./checkpoints/sat2map_pretrained
MODEL_FILE=./checkpoints/sat2map_pretrained/latest_net_G.pth
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/models-pytorch/sat2map.pth

wget -N $URL -O $MODEL_FILE