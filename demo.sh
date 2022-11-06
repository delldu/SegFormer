# python demo/image_demo.py \
# 	$1 \
# 	local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py \
# 	models/segformer.b0.512x512.ade.160k.pth \
# 	--device cuda:0 \
# 	--palette ade20k


# python demo/image_demo.py \
# 	$1 \
# 	local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py \
# 	models/segformer.b1.512x512.ade.160k.pth \
# 	--device cuda:0 \
# 	--palette ade20k


python demo/image_demo.py \
	$1 \
	local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py \
	models/segformer.b2.512x512.ade.160k.pth \
	--device cuda:0 \
	--palette ade20k
