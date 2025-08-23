classes = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire", ]
model = dict(group_size=128, num_group=1024)
image_size = 224
do_normalization = False
coreset = dict(ratio=0.1, eps=0.9)
max_train_iter = 400  # 400
max_test_iter = 1000
test = dict(n_reweight=3)
cache_dir = '/output/mvtec3d/pointmae'