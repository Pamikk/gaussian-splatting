from PIL import Image
import numpy as np
import os
exp_path = '/home/pami/exps_on/gs_depth/bicycle/train/ours_30000'
img_id = 4
img_fn = f'{img_id:05}.png'
gt = np.asarray(Image.open(os.path.join(exp_path,'gt',img_fn))).astype(float)/255.0
pred = np.asarray(Image.open(os.path.join(exp_path,'renders',img_fn))).astype(float)/255.0
diff = np.abs((gt-pred)).sum(axis=-1)*255*10
print(diff.max(),diff.mean(),diff.shape)
diff = Image.fromarray((diff).astype(np.uint8))
diff.save(os.path.join(exp_path,f'diff_{img_id:05}.png'))
