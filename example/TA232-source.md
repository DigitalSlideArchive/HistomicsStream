# Data source
The file `example/TA232.svs` comes from the zip file available as
```
https://stanfordmedicine.box.com/s/ub8e0wlhsdenyhdsuuzp6zhj0i82xrb1
```
from the web page
```
https://github.com/stanfordmlgroup/DLBCL-Morph
```
It is in that zip file as
```
DLBCL-Morph/TMA/MYC/TA232.svs
```

The corresponding mask `example/TA232-mask.png` is randomly generated in Python with
```python
import numpy as np
from PIL import Image
arr = np.random.randint(0, 2, (mask_height, mask_width), dtype=np.int8)
im = Image.fromarray(arr)
im.save("TA232-mask.png")
```
