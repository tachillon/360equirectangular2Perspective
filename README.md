
# 360equirectangular2Perspective

Transform a 360° equirectangular image to a perspective one.


## Dependency

Install opencv and numpy

```bash
  pip3 install opencv-python
  pip3 install numpy
```
    
## Usage/Examples

```python3
python3 equirec2perspect.py --img <path/to/input/image> --fov <field of view angle in degree> --theta <horizontal angle, right direction is positive> --phi <vertical angle, up direction is positive>
```

