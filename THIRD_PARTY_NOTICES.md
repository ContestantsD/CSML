# Third-party notices

This repository adapts code from the following third-party projects. Their
contributions are acknowledged here; refer to each project for its upstream
source and documentation.

## SubdivNet

The subdivision connectivity operations in `CSG/` (MAPS-style simplification
and hierarchical remeshing) are adapted from
[SubdivNet](https://github.com/lzhengning/SubdivNet), distributed under the
MIT License reproduced below.

```
MIT License

Copyright (c) 2021 Zheng-Ning Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MeshMAE

The patch-based masked-autoencoding backbone of the Cortical Transformer is
derived from [MeshMAE](https://github.com/liang3588/MeshMAE):

```
@inproceedings{meshmae2022,
  title={MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis},
  author={Liang, Yaqian and Zhao, Shanshan and Yu, Baosheng and Zhang, Jing and He, Fazhi},
  booktitle={European Conference on Computer Vision},
  year={2022},
}
```

The upstream MeshMAE repository does not include a license file at the time
of this release; no upstream license terms are therefore reproduced here.
The original implementation is credited and cited as above.
