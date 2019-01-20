# Poisson Image Editing

## Introduction

*PoissonImageEditing* implements a series of image processing algorithms based on Poisson editing theory, basically involves image seamless clone with two images as well as an arbitrary mask. Shared and static libraries are both provided along with an example program, which could be played around as an image blender.

In *PoissonImageEditing*, images are initially loaded and processed with `cv::Mat`, and subsequently converted into matrix representation in `Eigen3`, which takes over all linear algebra computing and equation solving.

The following illustrations may give you an intuitive idea about Poisson image editing:

| Naive | Poisson |
|:--------------:|:--------------:|
| ![Naive](/showcases/case0/naive.jpg?raw=true) | ![Poisson](/showcases/case0/result.png?raw=true) |

## Theory

Methods based on the manipulation of image gradients are a powerful tool for processing or combining images. For example operations such as seamless cloning, local illumination changes, texture flatten- ing or seamless tiling can be performed in a very simple and efficient way by combining/modifying the image gradients.

The idea proposed by P'erez et al. in 2003 mainly tells about an image processing techniques working on a gradient domain, taking advantage in simplicity and efficiency. Poisson editing theory plays a significant role in image combination algorithm, which is presented as follows:

<!-- \[
\min _{f} \iint_{\Omega} {\lvert \nabla f - \textbf{v} \rvert}^2 \ s.t.\ f \vert_{\partial \Omega} = f* \vert_{\partial \Omega}
\] -->

![](https://latex.codecogs.com/svg.latex?%5Cdpi%7B100%7D%20%5Cmin%20_%7Bf%7D%20%5Ciint_%7B%5COmega%7D%20%7B%5Clvert%20%5Cnabla%20f%20-%20%5Ctextbf%7Bv%7D%20%5Crvert%7D%5E2%20%5C%20s.t.%5C%20f%20%5Cvert_%7B%5Cpartial%20%5COmega%7D%20%3D%20f*%20%5Cvert_%7B%5Cpartial%20%5COmega%7D)

where Ω is the target region, ∂Ω denotes the boundaries between the source and target regions. *f\** is the already known function that exists in the whole domain (i.e. the background), while *f* is defined as the unknown function on region Ω. The solution of the equation must satisfy Euler-Largrange equation:

<!-- \[
\triangle f(x) = div(\textbf{v}(x)), x \in \Omega  \hspace{0.5em} with \ f \vert_{\partial \Omega} = f* \vert_{\partial \Omega}
\] -->

![](https://latex.codecogs.com/svg.latex?%5Ctriangle%20f%28x%29%20%3D%20div%28%5Ctextbf%7Bv%7D%28x%29%29%2C%20x%20%5Cin%20%5COmega%20%5Chspace%7B0.5em%7D%20with%20%5C%20f%20%5Cvert_%7B%5Cpartial%20%5COmega%7D%20%3D%20f*%20%5Cvert_%7B%5Cpartial%20%5COmega%7D)

where div(**v**) = ∂**v**/∂x + ∂**v**/∂y, of which we usually take the discrete differential form on the gradient field, and ▵ is Laplacian operator in discrete case:

<!-- \[
\mathcal{L}_z = \partial^2Z / \partial x^2 + \partial^2Z / \partial y^2 = -4Z_{x,y} + Z_{x+1,y} + Z_{x-1,y} + Z_{x,y+1} + Z_{x,y-1}
\] -->

![](https://latex.codecogs.com/svg.latex?%5Cdpi%7B100%7D%20%5Cmathcal%7BL%7D_z%20%3D%20%5Cpartial%5E2Z%20/%20%5Cpartial%20x%5E2%20&plus;%20%5Cpartial%5E2Z%20/%20%5Cpartial%20y%5E2%20%3D%20-4Z_%7Bx%2Cy%7D%20&plus;%20Z_%7Bx&plus;1%2Cy%7D%20&plus;%20Z_%7Bx-1%2Cy%7D%20&plus;%20Z_%7Bx%2Cy&plus;1%7D%20&plus;%20Z_%7Bx%2Cy-1%7D)

Apply Laplacian operator and differential operator over the unknown domain Ω and solve the linear equation with matrices to obtain the numerical solutions. More details are presented in the references.

## Installation

Before configure and compile *PoissonImageEditing*, you may need to setup the dependencies `OpenCV` and `Eigen3` first. Refer to [Installation for Windows](https://docs.opencv.org/3.4/d3/d52/tutorial_windows_install.html) or [Installation for Linux](https://docs.opencv.org/3.3.0/d7/d9f/tutorial_linux_install.html) for installation of `OpenCV`. As for `Eigen3`, it is much easier since it requires no build or compilation. You can simply download [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page), and unzip it into some directory which can be found by `CMake`.

You may note that, here are some convenient package manager which could help you install them directly. After configure these dependencies, you can clone this repository by the following command:

```
git clone https://github.com/peihaowang/PoissonImageEditing.git
```

Afterward, run `CMake` to configure the solution and start to build the generated project. In unix-like systems, you can use the following commands to make it:

```
mkdir build && cd build
cmake ..
make -j8
make install
```

Here are some troubleshoots you may need to pay attention:

1. During configuring, dependency missing error may occur for `CMake` failes to find the installed library. The quickest way is to create `3rdparty/lib` directory under the project folder and then copy the corresponding library into it.

2. Upgrading `CMake` or compilers could be a good choice if there occurs wired issues while you are configuring or compiling the project.

3. While running `make install`, `CMake` installs built libraries and executables into the system directories by default. To custom the installation path, please specify the argument `-DCMAKE_INSTALL_PREFIX`. See [CMAKE_INSTALL_PREFIX](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) for details.

## Usage

### Library

*PoissonImageEditing* is deployed under `LGPL` license, and can be used as both shared and static libraries. After building and installing *PoissonImageEditing*, you can include the header `PoissonImage.h` and link binary files in your own project. The basic usages of provided APIs are presented as follows:

```
PoissonImage::PoissonImage(GradientScheme gradientSchm = GradientScheme::Maximum, DiffOp gradientOp = DiffOp::Backward, DiffOp divOp = DiffOp::Forward);
```

Instantiate and initialize the `PoissonImage` object. The three parameters vary the details algorithms slightly, where `Gradient Scheme` makes the greatest difference:

| Naive | Replace | Average | Maximum |
|:--------------:|:--------------:|:----------------:|:----------------:|
| ![Naive](/showcases/case1/naive.jpg?raw=true) | ![Replace](/showcases/case1/replace.jpg?raw=true) | ![Average](/showcases/case1/average.jpg?raw=true) | ![Maximum](/showcases/case1/maximum.jpg?raw=true) |

```
void PoissonImage::seamlessClone(cv::InputArray src, cv::InputArray dst, cv::InputArray mask, const cv::Point& offset, cv::OutputArray output);
```

1. It requires input source and destination images as inputs and a target image on which output the final result.

2. Mask is provided to tell the algorithm rough boundaries of the objects in source image(Hence, their sizes should be the same).

3. The source image is initially put on the center of the destination image, but offset means that it is possible to move the source objects on the destination canvas by giving a translation.

## Executable

The sample code can be compiled into a command-line utility, which has basic features to combine two images together. To run the executable properly, see the following usages:

```
Usage: PoissonImageEditor <source-path> <destination-path> <output-path>
        [ <mask-path> [ <x-offset> <y-offset> [ <-r|-a|-m> [ <-b|-f|-c> <-b|-f|-c> ] ] ] ]
```

## References

1. Pérez, Patrick, Michel Gangnet, and Andrew Blake. "Poisson image editing." ACM Transactions on Graphics (TOG). Vol. 22. No. 3. ACM, 2003.
2. ["Poisson Image Editing", J. Matías Di Martino, Gabriele Facciolo, Enric Meinhardt-Llopis](http://www.ipol.im/pub/art/2016/163/)
3. ["Poisson Blending - Eric Yuan's Blog"](http://eric-yuan.me/poisson-blending/)
