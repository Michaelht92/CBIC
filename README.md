# content_based_image_classification
基于内容的图像分类仿真系统

使用图像底层特征进行特征表示：
1.颜色特征：HSV颜色空间颜色直方图
2.纹理特征：LBP纹理特征、Gabor变换纹理特征

分类器：支持向量机 采用“一对一”的组合方式构建基于支持向量机的图像多分类器

技术细节：
（1）编程语言：Python 2.7.8（64-bit）版本。主要用于图像特征算法提取实现与试验系统搭建。
（2）科学计算环境：Anaconda 2.1.0(64-bit) 科学计算包。Anaconda安装方便，且集成了很多关于Python科学计算的第三方库，如Numpy、Scipy、Matplotlib等模块。主要用在处理图像特征提取部分中的相关计算问题以及实验中相关图表的绘制。
（3）SVM工具：libsvm-3.20工具包。主要用来构建图像分类器，实现图像的分类训练与预测。
（4）编译器：JetBrains PyCharm 4.5.4。PyCharm是由JetBrains公司旗下的专门为Python打造的编译器，用于实验系统的开发与调试。
（5）数据库：MongoDB 2.6.5(64-bit)。MongoDB是一个基于分布式文件存储，具有性能高、容易使用、方便存储数据等优点的数据库。用于图像特征库的存储以及图像特征的存取。

系统说明：
imagefeatures:用于图像底层特征提取。
normal:图像特征不加权的图像分类数据准备。
weight:图像特征加权的图像分类数据准备。
svmtool:图像分类所用工具，包括 easy.py,grid.py进行交叉验证参数优选的工具，nsvm.py为进行分类的脚本。
test:包括一些系统中未用到的图像特征提取方法、实验过程中的仿真图以及其他工具。
warehouse：主要是图像特征库存储以及提取的相关操作。
log.py:为系统运行的log系统，记录中间过程。
meta.py:为系统运行的主流程，可在meta里面调用其他包里面的脚本，做相应的实验。


