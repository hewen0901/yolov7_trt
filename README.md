### YOLOV7目标检测算法的C++ TensorRT部署代码

#### 1. 算法介绍

- [yolov7目标检测算法链接](https://github.com/WongKinYiu/yolov7)，相关内容请自行跳转yolov7代码仓；

- 因yolov5前后处理和yolov7相差无几，故本仓库同样适用于yolov5算法的TensorRT部署，但需要注意anchor的尺度，根据网络训练时的anchor配置进行相应的修改；

- 以80类coco数据集为例。为了部署更加方便（个人想法），故修改了原始模型，即对yolov7在640x640分辨率下的三个输出头进行拼接处理，输出一个**[1,25200,85]**的tensor，具体操作如下所示；

  ```python
  ## 修改yolov7/models/yolo.py 23行 Detect类的forward函数
      def forward(self, x):
          # x = x.copy()  # for profiling
          z = []  # inference output
          self.training |= self.export
          for i in range(self.nl):
              x[i] = self.m[i](x[i])  # conv
              bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
              x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
              x[i] = x[i].sigmoid()
              z.append(x[i].view(bs, -1, self.no))
              if not self.training:  # inference
                  if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                      self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                  y = x[i].sigmoid()
                  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                  z.append(y.view(bs, -1, self.no))
                  
          return (torch.cat(z, 1))
  ```

#### 2. 使用说明

- 当前代码暂不支持多batch，若要支持多batch也不难，稍加修改即可；

- 使用步骤如下所示：

  ```sh
  mkdir build
  cd build
  cmake ..
  make -j4
  ./Test_app ../data/
  ```

  *备注：需针对性的修改tensorrt模型的路径和测试文件夹的路径！*

#### 3. 参考结果

- 截止到目前2022年7月13号，目前测试的是官方提供的yolov7.pth模型，在PC上使用**TensorRT-8.4.1.5**进行序列化后，fp16的trt模型大小为75.6Mb，在2070ti的显卡下，性能表现如下所示：

  | 平台       | fp16                                                         | 备注                                           |
  | ---------- | ------------------------------------------------------------ | ---------------------------------------------- |
  | PC         | 1. 预处理0.5ms<br />2. 前向计算5.0ms<br />3. 后处理0.15ms<br />4. 画图4.5ms | 画图的时间<br />长短和检测<br />目标个数相关； |
  | Jetson Tx2 | 1. 预处理2.6ms<br />2. 前向计算118.0ms<br />3. 后处理0.7ms<br />4. 画图8.5ms | 同上                                           |
  | 合计       | PC：5.65ms，TX2：121.3ms                                     | 不统计画图时间                                 |

  *备注：在英伟达TX2上速度慢的离谱，暂时不知道是哪里出了问题，有待进一步验证；*

  ​																																				20220713 -----by nero

  

