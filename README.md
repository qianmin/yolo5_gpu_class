# yolo5_gpu_class
yolov5 C++ GPU版本，速度70ms/frame
其他资源文件看yolo5_class

# net使用cuda加速
```
net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
```
# 常规函数使用UMat加速
```
Mat uFunction(frame)
{
  UMat uimg;
  Mat result;
  frame.copyto(uimg)
  ####process uimg like img
  
  canny(uimg)
  
  ####process uimg like img
  uimg.copyto(result)
  return result
}
mat->umat 1.5ms
umat版本函数有很多会被加速5,6倍
umat-mat 1ms
```
