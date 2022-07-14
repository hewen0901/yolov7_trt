/* * * * * * * * * * * * * * * * * * * * *
*   File:     yolov7_trt.h
*   Brief:    xxx
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#ifndef YOLOV7TRT_INC_H_
#define YOLOV7TRT_INC_H_

#include <export/export.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "inc/yolov7.hpp"

namespace siran
{

class Yolov7Trt
{
public:
    /// @brief Yolov7Trt construct function, load Enhance module TRT model, support
    ///        device id is 0 or 1, default parameter is 0, means using the first GPU.
    explicit Yolov7Trt(const int &iDeviceID = 0);
    ~Yolov7Trt();
    /// @brief Yolov7Trt interface function, input img, logger pointer, output enhanced-gpumat.
    int Yolov7Infer(const cv::Mat &src, ObjResult *pobj_result, const std::string *path = nullptr, const bool &verbos = false);
private:

    /// @brief Load Enhance module TRT model.
    int LoadEngine();

    /// @brief Limit input image size to even number.
    void LimitInputSize(const cv::cuda::GpuMat &img, int &dst_h, int &dst_w);

    /// @brief Transform 3 channels mat to one-dimensional array.
    int PreprocessImage(cv::cuda::GpuMat &img, float *data);

    void DetResizeImg(const cv::cuda::GpuMat &img,int max_size_len, float &ratio_h, float &ratio_w, cv::cuda::GpuMat &resize_img);

    /// @brief Enhance model inference module, input gpumat, output one-dimensional array pointer.
    float* DoInference(cv::cuda::GpuMat &img, const int &dst_h, const int &dst_w, const bool &verbos = false);

    int Yolov7Postprocess(float *trt_out, const cv::cuda::GpuMat &src, ObjResult *pobj_result);

private:
    /// @brief gpu device id
    int device_id_;

    /// @brief enhance TRT model path
    std::string engine_file_;
    std::vector<int> input_shape_;

    int64_t input_size_;
    int64_t output_size_;

    /// @brief enhance model TRT init handle
    nvinfer1::ICudaEngine* trt_engine_;
    nvinfer1::IExecutionContext* trt_context_;

    /// @brief Net input&&output numbers, eg. yolov3~v5 has three output
    ///        feature maps with different scales, so nb_bindings_ is 1+3 = 4.
    int nb_bindings_;
    void* trt_out_buffers_[2];
    float* trt_cpu_out_buffers_;
    cudaStream_t cuda_stream_;
    std::vector<int64_t> buffer_size_;
    std::vector<float> output_vec_;

    /// @brief Yolov7 postprocess code.
    Yolov7 *pyolov7_;

};

}

#endif
