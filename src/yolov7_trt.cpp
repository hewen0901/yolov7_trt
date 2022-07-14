/* * * * * * * * * * * * * * * * * * * * *
*   File:     yolov7_trt.cpp
*   Brief:    yolov7 TensorRT src code.
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#include "inc/yolov7_trt.h"
#include "inc/common.hpp"

#include <assert.h>
#include <time.h>
#include <chrono>

#define YOLOV7_TAG 1.0
#define ENGINE_ENHANCE_FILE_PATH "../models/yolov7_sim_2070ti_fp16.trt"

static const int kInputChannel = 3;
static const int kInputHeight  = 640;
static const int kInputWidth   = 640;
static const int NUM_CLASSES   = 80;
static const char *kInputBlobName  = "images";
static const char *kOutputBlobName = "output";

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5

namespace siran
{

/// @private function
static cv::Mat StaticResize(const cv::Mat& src);
static cv::cuda::GpuMat StaticResize(const cv::cuda::GpuMat& src);
static int ObjInfo2ObjResult(const cv::cuda::GpuMat &src, const obj_info &obj_result, ObjResult *pobj_result);


/// @brief Tools function, only for debuging.
///
static void DrawObjects(const cv::Mat& bgr, ObjResult *pobj_result, std::string &path, const bool &verbos = true);


/**
 * @brief Yolov7Trt::Yolov7Trt -- Yolov7Trt construct function, load Enhance module TRT model
 * @param iDeviceID    -- Supported device id is 0 or 1.
 */
Yolov7Trt::Yolov7Trt(const int &iDeviceID):
    engine_file_(ENGINE_ENHANCE_FILE_PATH),
    trt_engine_(nullptr),
    trt_context_(nullptr),
    pyolov7_(nullptr),
    nb_bindings_(0),
    input_size_(0),
    output_size_(0),
    device_id_(iDeviceID),
    input_shape_{kInputHeight, kInputWidth}
{
    SetCudaDevice(device_id_);
    LoadEngine();
    assert(trt_engine_ != nullptr);
    trt_context_ = trt_engine_->createExecutionContext();
    assert(trt_context_ != nullptr);
    nb_bindings_ = trt_engine_->getNbBindings();
    assert(trt_engine_->getNbBindings() == nb_bindings_);
    input_size_ = 1*kInputChannel*kInputHeight*kInputWidth;

    buffer_size_.resize(nb_bindings_);
    for(int i=0;i<nb_bindings_;++i)
    {
        auto out_dims = trt_engine_->getBindingDimensions(i);
        auto output_size = 1;
        for(int j=0;j<out_dims.nbDims;j++)
        {
            output_size *= out_dims.d[j];
        }
        buffer_size_[i] = (int)output_size;
    }
    std::cout<<"input_size "<<input_size_<<" output_size  "<<buffer_size_[1]<<std::endl;
    trt_cpu_out_buffers_ = (float*)malloc(buffer_size_[1]*sizeof(float));

    const int inputIndex = trt_engine_->getBindingIndex(kInputBlobName);
    assert(trt_engine_->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = trt_engine_->getBindingIndex(kOutputBlobName);
    assert(trt_engine_->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

    cudaStreamCreate(&cuda_stream_);
    pyolov7_ = new Yolov7(NUM_CLASSES);
}


Yolov7Trt::~Yolov7Trt()
{
    if(pyolov7_)
    {
        delete pyolov7_;
        pyolov7_ = nullptr;
    }
    if(trt_cpu_out_buffers_)
    {
        free(trt_cpu_out_buffers_);
        trt_cpu_out_buffers_ = nullptr;
    }
}

/**
 * @brief Yolov7Trt::LoadEngine -- Load TRT model.
 */
int Yolov7Trt::LoadEngine()
{
    int iret = 0;
    std::fstream existEngine;
    existEngine.open(engine_file_, std::ios::in);
    if (existEngine)
    {
        readTrtFile(engine_file_, trt_engine_);
        assert(trt_engine_ != nullptr);
    }
    else
    {
        std::cout << engine_file_ << " not exists!" << std::endl;
        return -1;
    }
    return iret;
}



/**
 * @brief Yolov7Trt::PreprocessImage -- Read image and transform 3 channels mat to one-dimensional array.
 * @param img                    -- input cv::cuda::GpuMat data
 * @param data                   -- output one-dim array
 */
int Yolov7Trt::PreprocessImage(cv::cuda::GpuMat &img, float* data)
{
    int iret = 0;
    if(img.empty())
    {
        return -1;
    }
    int dst_h = img.rows;
    int dst_w = img.cols;
    int rc = img.channels();
    int img_length = dst_h * dst_w;

    /// @brief One channel image length. HWC 640x640x3 --> CHW 3x640x640.
    std::vector<cv::cuda::GpuMat> split_img = {
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, data + img_length * 2), // R
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, data + img_length * 1), // G
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, data + img_length * 0)  // B
    };
    cv::cuda::split(img, split_img);
    return iret;
}


/**
 * @brief Yolov7Trt::DoInference
 * @param img
 * @param dst_h
 * @param dst_w
 * @param verbos
 * @return
 */
float* Yolov7Trt::DoInference(cv::cuda::GpuMat &img, const int &dst_h, const int &dst_w, const bool &verbos)
{
    /// @brief set GpuMat processing platform.
    SetCudaDevice(device_id_);

    /// @brief Apply cuda memory for tensorrt inculude input and all output.
    for (int i = 0; i < nb_bindings_; ++i)
    {
        if (i == 0)
        {
            cudaMalloc(&trt_out_buffers_[i], buffer_size_[i]*sizeof(float));
        }
        else
        {
            cudaMalloc(&trt_out_buffers_[i], buffer_size_[i]*sizeof(float));
        }
    }
    float* input = (float *) trt_out_buffers_[0];
    float *output = (float *) trt_out_buffers_[1];

    /// @brief Permute image.
    this->PreprocessImage(img, input);

    double time_start = GetCurrentTime();
    double time_process = 0.f;
    if(verbos)
    {
        time_process = GetCurrentTime() - time_start;
        printf("   @@@@@ YOLOV7 DoInference permute time: %.2fms\n", time_process);
    }

    /// @brief Define input param -- BCHW.
    nvinfer1::Dims4 input_dims{1, kInputChannel, dst_h, dst_w};
    trt_context_->setBindingDimensions(0, input_dims);

    time_start = GetCurrentTime();
    /// @brief Do inference processing.
        trt_context_->executeV2(trt_out_buffers_); // Synchronously execute inference a network.
//    trt_context_->enqueue(1, trt_out_buffers_, cuda_stream_, nullptr);// Asynchronously execute inference on a batch.
    if(verbos)
    {
        time_process = GetCurrentTime() - time_start;
        printf("   @@@@@ YOLOV7 DoInference forwart time: %.2fms\n", time_process);
    }

    return output;
}


int Yolov7Trt::Yolov7Infer(const cv::Mat &src, ObjResult *pobj_result, const std::string *path, const bool &verbos)
{
    int iret = 0;
    if(src.empty())
    {
        return -1;
    }
    const int dst_h = kInputHeight;
    const int dst_w = kInputWidth;
    float *gpu_out = nullptr;

    double time_start = GetCurrentTime();
    const double time_start1 = time_start;
    double time_process = 0.f;

    SetCudaDevice(device_id_);
    cv::cuda::GpuMat gpu_src;
    gpu_src.upload(src);
    cv::cuda::GpuMat resized_src;

    /// @brief Resize input image. BGR --> RGB f32 1/255.0, 640x640x3.
    resized_src = StaticResize(gpu_src);
    if(verbos)
    {
        double time_process = GetCurrentTime() - time_start;
        printf("##### YOLOV7 Resize time: %.2fms\n", time_process);
    }

    time_start = GetCurrentTime();
    gpu_out = this->DoInference(resized_src, dst_h, dst_w);
    if(verbos)
    {
        time_process = GetCurrentTime() - time_start;
        printf("@@@@@ YOLOV7 DoInference time: %.2fms\n", time_process);
    }

    memset(trt_cpu_out_buffers_, 0, buffer_size_[1]*sizeof(float));
    cudaMemcpyAsync(trt_cpu_out_buffers_, gpu_out, buffer_size_[1]*sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);

    time_start = GetCurrentTime();
    iret = Yolov7Postprocess(trt_cpu_out_buffers_, gpu_src, pobj_result);
    if(iret != 0)
    {
        return iret;
    }
    if(verbos)
    {
        time_process = GetCurrentTime() - time_start;
        printf("$$$$$ YOLOV7 Postprocess time: %.2fms\n", time_process);
        time_process = GetCurrentTime() - time_start1;
        printf("************************ YOLOV7 Processing time: %.2fms ************************\n", time_process);

        std::string tmp_path = " ";
        if(path != nullptr)
        {
            tmp_path = *path;
        }
        time_start = GetCurrentTime();
        DrawObjects(src, pobj_result, tmp_path, verbos);
        time_process = GetCurrentTime() - time_start;
        printf("%%%%% YOLOV7 DrawResult time: %.2fms\n", time_process);
    }

    /// @brief Free cuda memory.
    for(int i=0;i<nb_bindings_;++i)
    {
        cudaFree(trt_out_buffers_[i]);
        trt_out_buffers_[i] = nullptr;
    }
    return iret;
}


int Yolov7Trt::Yolov7Postprocess(float *trt_out, const cv::cuda::GpuMat &src, ObjResult *pobj_result)
{
    int iret = 0;
    if(nullptr == trt_out)
    {
        return iret;
    }
    obj_info tmp_obj_result;
    iret = pyolov7_->YoloProcess(trt_out, BBOX_CONF_THRESH, &tmp_obj_result);
    if(iret != 0)
    {
        return iret;
    }
    iret = ObjInfo2ObjResult(src, tmp_obj_result, pobj_result);
    if(iret != 0)
    {
        return iret;
    }
    return iret;
}


static cv::Mat StaticResize(const cv::Mat& src)
{
    float r = std::min(kInputWidth / (src.cols*1.0), kInputHeight / (src.rows*1.0));
    int unpad_w = r * src.cols;
    int unpad_h = r * src.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(src, re, re.size());
    cv::Mat out(kInputHeight, kInputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    out.convertTo(out, CV_32FC3, 1.0f / 255.0f);
    return out;
}


static cv::cuda::GpuMat StaticResize(const cv::cuda::GpuMat& src)
{
    float r = std::min(kInputWidth / (src.cols*1.0), kInputHeight / (src.rows*1.0));
    int unpad_w = r * src.cols;
    int unpad_h = r * src.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(src, re, re.size());
    cv::cuda::GpuMat out(kInputHeight, kInputWidth, CV_8UC3);//, cv::Scalar(114, 114, 114)
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    out.convertTo(out, CV_32FC3, 1.0f / 255.0f);
    return out;
}


static int ObjInfo2ObjResult(const cv::cuda::GpuMat &src, const obj_info &obj_result, ObjResult *pobj_result)
{
    int iret = 0;
    if(0 == obj_result.obj_num)
    {
        return -1;
    }
    const int img_h = src.rows;
    const int img_w = src.cols;
    const float ratio = float(img_w) / float(kInputWidth) > float(img_h) / float(kInputHeight)  ?
                float(img_w) / float(kInputWidth) : float(img_h) / float(kInputHeight);
    memset(pobj_result, 0, sizeof(ObjResult));
    pobj_result->obj_num = obj_result.obj_num;
    for(int i=0;i<obj_result.obj_num;++i)
    {
        pobj_result->obj_info[i].obj_box.x = (int)obj_result.obj_result[i].left * ratio;
        pobj_result->obj_info[i].obj_box.y = (int)obj_result.obj_result[i].low  * ratio;
        pobj_result->obj_info[i].obj_box.width = ((int)obj_result.obj_result[i].right - (int)obj_result.obj_result[i].left) * ratio;
        pobj_result->obj_info[i].obj_box.height = ((int)obj_result.obj_result[i].high - (int)obj_result.obj_result[i].low) * ratio;
        pobj_result->obj_info[i].obj_class = obj_result.obj_result[i].id;
        pobj_result->obj_info[i].obj_prob = obj_result.obj_result[i].prob;
    }
    return iret;
}


static void DrawObjects(const cv::Mat& bgr, ObjResult *pobj_result, std::string &path, const bool &verbos)
{
    int iret = 0;
    const std::string save_path = "./result";
    iret = CheckFolderExist(save_path);
    if(iret != 0)
    {
        return;
    }
    cv::Mat image = bgr.clone();
    for (size_t i = 0; i < pobj_result->obj_num; i++)
    {
        const ObjInfo& obj_info = pobj_result->obj_info[i];
        const int obj_x = obj_info.obj_box.x;
        const int obj_y = obj_info.obj_box.y ;
        const int obj_width = obj_info.obj_box.width;
        const int obj_height = obj_info.obj_box.height;
        const int obj_class = obj_info.obj_class;
        const float obj_prob = obj_info.obj_prob;

        cv::Scalar color = cv::Scalar(color_list[obj_class][0], color_list[obj_class][1], color_list[obj_class][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5)
        {
            txt_color = cv::Scalar(0, 0, 0);
        }else
        {
            txt_color = cv::Scalar(255, 255, 255);
        }
        cv::rectangle(image, cv::Rect(obj_x, obj_y, obj_width, obj_height), color * 255, 2);

        char text[256];
        memset(text, 0, 256*sizeof(char));
        sprintf(text, "%s %.1f%%", class_names[obj_class], obj_prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;
        int x = obj_x;
        int y = obj_y + 1;
        y = y>image.rows?image.rows:y;
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
    if(verbos == true)
    {
        char filepath[256];
        memset(filepath, 0, sizeof(filepath));
        sprintf(filepath, "%s/%s.jpg",save_path.c_str(), path.c_str());
        cv::imwrite(filepath, image);
    }
    else
    {
        cv::namedWindow("yolov7_window", cv::WINDOW_NORMAL);
        cv::imshow("yolov7_window", image);
        cv::waitKey(1);
    }
    return;
}

/**
 * @brief argsort -- Sort vector by w/h ratio.
 * @param array   -- input w/h ratio vector
 * @return        -- return sorted ratio vector
 */
template<typename T>
std::vector<int> argsort(const std::vector<T> &array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
    {
        array_index[i] = i;
    }
    std::sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });
    return array_index;
}



}


