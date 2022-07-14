/* * * * * * * * * * * * * * * * * * * * *
*   File:     yolov7_trt_c.cpp
*   Brief:    yolov7_trt c code api.
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#include "export/export.h"
#include "inc/yolov7_trt.h"
#include <vector>
#include <opencv2/opencv.hpp>


#define CAMERA_WIDTH  2560
#define CAMERA_HEIGHT 960

///@private function
std::vector<cv::Mat> SplitImage(const cv::Mat &image, int num,int type);


int Yolov7Infer(const void *src, ObjResult *pobj_result, const bool &verbos)
{
    int iret = 0;
    if(nullptr == src)
    {
        return -1;
    }
    cv::Mat *img = (cv::Mat*)src;
    static siran::Yolov7Trt Yolov7_trt= siran::Yolov7Trt(0);
    iret = Yolov7_trt.Yolov7Infer(*img, pobj_result, nullptr, verbos);
    if(iret != 0)
    {
        return iret;
    }
    return iret;
}

int Yolov7Infer2(const std::string &path, ObjResult *pobj_result, const bool &verbos)
{
    int iret = 0;
    if(" " == path)
    {
        return -1;
    }
    cv::Mat src = cv::imread(path, -1);
    if(src.empty())
    {
        return -1;
    }
    std::string str_filepath = path;
    std::string filename = str_filepath.substr(str_filepath.find_last_of('/')+1);
    std::string fileinname = filename.substr(0, filename.find_last_of("."));

    static siran::Yolov7Trt Yolov7_trt= siran::Yolov7Trt(0);
    iret = Yolov7_trt.Yolov7Infer(src, pobj_result, &fileinname, verbos);
    if(iret != 0)
    {
        return iret;
    }
    return iret;
}

int Yolov7Infer3(const int &camera_index, ObjResult *pobj_result, const bool &verbos)
{
    int iret = 0;
    if(-1 == camera_index)
    {
        return -1;
    }
    static siran::Yolov7Trt Yolov7_trt= siran::Yolov7Trt(0);

    cv::VideoCapture cap(camera_index);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);  // set dual-camera width
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);  // set dual-camera height

    if(!cap.isOpened())
    {
        return -1;
    }
    cv::Mat frame;
    while(true)
    {
        if(!cap.grab())
        {
            break;
        }
        cap >> frame;
        std::vector<cv::Mat> two_img = SplitImage(frame, 2, 1);
        iret = Yolov7_trt.Yolov7Infer(two_img[0], pobj_result, nullptr, verbos);
        if(iret != 0)
        {
            printf("Yolov5 infer null\n");
            continue;
        }
    }
    return iret;
}


std::vector<cv::Mat> SplitImage(const cv::Mat &image, int num,int type)
{
    int rows = image.rows;
    int cols = image.cols;
    std::vector<cv::Mat> v;
    if (type == 0)
    {
        for (size_t i = 0; i < num; i++)
        {
            int star = rows / num*i;
            int end = rows / num*(i + 1);
            if (i == num - 1)
            {
                end = rows;
            }
            v.push_back(image.rowRange(star, end));
        }
    }
    else if (type == 1)
    {
        for (size_t i = 0; i < num; i++)
        {
            int star = cols / num*i;
            int end = cols / num*(i + 1);
            if (i == num - 1)
            {
                end = cols;
            }
            v.push_back(image.colRange(star, end).clone());
        }
    }
    return  v;
}
