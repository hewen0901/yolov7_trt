/* * * * * * * * * * * * * * * * * * * * *
*   File:     yolov7.hpp
*   Brief:    yolov7 postprocess code.
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#pragma once
#include <vector>
#include <string>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <stdio.h>
#include <string.h>

#define MAX_OBJ_NUM 20
#define YOLOv7_WIDTH 640
#define YOLOv7_HEIGHT 640

inline int align_8(int num)
{
    return (num + 7) >> 3 << 3;
}

//int align_32(int num)
//{
//    return ceil((double)num/32.)*32;
//}

inline float sigmoid(const float& x)
{
    return 1/(1 + exp(-x));
}

typedef struct
{
    float left;  //xmin
    float right; //xmax
    float low;   //ymin
    float high;  //ymax
    int id;
    float prob;
} object_t;

typedef struct  OBJ_INFO_
{
    object_t obj_result[MAX_OBJ_NUM];
    int obj_num;
}obj_info, *pObj_info;


class Yolov7
{
private:
    int class_num;
    int input_w;
    int input_h;
    int output_align_len;
    int outputs_pixel[3][2];
    int stride[3];
    int MAX_OBJECTS;
    float anchors[3][3][2];

public:
    object_t *objects;
    int *valid;
    int object_num;
    Yolov7(const int &classNum)
        : input_w(YOLOv7_WIDTH),
          input_h(YOLOv7_HEIGHT),
          class_num(classNum),
          stride({8, 16, 32}),
          outputs_pixel({
    {80, 80},
    {40, 40},
    {20, 20}}),
          anchors({
    {{12, 16}, {19, 36}, {40, 28}}, //8
    {{36, 75}, {76, 55}, {72, 146}}, //16
    {{142, 110}, {192, 243}, {459, 401}}}//32
                  ),

          MAX_OBJECTS(512),
          object_num(0),
          output_align_len(((class_num + 5)*3))
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                printf("%d ", outputs_pixel[i][j]);
            }
            printf("\n");
        }
        memcpy(this->anchors, anchors, sizeof(this->anchors));
        objects = new object_t[MAX_OBJECTS];
        valid = new int[MAX_OBJECTS];
    }

    ~Yolov7()
    {
        delete[] objects;
        delete[] valid;
    }
    void NMS(object_t *objects, const int &object_num, int *valid_)
    {
        std::vector<int> idx(object_num);
        std::iota(idx.begin(), idx.end(), 0);
        object_t *objs = objects;
        std::sort(idx.begin(), idx.end(),[&objs](int i1, int i2) { return objs[i1].prob > objs[i2].prob; });
        for (int i_sort = 0; i_sort < object_num; ++i_sort)
        {
            int i = idx[i_sort];
            if (!valid_[i])
            {
                continue;
            }
            const object_t &obj1 = objects[i];
            float a1 = (obj1.right - obj1.left) * (obj1.high - obj1.low);

            for (int j_sort = i_sort + 1; j_sort < object_num; ++j_sort)
            {
                int j = idx[j_sort];
                if (!valid_[j] || objects[j].id != obj1.id)
                {
                    continue;
                }
                const object_t &obj2 = objects[j];
                float a2 = (obj2.right - obj2.left) * (obj2.high - obj2.low);
                float left = std::max(obj1.left, obj2.left);
                float low = std::max(obj1.low, obj2.low);
                float right = std::min(obj1.right, obj2.right);
                float high = std::min(obj1.high, obj2.high);
                float sa = std::max(0.f, right - left) * std::max(0.f, high - low);
                float iou = sa * 1. / (a1 + a2 - sa);
                if (iou > 0.4)
                {
                    valid_[j] = 0;
                }
            }
        }
        return;
    }

    int YoloProcess(float *fea_out, const float &g_thresh, obj_info *pObjInfo)
    {
        int iret = 0;
        if(nullptr == fea_out || NULL == pObjInfo)
        {
            printf("input error\n");
            return -1;
        }
        memset(pObjInfo, 0, sizeof(obj_info));
        int feat = 0;
        int object_num = 0;
        const int ac_num = 3;

        for (int stride_i = 0; stride_i < 3; stride_i++)
        {
            int feat_w = outputs_pixel[stride_i][1];
            int feat_h = outputs_pixel[stride_i][0];
            for (int ac_i = 0; ac_i < ac_num; ac_i++)
            {
                for (int h_i = 0; h_i < feat_h; h_i++)
                {
                    for (int w_i = 0; w_i < feat_w; w_i++)
                    {
                        float obj_conf = fea_out[4 + feat];
                        if(obj_conf > g_thresh)
                        {
                            int max_class_index = 0;
                            float max_class_conf = 0.f;
                            for (int class_i = 0; class_i < class_num; ++class_i)
                            {
                                if (fea_out[5 + class_i + feat] > max_class_conf)//feat[5 + class_i] > max_class_conf
                                {
                                    max_class_conf = fea_out[5 + class_i + feat];
                                    max_class_index = class_i;
                                }
                            }
                            float conf = max_class_conf * obj_conf;//
                            if (conf > g_thresh)
                            {
                                float cx = (fea_out[0 + feat] * 2.f - 0.5f + (float)w_i) / feat_w;
                                float cy = (fea_out[1 + feat] * 2.f - 0.5f + (float)h_i) / feat_h;
                                float w  = pow(fea_out[2 + feat] * 2.f, 2) * anchors[stride_i][ac_i][0] / input_w;
                                float h  = pow(fea_out[3 + feat] * 2.f, 2) * anchors[stride_i][ac_i][1] / input_h;
                                object_t &obj = objects[object_num];
                                valid[object_num] = 1;
                                obj.left  = std::max(0.f, std::min(1.f, cx - w / 2));
                                obj.right = std::max(0.f, std::min(1.f, cx + w / 2));
                                obj.low   = std::max(0.f, std::min(1.f, cy - h / 2));
                                obj.high  = std::max(0.f, std::min(1.f, cy + h / 2));
                                obj.prob  = conf;
                                obj.id = max_class_index;
                                ++object_num;
                            }
                        }
                        feat += class_num + 5;
                    }
                }
            }
        }
        if(0 == object_num)
        {
            return -999;
        }
        NMS(objects, object_num, valid);
        std::vector<int> obj_vec;
        obj_vec.resize(object_num);
        int index = 0;
        for(std::vector<int>::iterator iter = obj_vec.begin(); iter != obj_vec.end();)
        {
            if (valid[index])
            {
                *iter = index;
                iter++;
            }
            else
            {
                obj_vec.erase(iter);
            }
            index++;
        }
        for (int idx = 0; idx < obj_vec.size(); idx++)
        {
            pObjInfo->obj_result[idx].left  = objects[obj_vec[idx]].left * input_w;
            pObjInfo->obj_result[idx].low   = objects[obj_vec[idx]].low * input_h;
            pObjInfo->obj_result[idx].right = objects[obj_vec[idx]].right * input_w;
            pObjInfo->obj_result[idx].high  = objects[obj_vec[idx]].high * input_h;
            pObjInfo->obj_result[idx].id    = objects[obj_vec[idx]].id;
            pObjInfo->obj_result[idx].prob  = objects[obj_vec[idx]].prob;
            pObjInfo->obj_num++;
        }
        return iret;
    }

};

