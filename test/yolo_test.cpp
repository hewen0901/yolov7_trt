/* * * * * * * * * * * * * * * * * * * * *
*   File:     yolo_test.cpp
*   Brief:    yolov7_trt test code. use: ./Test_app ../data
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#include "export/export.h"
#include <iostream>
#include <string.h>

int main(int arv, char** arg)
{
    if(arv<2)
    {
        printf("use: ./Test_app filepath\n");
        return -1;
    }
    int iret = 0;
    char filepath[150];
    memset(filepath, 0, sizeof(filepath));
    sprintf(filepath, arg[1]);
    std::vector<std::string> filename_vec;
    iret = GetFilePath(filepath, filename_vec);
    if(iret != 0)
    {
        return iret;
    }

    const bool verbos = true;
    for(int i=0;i<filename_vec.size();++i)
    {
        const std::string temppath = filename_vec[i];
        std::string str_filepath = temppath;
        printf("input img path:%s\n", str_filepath.c_str());

        ObjResult obj_result;
        double time_start = GetCurrentTime();
        iret = Yolov7Infer2(temppath.c_str(), &obj_result, verbos);
        if(iret != 0)
        {
            printf("YOLOV7 error code is %d\n", iret);
            continue;
        }
        double time_process = GetCurrentTime() - time_start;
        printf("YOLOV7 processing time: %f\n", time_process);
    }
    printf("finished!\n");
}


