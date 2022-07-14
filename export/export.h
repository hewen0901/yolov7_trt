/* * * * * * * * * * * * * * * * * * * * *
*   File:     export.h
*   Brief:    xxx
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#ifndef YOLOV7TRT_EXPORT_H_
#define YOLOV7TRT_EXPORT_H_

#include <string>
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#define YOLOV7_MAX_OBJ_NUM 100


typedef struct ObjBox_
{
    int x;
    int y;
    int width;
    int height;
}ObjBox;

typedef struct ObjInfo_
{
    ObjBox  obj_box;
    float   obj_prob;
    int     obj_class;
}ObjInfo;

typedef struct ObjResult_
{
    ObjInfo   obj_info[YOLOV7_MAX_OBJ_NUM];
    int  obj_num;
}ObjResult;


int Yolov7Infer(const void* src, ObjResult *pobj_result, const bool &verbos = false);

int Yolov7Infer2(const std::__cxx11::string &path, ObjResult *pobj_result, const bool &verbos = false);

int Yolov7Infer3(const int &camera_index, ObjResult *pobj_result, const bool &verbos = false);

/**
 * @brief GetFilePath   -- Get all file paths in a folder.
 * @param path          -- input folder path
 * @param filename_vec  -- output file paths vector
 * @return              -- 0--success, -1--inputpath null
 */
int GetFilePath(const char *path, std::vector<std::string> &filename_vec);

/**
 * @brief GetCurrentTime -- Get current time.
 * @return               -- double type milliseconds
 */
double GetCurrentTime();

/**
 * @brief CheckFolderExist -- Check folder exist or not, if not, create it.
 * @param path             -- input folder path
 * @return                 -- 0--exist, -1--create failed
 */
int CheckFolderExist(const std::string &path);

/**
 * @brief SplitString
 * @param s
 * @param v
 * @param c
 */
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
