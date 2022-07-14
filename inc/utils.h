#ifndef YOLO_TRT_UTILS_H_
#define YOLO_TRT_UTILS_H_


#include <string>
#include <vector>
#include "export/export.h"

/**
 * @brief GetFilePath       -- Get all file paths in a folder.
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
 * @param path           -- input folder path
 * @return               -- 0--exist, -1--create failed
 */
int CheckFolderExist(const std::string &path);

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c);

#endif
