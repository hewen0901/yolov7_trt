/* * * * * * * * * * * * * * * * * * * * *
*   File:     utils.cpp
*   Brief:    common function.
*   Author:   hewen
*   Company:  SIRAN
*   E-mail:   senlin0901@gmail.com
*   Time:     2022/07/13
* * * * * * * * * * * * * * * * * * * * * */
#include "inc/utils.h"

#include <iostream>
#include <string>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "dirent.h"
#include <sys/time.h>
#include <unistd.h>

using namespace std;


/**
 * @brief GetFilePath       -- Get all file paths in a folder.
 * @param path          -- input folder path
 * @param filename_vec  -- output file paths vector
 * @return              -- 0--success, -1--inputpath null
 */
int GetFilePath(const char *path,std::vector<std::string> &filename_vec)
{
    DIR *dir = opendir(path);
    if(dir == NULL)
    {
        return -1;
    }
    struct dirent *p = NULL;
    struct stat buf;
    int res = -1;
    char sub_name[200] = {0};
    while(1)
    {
        p = readdir(dir);
        if(p == NULL)
        {
            break;
            if(errno != 0)
            {
                return -1;
            }
        }
        sprintf(sub_name,"%s/%s",path,p->d_name);
        res = stat(sub_name,&buf);
        if(res == -1)
        {
            return -1;
        }
        if(S_ISDIR(buf.st_mode))
        {
            if(strcmp(p->d_name,".") ==0 || strcmp(p->d_name,"..") == 0)
            {
                continue;
            }
            GetFilePath(sub_name,filename_vec);
        }
        if(S_ISREG(buf.st_mode))
        {
            filename_vec.push_back(sub_name);
        }
    }
    return 0;
}

/**
 * @brief GetCurrentTime -- Get current time.
 * @return               -- double type milliseconds
 */
double GetCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000.0;
}

/**
 * @brief CheckFolderExist -- Check folder exist or not, if not, create it.
 * @param path           -- input folder path
 * @return               -- 0--exist, -1--create failed
 */
int CheckFolderExist(const std::string &path)
{
    int iret = 0;
    string dir = path;
    if (access(dir.c_str(), 0) == -1)
    {
        cout<<dir<<" is not existing"<<endl;
        cout<<"now make it"<<endl;
#ifdef LINUX_OS
        int flag = mkdir(dir.c_str(), 0777);
#else
        int flag = mkdir(dir.c_str());
#endif
        if (flag == 0)
        {
            cout<<"make successfully"<<endl;
            return iret;
        }
        else
        {
            cout<<"make errorly"<<endl;
            return -1;
        }
    }
    return iret;
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
  {
    v.push_back(s.substr(pos1));
  }
  return;
}

