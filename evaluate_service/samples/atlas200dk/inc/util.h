/**
* @file util.h
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#ifndef INC_UTIL_H_
#define INC_UTIL_H_
#include <iostream>
#include <memory>
class Util
{
public:
    static char* ReadBinFile(std::shared_ptr<std::string> file_name_ptr, uint32_t* file_size, int32_t batchSize, bool& isDMalloc);
    static bool CheckFileExist(const std::string& file_name);
    static void ClassifyDump(const std::string& file_name, std::shared_ptr<std::string> data);
};
#endif //INC_UTIL_H_
