/**
* @file util.cpp
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include "inc/util.h"
#include "inc/tensor.h"
#include "inc/common.h"
#include "hiaiengine/api.h"
#include "hiaiengine/ai_memory.h"
/** 
* @ingroup Util
* @brief ReadBinFile Read the file and return the buffer.
* @param [in]:file_name
* @param [in]: file_size
* @param [out]: std::string
*/
char* Util::ReadBinFile(std::shared_ptr<std::string> file_name, 
    uint32_t* file_size, int32_t batchSize, bool& isDMalloc)
{
    std::filebuf *pbuf;
    std::ifstream filestr;
    size_t size;
    filestr.open(file_name->c_str(), std::ios::binary);
    if (!filestr)
    {
        return NULL;
    }

    pbuf = filestr.rdbuf();
    size = pbuf->pubseekoff(0, std::ios::end, std::ios::in)*batchSize;
    pbuf->pubseekpos(0, std::ios::in);
    char * buffer = nullptr;
    isDMalloc = true;
    HIAI_StatusT getRet = hiai::HIAIMemory::HIAI_DVPP_DMalloc(size, (void*&)buffer);
    if ((getRet != HIAI_OK) || (buffer == nullptr)) {
        buffer = new(std::nothrow) char[size];
        if(buffer != nullptr) {
           isDMalloc = false;
        }
    }

    pbuf->sgetn(buffer, size);
    *file_size = size;
    filestr.close();
    return buffer;
}

/**
* @ingroup Util
* @brief CheckFileExist
* @param [in]:file_name
* @param [out]: std::string
*/
bool Util::CheckFileExist(const std::string& file_name)
{
    std::ifstream f(file_name.c_str());
    return f.good();
}

/**
* @ingroup Util
* @brief ClassifyDump
* @param [in]:  file_name
* @param [in]:  data
*/
void Util::ClassifyDump(const std::string& file_name, std::shared_ptr<std::string> data)
{
    ddk::Tensor<float> num;
    num.fromarray(reinterpret_cast<float*>(const_cast<char*>(data->c_str())), DATA_NUM);
    (void)num.dump(file_name);

}
