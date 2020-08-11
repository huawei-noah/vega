/**
* @file classify_net_host.cpp
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#include <unistd.h>
#include <thread>
#include <hiaiengine/data_type.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include "inc/classify_net_host.h"
#include "inc/common.h"
#include "inc/util.h"
#include "inc/error_code.h"
#include "inc/sample_data.h"
#include "hiaiengine/ai_memory.h"
#include <time.h>
//using namespace std;
//for performance
int64_t start_time = 0;
int64_t end_time = 0;
int64_t aifull_time = 0;
int64_t total_time = 0;
int32_t batch_size = 1;
extern int64_t g_count;

void Delay(int time)
{
  clock_t now=clock();
  while(clock()-now < time);
}

void deleteMemoryDmalloc(void* ptr)
{
    hiai::HIAIMemory::HIAI_DVPP_DFree(ptr);
}

void deleteMemoryNew(void* ptr)
{
    if(ptr != nullptr) {
        delete[] reinterpret_cast<char*>(ptr);
    }
}

/**
* @ingroup SourceEngine
* @brief SourceEngine Process function
* @param [in]:arg0
*/
HIAI_IMPL_ENGINE_PROCESS("SourceEngine", SourceEngine, SOURCE_ENGINE_INPUT_SIZE)
{
    HIAI_ENGINE_LOG(this, HIAI_OK, "SourceEngine Process");
    // Obtain the path of the original file.
    std::shared_ptr<std::string> input_arg =
        std::static_pointer_cast<std::string>(arg0);
    if (nullptr == input_arg)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG,
            "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }

    for (uint32_t index = 0; index < SEND_COUNT; index++)
    {
        // Reads data and generates information.
        uint32_t tmpBuffSize = 0;
        bool isDMalloc = true;
        char* tmpBuffData = Util::ReadBinFile(input_arg, &tmpBuffSize, batch_size, isDMalloc);
        if (tmpBuffData == nullptr) {
            HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG,
                "alloc send buffer fail");
            return HIAI_INVALID_INPUT_MSG;
        }
        std::shared_ptr<EngineTransNewT> tmp_raw_data_ptr = std::make_shared<EngineTransNewT>();
        tmp_raw_data_ptr->buffer_size = tmpBuffSize;
        if(isDMalloc == true) {
            tmp_raw_data_ptr->trans_buff.reset((unsigned char*)tmpBuffData, deleteMemoryDmalloc);
        }
        else {
            tmp_raw_data_ptr->trans_buff.reset((unsigned char*)tmpBuffData, deleteMemoryNew);
        }
        // Transferred to ClassifyNet Engine
        HIAI_ENGINE_LOG(this, HIAI_OK, "SourceEngine Process:: begin to Senddata");
        hiai::Engine::SendData(0, "EngineTransNewT",
            std::static_pointer_cast<void>(tmp_raw_data_ptr), 10000);    
    }
    HIAI_ENGINE_LOG(this, HIAI_OK, "SourceEngine Process Success");
    return HIAI_OK;
}

/**
* @ingroup DestEngine
* @brief DestEngine Process function
* @param [in]:arg0
*/
HIAI_IMPL_ENGINE_PROCESS("DestEngine", DestEngine, DEST_ENGINE_INPUT_SIZE)
{
    HIAI_ENGINE_LOG(this, HIAI_OK, "DestEngine Process");
    std::shared_ptr<std::string> data_result_ptr =
        std::static_pointer_cast<std::string>(arg0);
    // Check whether the data_result_ptr is valid.
    if (nullptr == data_result_ptr)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG,
            "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }
    HIAI_ENGINE_LOG(this, HIAI_OK, "DestEngine Process:: already receive result data");
    // Send data_num and data_bbox to the callback function.
    hiai::Engine::SendData(0, "string",
        std::static_pointer_cast<void>(data_result_ptr));
    HIAI_ENGINE_LOG(this, HIAI_OK, "DestEngine Process Success");
    return HIAI_OK;
}
