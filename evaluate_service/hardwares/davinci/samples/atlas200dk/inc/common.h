/**
* @file vgg16_main
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#ifndef INC_COMMON_H_
#define INC_COMMON_H_
#include <hiaiengine/data_type_reg.h>
#include <iostream>
#include <string>

// Defines the global value.
// Defines the file path.
static std::string TEST_SRC_FILE_PATH = "";// = "./test_data//data/source_test.bin";
static std::string TEST_DEST_FILE_PATH ="";
static std::string GRAPH_CONFIG_FILE_PATH = "";// ="./test_data/config/sample.prototxt";
static std::string GRAPH_MODEL_PATH = "";

// Defines Graph,Engine ID
static const uint32_t GRAPH_ID = 100;
static const uint32_t SRC_ENGINE_ID = 1000;
static const uint32_t SRC_PORT_ID = 0;
static const uint32_t DST_ENGINE_ID = 1002;
static const uint32_t DEST_PORT_ID_0 = 0;
static const uint32_t DEST_PORT_ID_1 = 1;

// Defines Output shape
const std::vector<uint32_t> DATA_NUM = {10};

// Defines the global value
static std::mutex local_test_mutex;
static std::condition_variable local_test_cv_;
static const uint32_t MAX_SLEEP_TIMER = 30 * 60;
static const uint32_t MIN_ARG_VALUE = 2;
// Defines image parameters.
static const float IMG_DEPTH = 1.0;
static const uint32_t SEND_COUNT = 100;
static const std::string modelName = "ClassifyModel";

// Defines the message_type character string.
static const std::string message_type_engine_trans = "EngineTransT";

// Defines the number of Engine ports.
// Source Engine
#define SOURCE_ENGINE_INPUT_SIZE    1
#define SOURCE_ENGINE_OUTPUT_SIZE   1

// Dest Engine
#define DEST_ENGINE_INPUT_SIZE      1
#define DEST_ENGINE_OUTPUT_SIZE     1

// ClassifyNet Engine
#define CLASSIFYNET_ENGINE_INPUT_SIZE    1
#define CLASSIFYNET_ENGINE_OUTPUT_SIZE   1

#define IMAGE_INFO_DATA_NUM         (3)

// Defines the transmission structure.
typedef struct EngineTrans
{
    std::string trans_buff;
    uint32_t buffer_size;
    HIAI_SERIALIZE(trans_buff, buffer_size);
}EngineTransT;

#endif  // INC_COMMON_H_
