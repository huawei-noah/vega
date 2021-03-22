/**
* @file classify_net_host.h
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#ifndef INC_CLASSIFY_NET_HOST_H_
#define INC_CLASSIFY_NET_HOST_H_
#include <hiaiengine/api.h>
#include <hiaiengine/multitype_queue.h>
#include "inc/common.h"
/*
* Source Engine
*/
class SourceEngine : public hiai::Engine
{
    /**
    * @ingroup SourceEngine
    * @brief SourceEngine Process function
    * @param [in]: SOURCE_ENGINE_INPUT_SIZE, Source Engine in port
    * @param [in]: SOURCE_ENGINE_OUTPUT_SIZE, Source Engine out port
    * @param [out]: HIAI_StatusT
    */
    HIAI_DEFINE_PROCESS(SOURCE_ENGINE_INPUT_SIZE, SOURCE_ENGINE_OUTPUT_SIZE)
};

/*
* Dest Engine
*/
class DestEngine : public hiai::Engine
{
 public:
    DestEngine() :
        input_que_(DEST_ENGINE_INPUT_SIZE) {}
    /**
    * @ingroup SourceEngine
    * @brief SourceEngine Process function
    * @param [in]: DEST_ENGINE_INPUT_SIZE, Source Engine in port
    * @param [in]: DEST_ENGINE_OUTPUT_SIZE, Source Engine out port
    * @param [out]: HIAI_StatusT
    */
    HIAI_DEFINE_PROCESS(DEST_ENGINE_INPUT_SIZE, DEST_ENGINE_OUTPUT_SIZE)

 private:
    hiai::MultiTypeQueue input_que_;
};
#endif  // INC_CLASSIFY_NET_HOST_H_
