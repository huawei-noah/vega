/**
* @file classify_net_ai_engine.h
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/

#ifndef INC_CLASSIFY_NET_AI_ENGINE_H_
#define INC_CLASSIFY_NET_AI_ENGINE_H_
#include <hiaiengine/api.h>
#include <hiaiengine/ai_model_manager.h>
#include <string>
#include <vector>
#include <map>
#include "inc/common.h"
#include <hiaiengine/ai_tensor.h>
class ClassifyNetEngine : public hiai::Engine
{
 public:
    /**
    * @ingroup ClassifyNetEngine
    * @brief ClassifyNetEngine init function
    * @param [in]: config, Configuration parameters
    * @param [in]: model_desc, Model Description
    * @param [out]: HIAI_StatusT
    */
    HIAI_StatusT Init(const hiai::AIConfig &config,
        const std::vector<hiai::AIModelDescription>& model_desc);

    /**
    * @ingroup ~ClassifyNetEngine
    * @brief ~ClassifyNetEngine Destructor function
    */
    ~ClassifyNetEngine();
    /**
    * @ingroup ClassifyNetEngine
    * @brief ClassifyNetEngine executor function
    * @param [in]: CLASSIFYNET_ENGINE_INPUT_SIZE, numbers of in port
    * @param [in]: CLASSIFYNET_ENGINE_OUTPUT_SIZE, numbers of out out
    * @param [out]: HIAI_StatusT
    */
    HIAI_DEFINE_PROCESS(CLASSIFYNET_ENGINE_INPUT_SIZE, CLASSIFYNET_ENGINE_OUTPUT_SIZE);
 private:
    std::map<std::string, std::string> config_;                 // config map
    std::shared_ptr<hiai::AIModelManager> ai_model_manager_;    // Model Manager Instance
    std::vector<std::shared_ptr<hiai::IAITensor>> outDataVec_;
    std::vector<uint8_t*> outData_;
};
#endif  // INC_CLASSIFY_NET_AI_ENGINE_H_
