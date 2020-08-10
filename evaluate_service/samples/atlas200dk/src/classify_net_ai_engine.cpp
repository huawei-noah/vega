/**
* @file classify_net_ai_engine.cpp
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#include <unistd.h>
#include <thread>
#include <hiaiengine/api.h>
#include <hiaiengine/data_type.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include "inc/classify_net_ai_engine.h"
#include "inc/error_code.h"
#include "inc/sample_data.h"
#include "hiaiengine/ai_memory.h"
#include <time.h>

 ClassifyNetEngine::~ClassifyNetEngine() {
    // Release the pre-allocated memory of outData.
    for (auto buffer : outData_) {
        if (buffer != nullptr) {
            hiai::HIAIMemory::HIAI_DVPP_DFree(buffer);
            buffer = nullptr;
        }
    }
 }
/**
* @ingroup ClassifyNetEngine
* @brief ClassifyNetEngine init function
* @param [in]:arg0
*/
HIAI_StatusT ClassifyNetEngine::Init(const hiai::AIConfig &config,
    const std::vector<hiai::AIModelDescription>& model_desc)
{
    HIAI_ENGINE_LOG(this, HIAI_OK, "ClassifyNetEngine Init");
    hiai::AIStatus ret = hiai::SUCCESS;

    // Obtaining Configuration Parameters
    config_.clear();
    for (auto item : config.items()) {
        config_[item.name()] = item.value();
    }
    if (nullptr == ai_model_manager_) {
        ai_model_manager_ = std::make_shared<hiai::AIModelManager>();
    }

    // Init Model
    const char* model_path = config_["model_path"].c_str();
    std::vector<hiai::AIModelDescription> model_desc_vec;
    hiai::AIModelDescription model_desc_;
    model_desc_.set_path(model_path);
    model_desc_.set_key("");
    model_desc_.set_name(modelName);
    model_desc_vec.push_back(model_desc_);
    ret = ai_model_manager_->Init(config, model_desc_vec);

    if (ret != hiai::SUCCESS) {
        HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_INIT_FAIL,
            "hiai ai model manager init fail");
        return HIAI_AI_MODEL_MANAGER_INIT_FAIL;
    }

    std::vector<hiai::TensorDimension> inputTensorVec;
    std::vector<hiai::TensorDimension> outputTensorVec;
    ret = ai_model_manager_->GetModelIOTensorDim(modelName, inputTensorVec, outputTensorVec);
    if (ret != hiai::SUCCESS) {
        HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_INIT_FAIL,
            "hiai ai model manager init fail");
        return HIAI_AI_MODEL_MANAGER_INIT_FAIL;
    }
    // allocate OutData in advance
    HIAI_StatusT hiai_ret = HIAI_OK;
    for (size_t index = 0; index < outputTensorVec.size(); index++) {
        hiai::AITensorDescription outputTensorDesc = hiai::AINeuralNetworkBuffer::GetDescription();
        uint8_t* buffer = nullptr;
        hiai_ret = hiai::HIAIMemory::HIAI_DMalloc(outputTensorVec[index].size, (void*&)buffer, 1000);
        if ((hiai_ret != HIAI_OK) || (buffer == nullptr)) {
            printf("HIAI_DMalloc failed\n");
            continue;
        }
        outData_.push_back(buffer);
        shared_ptr<hiai::IAITensor> outputTensor =
            hiai::AITensorFactory::GetInstance()->CreateTensor(outputTensorDesc, buffer, outputTensorVec[index].size);
        outDataVec_.push_back(outputTensor);
    }

    HIAI_ENGINE_LOG(this, HIAI_OK, "ClassifyNetEngine init success");
    return HIAI_OK;
}

/**
* @ingroup ClassifyNetEngine
* @brief ClassifyNetEngine Process function
* @param [in]:arg0
*/
HIAI_IMPL_ENGINE_PROCESS("ClassifyNetEngine", ClassifyNetEngine, CLASSIFYNET_ENGINE_INPUT_SIZE)
{
    HIAI_ENGINE_LOG(this, HIAI_OK, "ClassifyNetEngine Process");
    HIAI_StatusT ret = HIAI_OK;
    std::vector<std::shared_ptr<hiai::IAITensor>> inDataVec;

    std::shared_ptr<EngineTransNewT> input_arg =
        std::static_pointer_cast<EngineTransNewT>(arg0);
    if (nullptr == input_arg)
    {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG,
            "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }
    // Transfer buffer to Framework directly, only one inputsize
    hiai::AITensorDescription inputTensorDesc =
        hiai::AINeuralNetworkBuffer::GetDescription();
    shared_ptr<hiai::IAITensor> inputTensor =
        hiai::AITensorFactory::GetInstance()->CreateTensor(inputTensorDesc,
        input_arg->trans_buff.get(), input_arg->buffer_size);
    // AIModelManager. fill in the input data.
    inDataVec.push_back(inputTensor);

    hiai::AIContext ai_context;
    clock_t start_time=clock();
    // Process work
    ret = ai_model_manager_->Process(ai_context,
        inDataVec, outDataVec_, 0);
    clock_t end_time=clock(); 
    cout<< "costTime  "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<endl;
    if (hiai::SUCCESS != ret)
    {   
        HIAI_ENGINE_LOG(this, HIAI_AI_MODEL_MANAGER_PROCESS_FAIL,
            "Fail to process ai model manager");
        return HIAI_AI_MODEL_MANAGER_PROCESS_FAIL;
    }
    cout<<"sucess"<<endl;
    // Convert the generated data to the buffer of the string type and send the data.
    for (uint32_t index = 0; index < outDataVec_.size(); index++)
    {
        HIAI_ENGINE_LOG(this, HIAI_OK, "ClassifyNetEngine SendData");
        std::shared_ptr<hiai::AINeuralNetworkBuffer> output_data = std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(outDataVec_[index]);
        std::shared_ptr<std::string> output_string_ptr =
            std::shared_ptr<std::string>(new std::string((char*)output_data->GetBuffer(), output_data->GetSize()));
        hiai::Engine::SendData(0, "string",
            std::static_pointer_cast<void>(output_string_ptr));
    }
    inDataVec.clear();
    return HIAI_OK;
}
