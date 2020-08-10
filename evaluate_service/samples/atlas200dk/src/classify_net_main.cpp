/**
 * @file classify_net_main
 *
 * Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
 *
 * @version 1.0
 *
 * @date 2018-6-7
 */
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include "hiaiengine/api.h"
#include "inc/error_code.h"
#include "inc/common.h"
#include "inc/data_recv.h"
#include "inc/util.h"
uint32_t g_count = 0;
const int MAX_SLEEP_TIMES = 16;
static bool is_test_result_ready = false;

/**
* @ingroup HIAI_InitAndStartGraph
* @brief Initializing and Creating Graph
* @param [in]
*/
HIAI_StatusT HIAI_InitAndStartGraph()
{
    // Step1: Init HiaiEngine
    HIAI_StatusT status = HIAI_Init(0);
    HIAI_ENGINE_LOG("[DEBUG] Go to start Graph");
    // Step2: Create Graph based on the configuration of the proto file.
    status = hiai::Graph::CreateGraph(GRAPH_CONFIG_FILE_PATH);
    if (status != HIAI_OK)
    {
        printf("Fail to create graph\n");
        HIAI_ENGINE_LOG(status, "Fail to create graph");
        return status;
    }
    HIAI_ENGINE_LOG("[DEBUG] create Graph success");

    // Step3: Set the Call Back callback function for the DST Engine.
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph)
    {
        printf("Fail to get the graph-%u instance\n", GRAPH_ID);
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return status;
    }

    // Configure the target data. Target Graph, Target Engine, and Target Port
    hiai::EnginePortID target_port_config;
    target_port_config.graph_id = GRAPH_ID;
    target_port_config.engine_id = DST_ENGINE_ID;
    target_port_config.port_id = DEST_PORT_ID_0;
    graph->SetDataRecvFunctor(target_port_config,
        std::shared_ptr<ClassifyNetDataRecvInterface>(
            new ClassifyNetDataRecvInterface(TEST_DEST_FILE_PATH)));
    return HIAI_OK;
}

/**
* @ingroup CheckAllFileExist
* @brief Check whether all files are generated.
*/
void CheckAllFileExist()
{
    for (int i = 0; i < MAX_SLEEP_TIMES; ++i) {
        if (g_count == SEND_COUNT)
        {
            std::unique_lock <std::mutex> lck(local_test_mutex);
            is_test_result_ready = true;
            printf("File %s generated\n", TEST_DEST_FILE_PATH.c_str());
            HIAI_ENGINE_LOG("Check Result success");
            return;
        }
        printf("Check Result, go into sleep 1 sec\n");
        HIAI_ENGINE_LOG("Check Result, go into sleep 1 sec");
        sleep(1);
    }
    printf("Check Result failed, timeout\n");
    HIAI_ENGINE_LOG("Check Result failed, timeout");
}

/**
* @ingroup main
* @brief main function
* @param [in]: argc, argv
*/
int main(int argc, char* argv[])
{
    printf("========== Test Start ==========\n");
    HIAI_StatusT ret = HIAI_OK;

    // The number of execution program parameters must be greater than or equal to 2.
    // Sample: classify_net_main vgg16/classify_net_main resnet_18
    
    // concatenate test_source_file/test_dest_file/test_graph_file
    TEST_SRC_FILE_PATH = "./test_data/data/input.bin";
    TEST_DEST_FILE_PATH = "./result_files/result_file";
    GRAPH_CONFIG_FILE_PATH = "./test_data/config/graph_sample.prototxt";
    GRAPH_MODEL_PATH = "./test_data/model/davinci_model.om";
    std::string output = "./result_files";
    if (access(output.c_str(), 0) == -1) {
        int flag = mkdir(output.c_str(), 0700);
        if (flag == 0) {
           HIAI_ENGINE_LOG("make output directory successfully");
        }
        else {
           printf("make output directory fail\n");
           HIAI_ENGINE_LOG(HIAI_ARG_NUMBER_NOK, "make output directory fail");
           return -1;
        }
    }
    // Delete the target file.
    remove(TEST_DEST_FILE_PATH.c_str());

    for (int i = 0; i < MAX_SLEEP_TIMES; ++i) {
        if (Util::CheckFileExist(GRAPH_MODEL_PATH)) {
            printf("File %s is ready\n", GRAPH_MODEL_PATH.c_str());
            break;
        }
        sleep(1);
        if (i == MAX_SLEEP_TIMES-1) {
            printf("model file:%s is not existence, timeout\n", GRAPH_MODEL_PATH.c_str());
        }
    }

    // Initializing and Creating Graph
    ret = HIAI_InitAndStartGraph();
    if (HIAI_OK != ret)
    {
        printf("Fail to init and start graph\n");
        HIAI_ENGINE_LOG("Fail to init and start graph");
        return -1;
    }
    printf("Init and start graph succeed\n");

    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph)
    {
        printf("Fail to get the graph-%u instance\n", GRAPH_ID);
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return -1;
    }

    // Send data to Source Engine
    hiai::EnginePortID target_engine;
    target_engine.graph_id = GRAPH_ID;
    target_engine.engine_id = SRC_ENGINE_ID;
    target_engine.port_id = SRC_PORT_ID;

    std::shared_ptr<std::string> src_string =
        std::shared_ptr<std::string>(new std::string(TEST_SRC_FILE_PATH));
    graph->SendData(target_engine, "string",
        std::static_pointer_cast<void>(src_string));

    // Waiting for processing result
    std::thread check_thread(CheckAllFileExist);
    check_thread.join();

    if (is_test_result_ready) {
        printf("========== Test Succeed ==========\n");
    } else {
        printf("========== Test Failed ==========\n");
    }
    // Destroy Graph
    hiai::Graph::DestroyGraph(GRAPH_ID);
    return 0;
}
