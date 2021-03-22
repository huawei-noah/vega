/**
* @file data_recv.h
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-6-7
*/
#ifndef INC_DATA_RECV_H_
#define INC_DATA_RECV_H_
#include <hiaiengine/api.h>
#include <string>
class ClassifyNetDataRecvInterface : public hiai::DataRecvInterface
{
 public:
    /**
    * @ingroup ClassifyNetDataRecvInterface
    * @brief construct function
    * @param [in]desc:std::string
    */
    ClassifyNetDataRecvInterface(const std::string& filename) :
        file_name_(filename) {}

    /**
    * @ingroup ClassifyNetDataRecvInterface
    * @brief RecvData RecvData callback,Save the File
    * @param [in]
    */
    HIAI_StatusT RecvData(const std::shared_ptr<void>& message);

 private:
    std::string file_name_;     // Target Save File
};
#endif  // INC_DATA_RECV_H_
