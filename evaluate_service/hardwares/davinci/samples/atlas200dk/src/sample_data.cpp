/**
* @file sample_main.cpp
*
* Copyright (C) <2018>  <Huawei Technologies Co., Ltd.>. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "inc/sample_data.h"
#include "hiaiengine/data_type.h"
// Registers serialization and deserialization functions.
/**
* @ingroup hiaiengine
* @brief GetTransSearPtr,        Serializes Trans data.
* @param [in] : data_ptr         Struct Pointer
* @param [out]:struct_str       Struct buffer
* @param [out]:data_ptr         Struct data pointer
* @param [out]:struct_size      Struct size
* @param [out]:data_size        Struct data size
* @author w00437212
*/
void GetTransSearPtr(void* inputPtr, std::string& ctrlStr, uint8_t*& dataPtr, uint32_t& dataLen)
{
    EngineTransNewT* engine_trans = (EngineTransNewT*)inputPtr;
    ctrlStr  = std::string((char*)inputPtr, sizeof(EngineTransNewT));
    dataPtr = (uint8_t*)engine_trans->trans_buff.get();
    dataLen = engine_trans->buffer_size;
}

/**
* @ingroup hiaiengine
* @brief GetTransSearPtr,             Deserialization of Trans data
* @param [in] : ctrl_ptr              Struct Pointer
* @param [in] : data_ptr              Struct data Pointer
* @param [out]:std::shared_ptr<void> Pointer to the pointer that is transmitted to the Engine
* @author w00437212
*/
std::shared_ptr<void> GetTransDearPtr(const char* ctrlPtr, const uint32_t& ctrlLen, const uint8_t* dataPtr, const uint32_t& dataLen)
{
    EngineTransNewT* engine_trans = (EngineTransNewT*)ctrlPtr;
    std::shared_ptr<EngineTransNewT> engineTranPtr(new EngineTransNewT);
    engineTranPtr->buffer_size = engine_trans->buffer_size;
    engineTranPtr->trans_buff.reset(const_cast<uint8_t*>(dataPtr), hiai::Graph::ReleaseDataBuffer);
    return std::static_pointer_cast<void>(engineTranPtr);
}

// RegisterEngineTransNewT
HIAI_REGISTER_SERIALIZE_FUNC("EngineTransNewT", EngineTransNewT, GetTransSearPtr, GetTransDearPtr);