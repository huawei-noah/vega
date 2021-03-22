/**
* @file sample_data.cpp
*
* Copyright (C) <2018>  <Huawei Technologies Co., Ltd.>. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "hiaiengine/data_type.h"
#include "hiaiengine/data_type_reg.h"

// Register the structure that the Engine transfers.
typedef struct EngineTransNew
{
    std::shared_ptr<uint8_t> trans_buff;
    uint32_t buffer_size;   // buffer size
}EngineTransNewT;