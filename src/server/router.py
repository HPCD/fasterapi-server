#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:app.py
@time:2021/11/22
"""
from typing import Dict
from fastapi import APIRouter,BackgroundTasks,Query,Request,status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.asyn_methods import async_veri
# from src.server import create_app
router = APIRouter(prefix='/faster-server/v1')



class RequestParams(BaseModel):
    image: str
    image_type: str =Query(...,min_length=3,max_length=6)



def task(msg):
    print('Task {} !'.format(msg))


@router.post('/test')
async def test(reqest_params: RequestParams,bg_tasks: BackgroundTasks):

    data = dict(reqest_params)
    print("classd",reqest_params,reqest_params.image)
    bg_tasks.add_task(async_veri,dict(reqest_params))
    return {"code":'E001','message':"Task is Running !"}