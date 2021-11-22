#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:main.py.py
@time:2021/11/22
"""
import uvicorn
from src.server import app
from src.server.router import router

# app = create_app()
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run('main:app',host="0.0.0.0",port=32666,reload=True)