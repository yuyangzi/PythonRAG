#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2024/7/30 10:11
@Author  : wang yi ming
@file for: 
"""

import redis
from config.local_config import REDIS


class RedisPool:
    """
    redis 连接池
    """
    REDIS_POOL = redis.ConnectionPool(host=REDIS["host"], port=REDIS["port"], password=REDIS["password"],
                                      max_connections=REDIS.get("max_conn") or 5, db=0, decode_responses=True)


class RedisClient(RedisPool):
    """
    redis 客户端
    """

    def __init__(self):
        pass

    def get_client(self):
        """
        获取一个连接
        :return:
        """
        pool = self.__class__.REDIS_POOL
        client = redis.StrictRedis(connection_pool=pool)
        return client


if __name__ == '__main__':
    redisClient = RedisClient()
    client = redisClient.get_client()
    pipe = client.pipeline()
    pipe.set("lzm_name", "lzm")
    pipe.set("lzm_age", None)
    import time

    # time.sleep(10)
    r = pipe.execute()
    print(r)
