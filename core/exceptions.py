class ServiceException(Exception):
    """服务层异常"""

    pass


class ServiceNotFoundException(ServiceException):
    """未找到记录"""

    pass


class ServiceMissMessageException(ServiceException):
    """缺少信息"""

    pass
