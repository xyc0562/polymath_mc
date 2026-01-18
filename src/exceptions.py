# Project-specific exceptions
from typing import Dict
import pydash

CODE_PARAM_ERROR = -100
CODE_UNIMPLEMENTED_ERROR = -101
CODE_SOCKET_ERROR = -102
CODE_UNKNOWN_ERROR = -103
CODE_JSON_ENCODE_ERROR = -104
CODE_TIMEOUT_ERROR = -105
CODE_UNSUPPORTED_ERROR = -106
CODE_NO_FUTURE_AVAILABLE = -108
CODE_REQUEST_RATE_LMT = -109
CODE_HTTP_ERROR = -110


def create_gs_error(code=None, msg=None, acc=None):
    return {
        'code': code,
        'acc': acc,
        'msg': msg
    }


class ConfigException(Exception):
    pass


class ApiException(Exception):
    def __init__(self, code=-1, msg='', acc=''):
        super(ApiException, self).__init__()
        self.code = code
        self.msg = msg or ''
        self.acc = acc

    def __str__(self):
        return 'acc: ' + str(self.acc) + ' code: ' + str(self.code) + ', msg: ' + str(self.msg)

    @property
    def _gs_error(self) -> Dict:
        return create_gs_error(code=self.code, msg=self.msg, acc=self.acc)


class JsonEncodeException(ApiException):

    def __init__(self, msg='', acc='', text=''):
        super(JsonEncodeException, self).__init__(CODE_JSON_ENCODE_ERROR, str(msg), acc)
        self.text = str(text)

    def __str__(self):
        return '%s, text: %s' % (super(JsonEncodeException, self).__str__(), self.text)


class RequestRateLmtException(ApiException):

    def __init__(self, msg):
        super(RequestRateLmtException, self).__init__(CODE_REQUEST_RATE_LMT, str(msg))


class PayloadException(ApiException):

    def __init__(self, code=-1, msg='', acc='', payload=None):
        super(PayloadException, self).__init__(code, msg, acc)
        self.payload = payload if pydash.is_dict(payload) else {}

    def __str__(self):
        return 'acc: %s, code: %s, msg: %s, payload: %s' %(self.acc, self.code, self.msg, self.payload)


class HttpException(ApiException):
    def __init__(self, code=-1, msg='', acc=''):
        super(HttpException, self).__init__(code, msg, acc)


class ParamException(ApiException):
    def __init__(self, msg, acc=''):
        super(ParamException, self).__init__(CODE_PARAM_ERROR, msg, acc)


class OrderNotFoundParamException(ParamException):
    def __init__(self, msg, acc=''):
        super(OrderNotFoundParamException, self).__init__('Cancel order error. Order not found, msg: %s' % msg, acc)


class TimeoutException(ApiException):
    def __init__(self, msg, acc=''):
        super(TimeoutException, self).__init__(CODE_TIMEOUT_ERROR, msg, acc)


class UnimplementedException(ApiException):
    def __init__(self, msg, acc=''):
        super(UnimplementedException, self).__init__(CODE_UNIMPLEMENTED_ERROR, msg, acc)


class UnsupportedException(ApiException):
    def __init__(self, msg, acc=''):
        super(UnsupportedException, self).__init__(CODE_UNSUPPORTED_ERROR, msg, acc)


class SocketException(ApiException):
    def __init__(self, msg, acc=''):
        super(SocketException, self).__init__(CODE_SOCKET_ERROR, msg, acc)


class FatalError(Exception):
    def __init__(self, code=-1, msg='', acc=''):
        super(FatalError, self).__init__()
        self.code = code
        self.msg = msg
        self.acc = acc

    def __str__(self):
        return 'code: ' + str(self.code) + ', msg: ' + str(self.msg) + ', acc: ' + str(self.acc)
