class StatusCodes:
    # 1xx 信息性状态码
    CONTINUE = 100  # 继续。客户端应继续发送请求的剩余部分，或者如果请求已经完成，则忽略这个响应。
    SWITCHING_PROTOCOLS = 101  # 切换协议。服务器根据客户端的请求切换协议。

    # 2xx 成功状态码
    OK = 200  # 请求成功。响应体中包含请求的资源。
    CREATED = 201  # 请求成功，并且服务器创建了一个新的资源。
    ACCEPTED = 202  # 请求已接受，但处理尚未完成。
    NON_AUTHORITATIVE_INFORMATION = 203  # 非权威性信息。请求成功，但响应的信息不是由原始服务器直接返回的。
    NO_CONTENT = 204  # 请求成功，但响应体为空。
    RESET_CONTENT = 205  # 重置内容。请求成功，客户端应重置文档视图。
    PARTIAL_CONTENT = 206  # 部分内容。服务器已经成功处理了部分GET请求。

    # 3xx 重定向状态码
    MULTIPLE_CHOICES = 300  # 多重选择。请求的资源有多个可用的表示，客户端需要进一步选择。
    MOVED_PERMANENTLY = 301  # 永久移动。请求的资源已永久移动到新的URL。
    FOUND = 302  # 临时移动。请求的资源临时移动到新的URL。
    SEE_OTHER = 303  # 查看其他。请求的资源可以被找到，但需要查看其他URL。
    NOT_MODIFIED = 304  # 未修改。请求的资源未修改，客户端可以使用缓存的版本。
    USE_PROXY = 305  # 使用代理。请求的资源必须通过代理访问。
    TEMPORARY_REDIRECT = 307  # 临时重定向。请求的资源临时移动到新的URL，但方法不变。
    PERMANENT_REDIRECT = 308  # 永久重定向。请求的资源已永久移动到新的URL，但方法不变。

    # 4xx 客户端错误状态码
    BAD_REQUEST = 400  # 请求语法错误，服务器无法理解。
    UNAUTHORIZED = 401  # 请求未授权，需要认证信息。
    PAYMENT_REQUIRED = 402  # 需要支付。保留用于将来可能的使用。
    FORBIDDEN = 403  # 服务器理解请求，但拒绝执行。
    NOT_FOUND = 404  # 请求的资源未找到。
    METHOD_NOT_ALLOWED = 405  # 请求方法不被允许。
    NOT_ACCEPTABLE = 406  # 请求的资源无法使用客户端请求的格式表示。
    PROXY_AUTHENTICATION_REQUIRED = 407  # 需要代理认证。
    REQUEST_TIMEOUT = 408  # 请求超时。
    CONFLICT = 409  # 请求冲突。
    GONE = 410  # 请求的资源已永久删除。
    LENGTH_REQUIRED = 411  # 请求需要一个`Content-Length`头。
    PRECONDITION_FAILED = 412  # 前置条件失败。
    PAYLOAD_TOO_LARGE = 413  # 请求体过大。
    URI_TOO_LONG = 414  # 请求的URI过长。
    UNSUPPORTED_MEDIA_TYPE = 415  # 请求的格式不被支持。
    RANGE_NOT_SATISFIABLE = 416  # 请求的范围无法满足。
    EXPECTATION_FAILED = 417  # 期望失败。
    I_AM_A_TEAPOT = 418  # 我是一个茶壶。用于拒绝请求，常用于测试。
    MISDIRECTED_REQUEST = 421  # 请求被错误地路由。
    UNPROCESSABLE_ENTITY = 422  # 请求被服务器正确解析，但包含无效的字段。
    LOCKED = 423  # 资源被锁定。
    FAILED_DEPENDENCY = 424  # 由于依赖关系失败。
    TOO_EARLY = 425  # 太早。服务器不愿意风险处理请求。
    UPGRADE_REQUIRED = 426  # 需要升级。
    PRECONDITION_REQUIRED = 428  # 需要前置条件。
    TOO_MANY_REQUESTS = 429  # 请求过多。
    REQUEST_HEADER_FIELDS_TOO_LARGE = 431  # 请求头字段过大。
    UNAVAILABLE_FOR_LEGAL_REASONS = 451  # 因法律原因不可用。

    # 5xx 服务器错误状态码
    INTERNAL_SERVER_ERROR = 500  # 服务器内部错误，无法完成请求。
    NOT_IMPLEMENTED = 501  # 服务器不支持请求的功能。
    BAD_GATEWAY = 502  # 服务器作为网关或代理，从上游服务器收到无效响应。
    SERVICE_UNAVAILABLE = 503  # 服务器暂时不可用，通常是因为过载或维护。
    GATEWAY_TIMEOUT = 504  # 服务器作为网关或代理，未及时从上游服务器收到响应。
    HTTP_VERSION_NOT_SUPPORTED = 505  # 服务器不支持请求的HTTP版本。
    VARIANT_ALSO_NEGOTIATES = 506  # 变体也协商。
    INSUFFICIENT_STORAGE = 507  # 存储空间不足。
    LOOP_DETECTED = 508  # 检测到循环。
    NOT_EXTENDED = 510  # 未扩展。
    NETWORK_AUTHENTICATION_REQUIRED = 511  # 需要网络认证。
