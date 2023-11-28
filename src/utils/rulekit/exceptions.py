from jpype import JException

class JavaBackendException(Exception):

    def __init__(self, java_exception: JException) -> None:
        super().__init__()
        self.message: str = (
            java_exception.toString() + 
            '\n\nFor detailed java stacktrace see `java_stack_trace` attribute of this exception object.'
        )
        self.java_exception: JException  = java_exception
        self.java_stack_trace: str = self._get_java_stack_trace()


    def _get_java_stack_trace(self) -> str:
        stack_trace_message: str = ''
        for element in self.java_exception.getStackTrace():
            stack_trace_message += f'\n{element.toString()}'
        return stack_trace_message


    def __str__(self) -> str:
        return str(self.message)