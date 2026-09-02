import sys
import traceback


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys):
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occurred in {filename}, line {line_number}: {error_message}"
        return error_message

    def __str__(self):
        return self.error_message