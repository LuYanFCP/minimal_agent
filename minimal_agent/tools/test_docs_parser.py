
from tools.docs_parser import GoogleStyleDocsParser

def add_numbers(a: int, b: int) -> int:
    """
    计算两个整数的和。


    Args:
        a (int): 第一个整数。
        b (int): 第二个整数。

    Returns:
        int: 两个整数的和。

    Raises:
        TypeError: 如果输入的参数不是整数。

    Examples:
        >>> add_numbers(1, 2)
        3
        >>> add_numbers(-1, 1)
        0
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers.")
    
    return a + b


def test_google_parse() -> None:
    parser = GoogleStyleDocsParser()

    assert add_numbers.__doc__

    result = parser.parse(add_numbers, add_numbers.__doc__)
    assert result.description == "计算两个整数的和。"
    assert result.name ==  'add_numbers'
    assert len(result.args) == 2
    assert result.args[0].arg_name == 'a'
    assert result.args[0].arg_type == 'int'
    assert result.args[0].arg_desc == '第一个整数。'

    assert result.args[1].arg_name == 'b'
    assert result.args[1].arg_type == 'int'
    assert result.args[1].arg_desc == '第二个整数。'

