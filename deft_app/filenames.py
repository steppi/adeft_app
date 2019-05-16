"""These functions are used to escape and unescape lower case characters
filenames to make the deft_app compatible with case insensitive file systems.
The Deft app design involves using filenames as keys in an implicit database,
which is broken in a case insensitive file system.
"""


_escape_map = {'/': '0',
               '\\': '1',
               '?': '2',
               '%': '3',
               '*': '4',
               ':': '5',
               '|': '6',
               '"': '7',
               '<': '8',
               '>': '9',
               '.': ',',
               '_': '_'}


def _escape(char):
    if char in _escape_map:
        return '_' + _escape_map[char]
    elif char.islower():
        return '_' + char.upper()
    else:
        return char


def escape_filename(filename):
    """Convert filename for one with escape character before lowercase

    This is done to handle case insensitive file systems. _ is used as an
    escape character. It is also an escape character for itself.
    """
    return ''.join([_escape(char) for char in filename])
