"""These functions are used to escape and unescape lower case characters
filenames to make the deft_app compatible with case insensitive file systems.
The Deft app design involves using filenames as keys in an implicit database,
which is broken in a case insensitive file system.
"""


def escape_lower_case(file_name):
    """Convert filename for one with escape character before lowercase

    This is done to handle case insensitive file systems. _ is used as an
    escape character. It is also an escape character for itself.
    """
    return ''.join(['_' + char.upper() if char.islower()
                    or char == '_' else char for char in file_name])


def unescape_lower_case(file_name):
    """Inverse of previous function."""
    underscore = False
    output = []
    for char in file_name:
        if underscore:
            output.append(char.lower())
            underscore = False
        elif char == '_':
            underscore = True
        else:
            output.append(char)
    return ''.join(output)
