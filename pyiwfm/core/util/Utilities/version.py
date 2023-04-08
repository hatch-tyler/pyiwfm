from pyiwfm.core.util.Utilities.general_utilities import first_location, clean_special_characters
from pyiwfm.core.util.Utilities.message_logger import set_last_message, message_array, FATAL

MODNAME = "Class_Version::"

# Note: Version destructor 'kill' is not included because it is unnecessary in python

class Version:
    def __init__(self, version):
        self._version = version

    @classmethod
    def version_new_from_components(cls, version, revision):
        # get the length of the revision string
        rev_length = len(revision)

        return cls(version[:-rev_length] + revision)

    @classmethod
    def version_new_from_full_string(cls, version):
        return cls(version)

    def get_version(self):
        """ Return the version """
        return self._version

    def is_defined(self):
        if len(self._version) == 0:
            return False
        return True
    
def read_version(in_file, component):
    """
    Read version from a file
    
    Parameters
    ----------
    in_file : file object
        open file object
        
    component : str
        IWFM component e.g. Stream, Root Zone, etc.

    Returns
    -------
    tuple[str, int]
        version, status
    """
    this_procedure = MODNAME + "ReadVersion"

    line = in_file.read_line()
    line = clean_special_characters(line)
    start = first_location("#", line)

    if start <= 0:
        message_array.append(f"Error in identifying the version number of the {component} component!")
        message_array.append("Make sure that the version number is listed at the first line of the")
        message_array.append(f"{component} input file (see the input file template for format)")
        set_last_message(message_array, FATAL, this_procedure)
        version = ""
        status = -1
        return version, status
    
    version = line[1:].strip()
    status = 0

    return version, status
    