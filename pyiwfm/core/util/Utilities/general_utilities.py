'''
general_utilities.py
Author: Tyler Hatch PhD, PE

This is the general_utilities module of the python version of IWFM
'''
import numpy as np

# local imports
from message_logger import set_last_message
from message_logger import FATAL

this_procedure = 'GeneralUtilities'

# Public text utilities
def int_to_text(integer_value):
    ''' converts an integer to a string, if possible '''
    return str(integer_value)

def text_to_int(text_value):
    ''' converts string to an integer, if possible '''
    return int(text_value)
              
def arrange_text_master(input_data, line_length):
    ''' concatenates and centers text in a line based on the provided line_length
    
    Parameters
    ----------
    input_data : list or str
        string or list of strings to be arranged
    
    line_length : int
        total length of line to center text
    
    Returns
    -------
    str
        concatenated strings centered based on line_length
    '''
    if isinstance(input_data, list):
        work_string = ''.join(input_data)
    else:
        work_string = input_data

    string_length = len(work_string)

    if string_length > line_length:
        return work_string[:line_length]
    else:
        number_of_leading_spaces = (line_length - string_length)/2
        return ' ' * number_of_leading_spaces + work_string

def arrange_1_string(input_data, line_length):
    return arrange_text_master(input_data, line_length)

def arrange_2_string(input_string1, input_string2, line_length):
    input_data = [input_string1, input_string2]

    return arrange_text_master(input_data, line_length)

def arrange_3_string(input_string1, input_string2, input_string3, line_length):
    input_data = [input_string1, input_string2, input_string3]

    return arrange_text_master(input_data, line_length)

def arrange_4_string(input_string1, input_string2, input_string3, input_string4, line_length):
    input_data = [input_string1, input_string2, input_string3, input_string4]

    return arrange_text_master(input_data, line_length)

def arrange_5_string(input_string1, input_string2, input_string3, input_string4, input_string5, line_length):
    input_data = [input_string1, input_string2, input_string3, input_string4, input_string5]

    return arrange_text_master(input_data, line_length)

def first_location(character, search_string, back=False):
    ''' returns the index of the character in the search_string. 
    Parameters
    ----------
    character : str
        single character to search for
        
    search_string : str
        string to search for the character
        
    back : bool, default=False
        if False, returns first occurrence from the left. 
        if True, returns first occurrence from the right
    
    Returns
    -------
    int
        index of the string corresponding to first occurrence of the character
        or None if character is not found
    '''
    if not back:
        return search_string.find(character)

    else:
        return search_string.rfind(character)

def count_occurence(character, search_string):
    ''' returns the number of occurrences of a character within a string '''
    count = 0
    for c in search_string:
        if c == character:
            count += 1

    return count

def find_substring_in_string(substring, string, case_sensitive=True):
    ''' finds the index of the beginning of the substring in the string 
    
    Parameters
    ----------
    substring : str
        string being looked for in the other string
        
    string : str
        string searched for the presence of the substring 
        
    case_sensitive : bool, default=True
        flag to determine if the substring needs to match the case of the string 
        
    Returns
    -------
    int : starting index of substring in string. if substring is not found returns -1
    '''
    if case_sensitive:
        return string.find(substring)

    else:
        string = string.lower()
        return string.find(substring.lower())

def upper_case(string):
    return string.upper()

def lower_case(string):
    return string.lower()

def strip_text_until_character(in_text, character, back=False):
    location = first_location(character, in_text, back)
    
    if location > -1:
        return in_text[:location]
    
    return in_text

def clean_special_characters(string):
    ''' replaces special characters (ascii values less than 32) with a space '''
    for i, c in enumerate(string):
        if ord(c) < 32:
            string[i] = " "
    
    return string

def clean_special_characters_string_array(string_array):
    ''' cleans special characters for each string in a string array (list) '''
    return [clean_special_characters(string) for string in string_array]

def get_start_location(string, column_number):
    ''' returns the index of the start location of the column 
    
    Parameters
    ----------
    string : str
        data string to locate start location of column number
        
    column_number : int
        column number to locate start location
        
    Returns
    -------
    int
        index for the start location corresponding to the column number
    '''
    work_string = clean_special_characters(string)
    work_string = strip_text_until_character(work_string, '/', back=True)

    # count columns, spaces or comma in string
    column_counter = 0

    # initially set current location to empty
    is_current_location_empty = True

    # loop through each character in the string
    for work_location, c in enumerate(string):
        is_previous_location_empty = is_current_location_empty
        is_current_location_empty = False
        
        # check if the current character is a space or comma
        if (c == ' ') or (c == ','):
            is_current_location_empty = True
        
        # only increase the column count if the current location is not empty 
        # and the previous location was empty
        if is_previous_location_empty and (not is_current_location_empty):
            column_counter += 1

        # when the column counter equals the column number return the location
        if column_counter == column_number:
            return work_location

def prepare_title(title_lines, title_length, title_start_location):
    ''' formats the title
    '''
    lead = '*'

    title = lead + '*' * title_length + '\n'

    for line in title_lines:
        title = title + lead + '*' + line[:title_length-2] + '*' + '\n'

    title = title + lead + '*' * title_length

    return title

def alloc_1D_int_array():
    raise NotImplementedError

def alloc_1D_real_array():
    raise NotImplementedError

def alloc_1D_logical_array():
    raise NotImplementedError

def alloc_1D_character_array():
    raise NotImplementedError

def alloc_2D_character_array():
    raise NotImplementedError

def alloc_2D_real_array():
    raise NotImplementedError

def alloc_3D_real_array():
    raise NotImplementedError

def alloc_2d_int_array():
    raise NotImplementedError

def alloc_pointer_to_1D_logical_array():
    raise NotImplementedError

def alloc_pointer_to_1D_real_array():
    raise NotImplementedError

def alloc_pointer_to_1D_character_array():
    raise NotImplementedError

def locate_item_in_array(item, list_of_items):
    ''' returns the index of the item in the list of items
    
    Parameters
    ----------
    item : int, float, or str
        item to locate in the array
        
    list_of_items : list, np.ndarray
        array of items

    Returns
    -------
    int
        index of the first location matching them item 
    '''
    if isinstance(list_of_items, list):
        return list_of_items.index(item)

    if isinstance(list_of_items, np.ndarray):
        return np.where(list_of_items == item)[0][0]

def locate_set_in_array(items_list, list_of_items):
    ''' locates the first occurrence of each item in the items_list within the list_of_items
    
    Parameters
    ----------
    items_list : list or np.ndarray
        array of items to locate in list_of_items
        
    list_of_items : list or np.ndarray
        array of items where the index is located
        
    Returns
    -------
    list or np.ndarray
        list of locations of first occurrences 
    '''
    locations = []
    
    for item in items_list:
        locations.append(locate_item_in_array(item, list_of_items))

    return locations

def normalize_array(arr):
    ''' rescales an array by its sum '''
    if isinstance(arr, np.ndarray):
        total = arr.sum()

        if total != 0:
            return arr/total

    if isinstance(arr, list):
        total = sum(arr)

        if total != 0:
            return [val/total for val in arr]

def shell_sort_no_second_array(arr):
    ''' sorts an integer array using shell sort 
    
    Parameters
    ----------
    arr : list or np.ndarray
    
    Returns
    -------
    list or np.ndarray
        same as input 
    '''
    
    # check that arr is array-like
    if not isinstance(arr, (list, np.ndarray)):
        raise TypeError("arr must be a list or np.ndarray")

    # check that array of type list is an integer array
    if isinstance(arr, list) and not all([isinstance(val, int) for val in arr]):
        raise ValueError("All values in array must be integers")

    if isinstance(arr, np.ndarray) and arr.dtype != np.dtype('int'):
        raise ValueError("arr must be an integer array")

    # set initial increment
    inc = 1

    # determine length of arr
    n = len(arr)

    # scale increment based on length of array
    while True:
        inc = 3*inc + 1

        if inc > n:
            break
    
    # sort array
    while True:
        # rescale increment for subset of array
        inc = int(inc/3)
        
        # loop through subset of array and compare values
        for i in range(inc, n):
            va = arr[i]
            j = i
            
            # if array value at location j-inc is greater than location i 
            # insert larger value to larger index
            while arr[j-inc] > va:
                arr[j] = arr[j-inc]
                j =  j-inc
                
                # exit loop when index j is less than or equal to current inc
                if j <= inc:
                    break
            
            arr[j] = va

        # exit loop if current inc is less than or equal to first index
        # in fortran arrays are 1-based whereas python arrays are 0-based
        if inc <= 0:
            break

    return arr
    

def shell_sort_second_array(arr1, arr2):
    ''' sorts a second array based on the values of another array using shell sort
    
    Parameters
    ----------
    arr1 : list or np.ndarray of integers

    arr2 : list or np.ndarray of any type
    
    Returns
    -------
    list or np.ndarray
        same as input 
    '''
    
    # check that arr1 is array-like
    if not isinstance(arr1, (list, np.ndarray)):
        raise TypeError("arr1 must be a list or np.ndarray")

    # check that array of type list is an integer array
    if isinstance(arr1, list) and not all([isinstance(val, int) for val in arr1]):
        raise ValueError("All values in arr1 must be integers")

    # check that array of type np.ndarray is an integer array
    if isinstance(arr1, np.ndarray) and arr1.dtype != np.dtype('int'):
        raise ValueError("arr1 must be an integer array")

    # check that arr2 is array-like
    if not isinstance(arr2, (list, np.ndarray)):
        raise TypeError("arr2 must be a list or np.ndarray")

    # check that length of arr1 and arr2 are equal
    if len(arr1) != len(arr2):
        raise ValueError("arr1 and arr2 must be the same length")

    # set initial increment
    inc = 1

    # determine length of arr
    n = len(arr1)

    # scale increment based on length of array
    while True:
        inc = 3*inc + 1

        if inc > n:
            break
    
    # sort array
    while True:
        # rescale increment for subset of array
        inc = int(inc/3)
        
        # loop through subset of array and compare values
        for i in range(inc, n):
            va = arr1[i]
            ib = arr2[i]
            j = i
            
            # if array value at location j-inc is greater than location i 
            # insert larger value to larger index
            while arr1[j-inc] > va:
                arr1[j] = arr1[j-inc]
                arr2[j] = arr2[j-inc]
                j =  j-inc
                
                # exit loop when index j is less than or equal to current inc
                if j <= inc:
                    break
            
            arr1[j] = va
            arr2[j] = ib

        # exit loop if current inc is less than or equal to first index
        # in fortran arrays are 1-based whereas python arrays are 0-based
        if inc <= 0:
            break

    return arr1, arr2





    
        

# ReplaceString          
# FirstLocation          
# CountOccurance         
# FindSubStringInString  
# CleanSpecialCharacters 
# UpperCase              
# LowerCase              
# StripTextUntilCharacter
# GetStartLocation       
# LineFeed               
# LEN_TRIM_ARRAY         
# PrepareTitle           
# String_Copy_C_F        
# String_Copy_F_C        
# CString_Len            
# GenericString          
# GenericString_To_String
# String_To_GenericString
# AppendString_To_GenericString

# # Public array utilities
# AllocArray              
# AllocPointerToArray     
# LocateInList            
# NormalizeArray          
# ShellSort               
# GetUniqueArrayComponents
# GetArrayData                    
                                            
# # Public directory utilities               
# ConvertPathToWindowsStyle    
# ConvertPathToLinuxStyle      
# IsPathWindowsStyle           
# IsAbsolutePathname           
# StripFileNameFromPath        
# EstablishAbsolutePathFileName
# GetFileDirectory                
                                
# # Public misc. utilities                   
# GetDate                      
# GetTime                      
# Tolerance                    
# ConvertID_To_Index