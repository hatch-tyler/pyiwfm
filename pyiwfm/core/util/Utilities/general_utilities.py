'''
general_utilities.py
Author: Tyler Hatch PhD, PE

This is the general_utilities module of the python version of IWFM
'''
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
    pass




    
        

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